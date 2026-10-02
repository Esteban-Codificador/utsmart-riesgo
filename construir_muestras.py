"""Construye las tablas de entrenamiento a partir de celdas, eventos y lluvia (contratos v2).

Modelo A — susceptibilidad (estático, nivel celda):
  presencias = celdas con >=1 evento (cualquier precisión de fecha; solo importa el dónde).
  pseudo-ausencias = celdas sin evento, fuera de un buffer de 1 km de cualquier presencia,
  muestreadas con la MISMA distribución de dist_via_m que las presencias (fondo de grupo
  objetivo). Así el sesgo de reporte afecta igual a 1s y 0s y el modelo no aprende "cerca
  de vía = deslizamiento".

Modelo B — detonante (case-crossover estratificado en el tiempo):
  por cada evento con fecha diaria, el día del evento es el caso y los demás días del mismo
  mes-año en la misma celda son controles. Lo estático se cancela dentro del estrato, el
  sesgo de reporte espacial también; solo queda el efecto de la lluvia.

Uso: python construir_muestras.py --datos data/sintetico
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import contratos as C
from generar_sintetico import antecedentes

RAZON_AUSENCIAS = 2       # pseudo-ausencias por presencia
BUFFER_CELDAS = 1         # 1 km: 2 km cubría ~90% del depto con presencias densas (verificado)
BLOQUE_KM = 20            # bloques para CV espacial
N_FOLDS = 5
MAX_PRECISION_ESPACIAL_M = 1000   # solo para case-crossover


def muestras_susceptibilidad(celdas, eventos, rng):
    col = ((celdas["x_m"] - celdas["x_m"].min()) // 1000).astype(int).values
    fil = ((celdas["y_m"] - celdas["y_m"].min()) // 1000).astype(int).values
    pres = celdas["cell_id"].isin(eventos["cell_id"]).values

    malla = np.zeros((fil.max() + 1 + 2 * BUFFER_CELDAS, col.max() + 1 + 2 * BUFFER_CELDAS), bool)
    malla[fil[pres] + BUFFER_CELDAS, col[pres] + BUFFER_CELDAS] = True
    cerca = np.zeros_like(malla)
    for dy in range(-BUFFER_CELDAS, BUFFER_CELDAS + 1):
        for dx in range(-BUFFER_CELDAS, BUFFER_CELDAS + 1):
            cerca |= np.roll(np.roll(malla, dy, 0), dx, 1)
    elegible = ~pres & ~cerca[fil + BUFFER_CELDAS, col + BUFFER_CELDAS]

    # emparejar distribución de dist_via_m por deciles de las presencias
    cortes = np.quantile(celdas.loc[pres, "dist_via_m"], np.linspace(0, 1, 11)[1:-1])
    dec = np.digitize(celdas["dist_via_m"].values, cortes)
    objetivo = np.bincount(dec[pres], minlength=10) * RAZON_AUSENCIAS
    elegidas = []
    for d in range(10):
        pool = np.flatnonzero(elegible & (dec == d))
        k = min(objetivo[d], len(pool))
        if k < objetivo[d]:
            print(f"aviso: decil {d} de dist_via con {len(pool)} candidatas para {objetivo[d]}")
        elegidas.append(rng.choice(pool, k, replace=False))
    aus = np.concatenate(elegidas)

    idx = np.concatenate([np.flatnonzero(pres), aus])
    bloque = (col[idx] // BLOQUE_KM) * 1000 + (fil[idx] // BLOQUE_KM)
    bloques = np.unique(bloque)
    fold_de = dict(zip(bloques, rng.permutation(len(bloques)) % N_FOLDS))
    n_p = int(pres.sum())
    return pd.DataFrame({
        "cell_id": celdas["cell_id"].values[idx],
        "etiqueta": np.r_[np.ones(n_p), np.zeros(len(aus))].astype("int8"),
        "estrategia": ["presencia"] * n_p + ["pseudoausencia_buffer1km_sesgo_emparejado"] * len(aus),
        "bloque_espacial": np.array([fold_de[b] for b in bloque], dtype="int16"),
        "peso": np.r_[np.ones(n_p), np.full(len(aus), 1 / RAZON_AUSENCIAS)].astype("float32"),
    })


def muestras_case_crossover(celdas, eventos, lluvia):
    # imprecisión espacial > tamaño de celda manda el evento al píxel de lluvia equivocado
    # y atenúa el efecto de corto plazo (verificado: b_p3d -8% incluyendo ±5 km).
    # Nulos se aceptan (supuesto: SIMMA sin precisión declarada ~ precisión de vereda).
    prec_ok = eventos["precision_espacial_m"].isna() | (eventos["precision_espacial_m"] <= MAX_PRECISION_ESPACIAL_M)
    ev = eventos[eventos["usable_temporal"] & prec_ok].merge(celdas[["cell_id", "pixel_lluvia_id"]], on="cell_id")
    ev["fecha_evento"] = pd.to_datetime(ev["fecha_evento"])
    # NO deduplicar por celda-día: varios eventos el mismo día son varios casos, y se
    # concentran en días de lluvia fuerte; quitarlos atenúa el efecto. Los duplicados de
    # registro (mismo evento reportado dos veces) se limpian antes, al ingerir SIMMA.

    piv = lluvia.pivot(index="fecha", columns="pixel_lluvia_id", values="precip_mm").sort_index()
    piv.index = pd.to_datetime(piv.index)
    ant = antecedentes(piv.values)
    pix_pos = pd.Index(piv.columns)
    f_pos = pd.Index(piv.index)

    filas = []
    for e in ev.itertuples():
        mes = pd.date_range(e.fecha_evento.to_period("M").to_timestamp(),
                            e.fecha_evento.to_period("M").to_timestamp(how="end").normalize())
        filas.append(pd.DataFrame({"estrato_id": e.evento_id, "cell_id": e.cell_id,
                                   "pixel_lluvia_id": e.pixel_lluvia_id, "fecha": mes,
                                   "es_caso": (mes == e.fecha_evento).astype("int8")}))
    cc = pd.concat(filas, ignore_index=True)
    ti = f_pos.get_indexer(cc["fecha"])
    pi = pix_pos.get_indexer(cc["pixel_lluvia_id"])
    cc = cc[ti >= 29].reset_index(drop=True)
    ti, pi = ti[ti >= 29], pi[ti >= 29]
    for k, m in ant.items():
        cc[k] = m[ti, pi]
    cc["fecha"] = cc["fecha"].dt.date
    return cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="data/sintetico")
    ap.add_argument("--semilla", type=int, default=7)
    a = ap.parse_args()
    d = Path(a.datos)
    rng = np.random.default_rng(a.semilla)
    celdas = pd.read_parquet(d / "celdas.parquet")
    eventos = pd.read_parquet(d / "eventos.parquet")
    lluvia = pd.read_parquet(d / "lluvia_diaria.parquet")

    for nombre, df in [("celdas", celdas), ("eventos", eventos), ("lluvia_diaria", lluvia)]:
        err = C.validar(nombre, df, celdas)
        if err:
            raise SystemExit("\n".join(err))

    ms = muestras_susceptibilidad(celdas, eventos, rng)
    cc = muestras_case_crossover(celdas, eventos, lluvia)
    C.escribir("muestras_susceptibilidad", ms, d / "muestras_susceptibilidad.parquet")
    C.escribir("muestras_case_crossover", cc, d / "muestras_case_crossover.parquet")
    print(f"susceptibilidad: {int(ms.etiqueta.sum())} presencias, {int((ms.etiqueta == 0).sum())} pseudo-ausencias")
    print(f"case-crossover: {cc.estrato_id.nunique()} estratos, {len(cc)} filas")


if __name__ == "__main__":
    main()
