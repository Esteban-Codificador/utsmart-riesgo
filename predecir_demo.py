"""Produce predicciones.parquet (contrato v2) para una fecha histórica: insumo del frontend.

Combinación (por qué no es "susceptibilidad x detonante" a ojo):
  tasa_diaria(celda, t) = c * exp(logit_susc(celda)) * RR(t)
  - exp(logit) ~ tasa relativa de la celda (diseño caso-control, eventos raros).
  - RR(t) = exp(b·log(1+x_t)) del case-crossover, relativo al día típico del mismo mes.
  - c se calibra para que la suma histórica de tasas = eventos reportados observados.
  => tasa_diaria es P(evento REPORTADO en la celda ese día). Decirlo así ante el jurado.
Incertidumbre:
  - susceptibilidad: Venn-ABERS inductivo (intervalo [p0, p1] por celda).
  - RR: 2000 draws de b ~ N(b_hat, Cov) -> percentiles 2.5/97.5.
  - tasa: combinación conservadora de extremos.

Uso: python predecir_demo.py --datos data/sintetico [--fecha 2017-04-20]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.conditional_models import ConditionalLogit

import contratos as C
from generar_sintetico import antecedentes

warnings.filterwarnings("ignore")
VERSION = "demo-sintetico-0.1"
FISICAS = ["pendiente_media_deg", "twi_medio", "dist_drenaje_m"]
DETONANTE = ["p3d_mm", "p30d_mm"]


def X(df):
    out = df[FISICAS].copy()
    out["dist_drenaje_m"] = np.log(out["dist_drenaje_m"])
    return pd.concat([out, pd.get_dummies(df[["litologia_cod", "cobertura_clc_cod"]], dtype=float)], axis=1)


def venn_abers(s_cal, y_cal, s_test):
    """IVAP: para cada intervalo entre scores de calibración, p0/p1 con isotónica."""
    grid = np.unique(s_cal)
    puntos = np.r_[grid[0] - 1, (grid[:-1] + grid[1:]) / 2, grid[-1] + 1]
    p0, p1 = np.empty(len(puntos)), np.empty(len(puntos))
    for i, s in enumerate(puntos):
        xs = np.r_[s_cal, s]
        for etq, dest in [(0, p0), (1, p1)]:
            iso = IsotonicRegression(out_of_bounds="clip").fit(xs, np.r_[y_cal, etq])
            dest[i] = iso.predict([s])[0]
    pos = np.searchsorted(grid, s_test)          # intervalo de cada score de prueba
    lo, hi = p0[pos], p1[pos]
    return lo, hi, hi / (1 - lo + hi)            # p combinado (Vovk et al.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="data/sintetico")
    ap.add_argument("--fecha", default=None, help="por defecto: el día histórico con más eventos")
    a = ap.parse_args()
    d = Path(a.datos)
    rng = np.random.default_rng(0)
    celdas = pd.read_parquet(d / "celdas.parquet")
    eventos = pd.read_parquet(d / "eventos.parquet")
    lluvia = pd.read_parquet(d / "lluvia_diaria.parquet")
    ms = pd.read_parquet(d / "muestras_susceptibilidad.parquet").merge(celdas, on="cell_id")
    cc = pd.read_parquet(d / "muestras_case_crossover.parquet")

    # --- susceptibilidad: entrena en folds 0-3, calibra en fold 4 (bloques espaciales)
    xm, xa = X(ms), X(celdas).reindex(columns=X(ms).columns, fill_value=0)
    mu, sd = xm.mean(), xm.std()
    tr, cal = ms.bloque_espacial != 4, ms.bloque_espacial == 4
    lr = LogisticRegression(max_iter=3000).fit((xm[tr] - mu) / sd, ms.etiqueta[tr], sample_weight=ms.peso[tr])
    s_cal = lr.decision_function((xm[cal] - mu) / sd)
    logit_all = lr.decision_function((xa - mu) / sd)
    s_inf, s_sup, s_p = venn_abers(s_cal, ms.etiqueta[cal].values, logit_all)

    # --- detonante
    res = ConditionalLogit(cc.es_caso, np.log1p(cc[DETONANTE]), groups=cc.estrato_id).fit(disp=0)
    b = res.params.values
    draws = rng.multivariate_normal(b, res.cov_params().values, 2000)

    piv = lluvia.pivot(index="fecha", columns="pixel_lluvia_id", values="precip_mm").sort_index()
    ant = {k: np.log1p(m) for k, m in antecedentes(piv.values).items()}  # detonante en log(1+mm)
    fechas = pd.to_datetime(piv.index)
    if a.fecha:
        f_obj = pd.Timestamp(a.fecha)
    else:
        f_obj = pd.to_datetime(eventos.loc[eventos.usable_temporal, "fecha_evento"]).value_counts().idxmax()
    t = fechas.get_loc(f_obj)
    pix = pd.Index(piv.columns).get_indexer(celdas.pixel_lluvia_id)
    x_t = np.column_stack([ant[k][t, pix] for k in DETONANTE])

    # RR relativo al día típico del mes (lo que estima el case-crossover)
    mes = fechas.month == f_obj.month
    x_ref = np.column_stack([ant[k][mes][30:].mean(axis=0)[pix] for k in DETONANTE])
    dx = x_t - x_ref
    rr = np.exp(dx @ b)
    rr_draws = np.exp(dx @ draws.T)
    rr_inf, rr_sup = np.percentile(rr_draws, [2.5, 97.5], axis=1)

    # --- escala absoluta: suma histórica de tasas = eventos reportados con fecha útil por día
    lam_rel = np.exp(logit_all)
    hist_rr = np.exp(np.stack([ant[k][30:] for k in DETONANTE], -1) @ b)        # días x píxeles
    lam_pix = np.bincount(pix, weights=lam_rel, minlength=piv.shape[1])
    rr_mes_medio = np.exp(x_ref @ b)  # renormaliza porque rr está centrado en el mes
    c = len(eventos) / (hist_rr * lam_pix).sum()
    tasa = c * lam_rel * rr * rr_mes_medio
    odds = lambda p: p / (1 - p)
    f_inf = odds(np.clip(s_inf, 1e-4, 1 - 1e-4)) / odds(np.clip(s_p, 1e-4, 1 - 1e-4))
    f_sup = odds(np.clip(s_sup, 1e-4, 1 - 1e-4)) / odds(np.clip(s_p, 1e-4, 1 - 1e-4))
    tasa_inf = tasa * np.minimum(f_inf, 1) * np.minimum(rr_inf / rr, 1)
    tasa_sup = tasa * np.maximum(f_sup, 1) * np.maximum(rr_sup / rr, 1)

    # categorías con umbrales fijos sobre climatología (no sobre el día: si todo está bajo, se ve bajo)
    muestra_dias = rng.choice(hist_rr.shape[0], 365, replace=False)
    clim = (c * lam_rel[None, :] * hist_rr[muestra_dias][:, pix]).ravel()
    umbrales = np.quantile(clim, [0.50, 0.80, 0.95])
    cat = np.array(C.CATEGORIA_RIESGO)[np.digitize(tasa, umbrales)]

    pred = pd.DataFrame({
        "cell_id": celdas.cell_id, "fecha_objetivo": f_obj.date(),
        "fecha_ultimo_dato_lluvia": f_obj.date(),   # histórico: sin rezago. En vivo: < fecha_objetivo
        "susceptibilidad": s_p, "susceptibilidad_inf": np.minimum(s_inf, s_p),
        "susceptibilidad_sup": np.maximum(s_sup, s_p),
        "rr_lluvia": rr, "rr_lluvia_inf": np.minimum(rr_inf, rr), "rr_lluvia_sup": np.maximum(rr_sup, rr),
        "tasa_diaria": tasa, "tasa_diaria_inf": tasa_inf, "tasa_diaria_sup": tasa_sup,
        "categoria": cat, "version_modelo": VERSION,
    })
    for col in pred.columns[3:-2]:
        pred[col] = pred[col].astype("float32")
    C.escribir("predicciones", pred, d / "predicciones.parquet")

    obs = eventos[pd.to_datetime(eventos.fecha_evento) == f_obj]
    p_obs = pred.set_index("cell_id").loc[obs.cell_id]
    print(f"fecha {f_obj.date()}: {len(obs)} eventos observados, {pred.tasa_diaria.sum():.1f} esperados")
    print("categoría de celdas con evento:", p_obs.categoria.value_counts().to_dict())
    print("distribución general:", pred.categoria.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
