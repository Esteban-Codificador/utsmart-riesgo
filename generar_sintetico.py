"""Generador de datos sintéticos que cumplen contratos v2.

No es ruido: planta una señal conocida (susceptibilidad + detonante + sesgo de reporte +
imprecisión de fechas) y la guarda en verdad_sintetica.json, para que el pipeline
demuestre que recupera los parámetros antes de tocar datos reales.

Uso: python generar_sintetico.py --salida data/sintetico --anio-ini 2010 --anio-fin 2024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import contratos as C

LON0, LAT0 = -73.35, 6.95          # centro aproximado de Santander
SEMI_X_KM, SEMI_Y_KM = 95, 102     # elipse ~30.4k km², área real ~30.5k km²
N_MPIOS = 87

LITO = {"Q-al": -0.6, "N-sed": 0.3, "K-lut": 0.7, "J-ign": 0.0, "PZ-met": 0.5, "T-arc": 0.9}
CLC = {"bosque": -0.5, "pastos": 0.4, "cultivos": 0.2, "urbano": 0.1, "arbustal": 0.0}
VERDAD = {
    "b_pendiente_deg": 0.06, "b_twi_z": 0.35, "b_dist_drenaje_z": -0.25,
    "b_litologia": LITO, "b_cobertura": CLC,
    "b_log1p_p3d_mm": 0.9, "b_log1p_p30d_mm": 0.6,   # detonante satura: log(1+mm)
    "reporte_intercepto": 0.4, "reporte_b_dist_via_km": -0.35, "reporte_b_logpob_z": 0.35,
    "eventos_reales_objetivo": 7000,
    "precision_fecha_probs": {"dia": 0.55, "mes": 0.20, "anio": 0.18, "desconocida": 0.07},
}


def campo_suave(rng, nfil, ncol, escala_celdas):
    """Campo gaussiano suave z-normalizado vía filtro en Fourier."""
    ruido = rng.standard_normal((nfil, ncol))
    ky = np.fft.fftfreq(nfil)[:, None]
    kx = np.fft.fftfreq(ncol)[None, :]
    filtro = np.exp(-2 * (np.pi * escala_celdas) ** 2 * (kx ** 2 + ky ** 2))
    f = np.real(np.fft.ifft2(np.fft.fft2(ruido) * filtro))
    return (f - f.mean()) / f.std()


def zscore(v):
    return (v - v.mean()) / v.std()


def generar_celdas(rng):
    ncol, nfil = 2 * SEMI_X_KM + 2, 2 * SEMI_Y_KM + 2
    cols, filas = np.meshgrid(np.arange(ncol), np.arange(nfil))
    dx = cols - ncol / 2 + 0.5
    dy = filas - nfil / 2 + 0.5
    deformacion = 1 + 0.12 * campo_suave(rng, nfil, ncol, 40)   # borde irregular
    dentro = (dx / SEMI_X_KM) ** 2 + (dy / SEMI_Y_KM) ** 2 <= deformacion.clip(0.8, 1.2)

    elev = 1500 + 900 * campo_suave(rng, nfil, ncol, 15) + 300 * campo_suave(rng, nfil, ncol, 4)
    elev = elev.clip(100, 3900)
    gy, gx = np.gradient(elev, 1000.0)
    pend = np.degrees(np.arctan(np.hypot(gx, gy))) * 6 + 4 + 3 * rng.standard_normal(elev.shape)
    pend = pend.clip(0, 60)
    twi = 8 - 0.12 * pend + 1.2 * campo_suave(rng, nfil, ncol, 3)
    curv = 0.3 * campo_suave(rng, nfil, ncol, 2)
    dist_dren = np.exp(5.5 + 0.6 * campo_suave(rng, nfil, ncol, 3))
    precip_anual = 1800 + 600 * campo_suave(rng, nfil, ncol, 25)
    # vías por valles: más lejos en zonas altas y empinadas
    dist_via = np.exp(6.8 + 0.5 * zscore(elev) + 0.6 * campo_suave(rng, nfil, ncol, 6))
    pob = np.exp(5 - 0.6 * zscore(elev) + 0.8 * campo_suave(rng, nfil, ncol, 5))

    lito_idx = np.digitize(campo_suave(rng, nfil, ncol, 10), np.quantile(rng.standard_normal(10000), [.15, .35, .55, .75, .9]))
    clc_idx = np.digitize(campo_suave(rng, nfil, ncol, 6) - 0.3 * zscore(elev), [-1.0, -0.2, 0.5, 1.8])

    m = dentro.ravel()
    c, f = cols.ravel()[m], filas.ravel()[m]
    x_m = 4_900_000 + c * 1000.0 + 500
    y_m = 2_300_000 + f * 1000.0 + 500
    lon = LON0 + (c - ncol / 2) / 110.6 / np.cos(np.radians(LAT0))
    lat = LAT0 + (nfil / 2 - f) / 110.6

    semillas = rng.choice(len(c), N_MPIOS, replace=False)
    d2 = (c[:, None] - c[semillas][None]) ** 2 + (f[:, None] - f[semillas][None]) ** 2
    mpio = np.argmin(d2, axis=1)
    codigos = np.array([f"68{n:03d}" for n in sorted(rng.choice(np.arange(1, 900), N_MPIOS, replace=False))])

    borde = ~np.pad(dentro, 1)[1:-1, 2:] | ~np.pad(dentro, 1)[1:-1, :-2] | ~np.pad(dentro, 1)[2:, 1:-1] | ~np.pad(dentro, 1)[:-2, 1:-1]
    area = np.where(borde.ravel()[m], rng.uniform(0.3, 1.0, m.sum()), 1.0)

    def g(a):
        return a.ravel()[m].astype("float32")

    lito_cods, clc_cods = np.array(list(LITO)), np.array(list(CLC))
    return pd.DataFrame({
        "cell_id": [f"C{a:04d}_{b:04d}" for a, b in zip(c, f)],
        "pixel_lluvia_id": [f"P{a // 5:03d}_{b // 5:03d}" for a, b in zip(c, f)],
        "x_m": x_m, "y_m": y_m, "lon": lon, "lat": lat,
        "cod_dane_mpio": codigos[mpio],
        "area_km2": area.astype("float32"),
        "elev_media_m": g(elev), "pendiente_media_deg": g(pend),
        "pendiente_p90_deg": g(pend * 1.35 + 2), "twi_medio": g(twi),
        "curvatura_plan_media": g(curv), "litologia_cod": lito_cods[lito_idx.ravel()[m]],
        "cobertura_clc_cod": clc_cods[clc_idx.ravel()[m]], "dist_drenaje_m": g(dist_dren),
        "precip_media_anual_mm": g(precip_anual), "dist_via_m": g(dist_via), "poblacion": g(pob),
        "_col": c, "_fil": f,
    })


def generar_lluvia(rng, celdas, fechas):
    pix = celdas.groupby("pixel_lluvia_id")["precip_media_anual_mm"].mean()
    n_d, n_p = len(fechas), len(pix)
    doy = fechas.dayofyear.values
    # régimen bimodal andino: picos abr-may y oct-nov
    estacional = 1 + 0.55 * np.cos(2 * np.pi * (doy - 120) / 182.5)
    regional = np.zeros(n_d)
    e = rng.standard_normal(n_d)
    for t in range(1, n_d):
        regional[t] = 0.7 * regional[t - 1] + 0.7 * e[t]
    local = rng.standard_normal((n_d, n_p))
    humedo = (0.8 * regional[:, None] + 0.6 * local + 0.25 * (estacional[:, None] - 1) * 2) > 0.1
    media_dia = (pix.values[None, :] / 365.0) * estacional[:, None]
    p_humedo = humedo.mean()
    cantidad = rng.gamma(0.75, 1.0, (n_d, n_p)) / 0.75 * media_dia / p_humedo
    precip = np.where(humedo, np.minimum(cantidad, 300.0), 0.0).astype("float32")  # tope físico
    return pix.index.values, precip


def antecedentes(precip):
    """Matrices (días x píxeles) de acumulados antecedentes incluyendo el día t."""
    cs = np.vstack([np.zeros((1, precip.shape[1])), np.cumsum(precip, axis=0, dtype="float64")])
    out = {}
    for k in [1, 3, 7, 15, 30]:
        idx = np.arange(precip.shape[0])
        out[f"p{k}d_mm"] = (cs[idx + 1] - cs[np.maximum(idx + 1 - k, 0)]).astype("float32")
    api = np.zeros_like(precip, dtype="float64")
    for i in range(30):
        api[i:] += (0.9 ** i) * precip[: precip.shape[0] - i]
    out["api_k090"] = api.astype("float32")
    return out


def generar_eventos(rng, celdas, pix_ids, fechas, ant):
    lito = celdas["litologia_cod"].map(LITO).values
    clc = celdas["cobertura_clc_cod"].map(CLC).values
    log_lam = (VERDAD["b_pendiente_deg"] * celdas["pendiente_media_deg"].values
               + VERDAD["b_twi_z"] * zscore(celdas["twi_medio"].values)
               + VERDAD["b_dist_drenaje_z"] * zscore(np.log(celdas["dist_drenaje_m"].values))
               + lito + clc + np.log(celdas["area_km2"].values))
    lam = np.exp(log_lam - log_lam.max())
    pix_idx = pd.Index(pix_ids).get_indexer(celdas["pixel_lluvia_id"])
    lam_pix = np.bincount(pix_idx, weights=lam, minlength=len(pix_ids))

    inicio = 30  # el detonante necesita 30 días de lluvia previa
    rr = np.exp(VERDAD["b_log1p_p3d_mm"] * np.log1p(ant["p3d_mm"][inicio:])
                + VERDAD["b_log1p_p30d_mm"] * np.log1p(ant["p30d_mm"][inicio:]))
    tasa = rr * lam_pix[None, :]
    escala = VERDAD["eventos_reales_objetivo"] / tasa.sum()
    conteos = rng.poisson(tasa * escala)
    d_idx, p_idx = np.nonzero(conteos)
    reps = conteos[d_idx, p_idx]
    d_idx, p_idx = np.repeat(d_idx, reps) + inicio, np.repeat(p_idx, reps)

    # celda dentro del píxel, proporcional a lambda
    orden = np.argsort(pix_idx, kind="stable")
    lam_ord, pix_ord = lam[orden], pix_idx[orden]
    limites = np.searchsorted(pix_ord, np.arange(len(pix_ids) + 1))
    celda_ev = np.empty(len(p_idx), dtype=int)
    for i, p in enumerate(p_idx):
        a, b = limites[p], limites[p + 1]
        w = lam_ord[a:b]
        celda_ev[i] = orden[a + rng.choice(b - a, p=w / w.sum())]

    # sesgo de reporte: solo se registra lo que alguien ve y reporta
    dv = celdas["dist_via_m"].values[celda_ev] / 1000
    lp = zscore(np.log(celdas["poblacion"].values))[celda_ev]
    p_rep = 1 / (1 + np.exp(-(VERDAD["reporte_intercepto"] + VERDAD["reporte_b_dist_via_km"] * dv
                              + VERDAD["reporte_b_logpob_z"] * lp)))
    rep = rng.random(len(celda_ev)) < p_rep
    celda_ev, d_idx = celda_ev[rep], d_idx[rep]
    n = len(celda_ev)

    fecha_real = fechas.values[d_idx]
    probs = VERDAD["precision_fecha_probs"]
    prec = rng.choice(list(probs), n, p=list(probs.values()))
    fr = pd.to_datetime(fecha_real)
    fecha_ev = pd.Series(fr)
    fecha_ev[prec == "mes"] = fr[prec == "mes"].to_period("M").to_timestamp()
    fecha_ev[prec == "anio"] = fr[prec == "anio"].to_period("Y").to_timestamp()  # trampa real: 01-ene
    fecha_ev[prec == "desconocida"] = pd.NaT
    lag = rng.geometric(1 / 20, n)
    fecha_rep = pd.Series(fr + pd.to_timedelta(lag, "D"))
    fecha_rep[rng.random(n) < 0.1] = pd.NaT

    prec_esp = rng.choice([100, 500, 1000, 5000], n, p=[.2, .4, .3, .1]).astype("float32")
    jit = rng.standard_normal((n, 2)) * (prec_esp[:, None] / 2) / 1000  # km
    c0, f0 = celdas["_col"].values[celda_ev], celdas["_fil"].values[celda_ev]
    c1, f1 = np.rint(c0 + jit[:, 0]).astype(int), np.rint(f0 + jit[:, 1]).astype(int)
    ids_validos = set(celdas["cell_id"])
    cand = [f"C{a:04d}_{b:04d}" for a, b in zip(c1, f1)]
    cell_obs = np.where([x in ids_validos for x in cand], cand, celdas["cell_id"].values[celda_ev])
    lon = celdas["lon"].values[celda_ev] + jit[:, 0] / 110.6 / np.cos(np.radians(LAT0))
    lat = celdas["lat"].values[celda_ev] - jit[:, 1] / 110.6

    tipos = rng.choice(["deslizamiento", "caida", "flujo", "reptacion", "otro"], n, p=[.65, .15, .12, .03, .05])
    ev = pd.DataFrame({
        "evento_id": [f"EV{i:06d}" for i in range(n)],
        "simma_id": np.where(rng.random(n) < 0.95, [str(1_000_000 + i) for i in rng.permutation(n)], None),
        "tipo_movimiento": tipos,
        "fecha_evento": fecha_ev.dt.date.values,
        "precision_fecha": prec,
        "fecha_reporte": fecha_rep.dt.date.values,
        "lon": lon, "lat": lat, "precision_espacial_m": prec_esp,
        "cell_id": cell_obs,
    })
    ev["fecha_evento"] = ev["fecha_evento"].where(ev["precision_fecha"] != "desconocida", None)
    ev["fecha_reporte"] = ev["fecha_reporte"].where(pd.notna(ev["fecha_reporte"]), None)
    f_min = fechas[30].date()
    ev["usable_temporal"] = (ev["precision_fecha"] == "dia") & ev["fecha_evento"].map(
        lambda d: d is not None and pd.notna(d) and d >= f_min)
    return ev, int(rep.size), int(rep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="data/sintetico")
    ap.add_argument("--anio-ini", type=int, default=2010)
    ap.add_argument("--anio-fin", type=int, default=2024)
    ap.add_argument("--semilla", type=int, default=2027)
    a = ap.parse_args()
    rng = np.random.default_rng(a.semilla)
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)

    celdas = generar_celdas(rng)
    fechas = pd.date_range(f"{a.anio_ini - 1}-12-01", f"{a.anio_fin}-12-31", freq="D")
    pix_ids, precip = generar_lluvia(rng, celdas, fechas)
    ant = antecedentes(precip)
    eventos, n_reales, n_rep = generar_eventos(rng, celdas, pix_ids, fechas, ant)

    lluvia = pd.DataFrame({
        "pixel_lluvia_id": np.tile(pix_ids, len(fechas)),
        "fecha": np.repeat(fechas.date, len(pix_ids)),
        "precip_mm": precip.ravel(),
        "fuente": "sintetico",
    })
    C.escribir("celdas", celdas.drop(columns=["_col", "_fil"]), out / "celdas.parquet")
    C.escribir("eventos", eventos, out / "eventos.parquet")
    C.escribir("lluvia_diaria", lluvia, out / "lluvia_diaria.parquet")

    resumen = {**VERDAD, "n_celdas": len(celdas), "n_pixeles": len(pix_ids), "n_dias": len(fechas),
               "eventos_reales": n_reales, "eventos_reportados": n_rep,
               "eventos_usables_temporal": int(eventos["usable_temporal"].sum()),
               "contrato": C.VERSION_CONTRATO, "semilla": a.semilla}
    (out / "verdad_sintetica.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps({k: resumen[k] for k in ["n_celdas", "n_pixeles", "n_dias", "eventos_reales",
                                              "eventos_reportados", "eventos_usables_temporal"]}))


if __name__ == "__main__":
    main()
