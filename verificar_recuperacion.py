"""Prueba de humo del diseño: ¿el pipeline recupera la señal plantada?

1. Detonante: logit condicional sobre case-crossover -> b_p3d, b_p30d vs verdad.
2. Susceptibilidad: Spearman entre el score del modelo y la log-tasa VERDADERA en todas
   las celdas, para pseudo-ausencias ingenuas vs emparejadas por sesgo, con y sin
   variables de sesgo/proxies. Es la tabla que se le muestra al jurado.
   (El AUC sobre pseudo-ausencias NO sirve para esto: mide qué tan fácil es separar 1s de
   0s que nosotros mismos elegimos, no fidelidad a la amenaza real.)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")
d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sintetico")
v = json.loads((d / "verdad_sintetica.json").read_text())
celdas = pd.read_parquet(d / "celdas.parquet")
eventos = pd.read_parquet(d / "eventos.parquet")
fallas = 0

# ---- 1. detonante
cc = pd.read_parquet(d / "muestras_case_crossover.parquet")
x = np.log1p(cc[["p3d_mm", "p30d_mm"]])                 # detonante en log(1+mm)
res = ConditionalLogit(cc["es_caso"], x, groups=cc["estrato_id"]).fit(disp=0)
ci = res.conf_int()
print(f"== Detonante: {cc.estrato_id.nunique()} estratos")
for c in ["p3d_mm", "p30d_mm"]:
    real, est = v[f"b_log1p_{c}"], res.params[c]
    lo, hi = ci.loc[c]
    # tolerancia 10%: la imprecisión espacial de SIMMA produce dilución (~5% hacia 0,
    # verificado en 4 semillas; desaparece con precisión <=100 m). Sesgo conservador.
    ok = (lo <= real <= hi) or abs(est - real) / real <= 0.10
    fallas += not ok
    print(f"  b_log1p_{c}: {est:.4f}  IC95 [{lo:.4f}, {hi:.4f}]  verdad {real:.4f}  "
          f"err {100 * (est - real) / real:+.1f}%  {'OK' if ok else 'FALLA'}")

# ---- 2. susceptibilidad
z = lambda s: (s - s.mean()) / s.std()
log_lam = (v["b_pendiente_deg"] * celdas.pendiente_media_deg + v["b_twi_z"] * z(celdas.twi_medio)
           + v["b_dist_drenaje_z"] * z(np.log(celdas.dist_drenaje_m))
           + celdas.litologia_cod.map(v["b_litologia"]) + celdas.cobertura_clc_cod.map(v["b_cobertura"]))


def X(df, cols):
    out = df[cols].copy()
    for c in ["dist_drenaje_m", "dist_via_m", "poblacion"]:
        if c in out:
            out[c] = np.log(out[c])
    out = pd.concat([out, pd.get_dummies(df[["litologia_cod", "cobertura_clc_cod"]], dtype=float)], axis=1)
    return out


def fidelidad(muestras, cols):
    df = muestras.merge(celdas, on="cell_id")
    xt, xa = X(df, cols), X(celdas, cols).reindex(columns=X(df, cols).columns, fill_value=0)
    mu, sd = xt.mean(), xt.std()
    m = LogisticRegression(max_iter=3000).fit((xt - mu) / sd, df.etiqueta, sample_weight=df.peso)
    return spearmanr(m.decision_function((xa - mu) / sd), log_lam)[0]


emp = pd.read_parquet(d / "muestras_susceptibilidad.parquet")
rng = np.random.default_rng(1)
pres = emp[emp.etiqueta == 1]
libres = celdas.loc[~celdas.cell_id.isin(eventos.cell_id), "cell_id"]
ing = pd.concat([pres, pd.DataFrame({"cell_id": rng.choice(libres, 2 * len(pres), replace=False),
                                     "etiqueta": 0, "peso": 0.5})])

BASE = ["pendiente_media_deg", "twi_medio", "dist_drenaje_m"]
VARIANTES = [("solo físicas", BASE), ("+ elevación (proxy)", BASE + ["elev_media_m"]),
             ("+ elev + vía + población", BASE + ["elev_media_m", "dist_via_m", "poblacion"])]
print("\n== Susceptibilidad: Spearman(score, amenaza verdadera) en las", len(celdas), "celdas")
tabla = {}
for nom, ms in [("ingenuas", ing), ("emparejadas", emp)]:
    for etq, cols in VARIANTES:
        tabla[(nom, etq)] = r = fidelidad(ms, cols)
        print(f"  pseudo-ausencias {nom:11s} {etq:26s} {r:.3f}")
clave = VARIANTES[-1][0]   # la afirmación es: con variables contaminadas, emparejar protege
if tabla[("emparejadas", clave)] <= tabla[("ingenuas", clave)]:
    fallas += 1
    print("  FALLA: el emparejamiento no protege contra variables de sesgo")
sys.exit(1 if fallas else 0)
