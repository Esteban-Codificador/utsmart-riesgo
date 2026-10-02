# Contratos de datos v2 — UTSmart 2027 · Movimientos en masa, Santander

Versión `2.1.0` (2.1: + `eventos_crudos`, staging de ingesta SIMMA) · Fuente de verdad ejecutable: `contratos.py` (esquemas pyarrow + `validar()`).
Si este documento y `contratos.py` difieren, manda `contratos.py`.

## Decisiones congeladas

| # | Decisión | Por qué (evidencia en sintético, `verificar_recuperacion.py`) |
|---|---|---|
| D1 | Unidad espacial: **celda regular de 1 km** en EPSG:9377 (~31k celdas). Veredas solo como atributo para agregación en el frontend. | Veredas tienen áreas de 0.1 a >50 km²: mezcla tamaño con riesgo. Malla regular = covariables comparables y CV espacial limpia. |
| D2 | Lluvia se guarda **por píxel CHIRPS (~5 km)**, no por celda. `celdas.pixel_lluvia_id` es la FK. | La celda no agrega resolución de lluvia; replicarla a 31k celdas × 9k días = 280M filas inútiles. Píxel: ~7.5M filas, 24 MB. |
| D3 | **Dos modelos, no una tabla `evento 0/1`.** A = susceptibilidad estática (celda). B = detonante por case-crossover (celda-día). | Una tabla celda-día única tiene ~110M filas y 99.99% ceros. Case-crossover cancela todo lo estático y el sesgo espacial de reporte dentro del estrato. |
| D4 | Pseudo-ausencias: celdas sin evento, **fuera de buffer de 1 km**, muestreadas con la **misma distribución de `dist_via_m` que las presencias** (fondo de grupo objetivo), razón 1:2, peso 0.5. | Con variables contaminadas, fidelidad a la amenaza real 0.959 vs 0.943 (ingenuas); ganó 4/4 semillas. Buffer de 2 km era dañino (0.94 vs 0.97): con presencias densas cubre ~90% del depto. |
| D5 | `dist_via_m` y `poblacion` se entregan con **uso = `sesgo`**: no entran al modelo, sí a diagnóstico y al muestreo. | Son mayormente "dónde alguien vio y reportó". Matiz para el jurado: los cortes de talud viales sí causan deslizamientos; por eso se muestra modelo con/sin. |
| D6 | Detonante en **log(1 + mm)** acumulado (3 y 30 días). Nunca lineal en mm. | Lineal en mm: un píxel con 556 mm produjo e^18 de riesgo relativo y 22% de todos los eventos en un día. log1p satura como los umbrales intensidad-duración. |
| D7 | Case-crossover **estratificado por mes-año**, todos los días del mes como controles; solo eventos con `precision_fecha = dia` y `precision_espacial_m ≤ 1000` (o nula). | Recupera β plantado con error ≤12%. Precisión espacial ±5 km atenúa β_p3d un 8% adicional. Residual ~5% de dilución conservadora (desaparece con ≤100 m). |
| D8 | **No deduplicar eventos por celda-día.** Solo deduplicar registros repetidos del mismo evento (al ingerir SIMMA). | Deduplicar celda-día eliminaba 58% de los casos, justo los de días de tormenta → subestima el efecto de la lluvia. |
| D9 | Salida = **tasa diaria de evento reportado** = `c · exp(logit_susc) · RR(t)`, con `c` calibrado al conteo histórico. No "susceptibilidad × lluvia". | Es una tasa de Poisson coherente; el producto de dos scores no es probabilidad de nada. Se dice "reportado" porque SIMMA subregistra. |
| D10 | Incertidumbre: **Venn-ABERS** inductivo para susceptibilidad (fold espacial 4 = calibración); IC del RR por 2000 draws de β; tasa con combinación conservadora de extremos. | MAPIE da conjuntos de etiquetas, no intervalos de probabilidad (issue #1 de la revisión). |
| D11 | Categorías con **umbrales fijos de climatología** (p50/p80/p95 de la tasa histórica), no relativos al día. | Si el día es seco, el mapa debe verse bajo. Umbrales relativos al día siempre pintan 5% rojo. |
| D12 | `fecha_ultimo_dato_lluvia` explícita en cada predicción. Demo del jurado con **fechas históricas**. | CHIRPS final tiene ~3 semanas de rezago; prelim ~2 días. Mostrar "hoy" con datos de hace 3 semanas es un flanco abierto. |

## Tablas

| Tabla | Clave | Filas (Santander, 2010–2024) | Productor → Consumidor |
|---|---|---|---|
| `celdas` | `cell_id` | ~31k | ingesta geo → todos |
| `eventos_crudos` | `evento_id` | ~4.4k (Catálogo 3.919 + Inventario 469) | `ingesta/` → geolocalización |
| `eventos` | `evento_id` | los de `eventos_crudos` con ubicación | geolocalización → muestras |
| `lluvia_diaria` | `pixel_lluvia_id, fecha` | ~7.5M | ingesta CHIRPS → muestras, predicción |
| `muestras_susceptibilidad` | `cell_id` | ~9k | `construir_muestras.py` → modelo A |
| `muestras_case_crossover` | `estrato_id, fecha` | ~50k | `construir_muestras.py` → modelo B |
| `predicciones` | `cell_id, fecha_objetivo` | 31k por fecha | modelo → frontend |

Columnas, tipos y nulabilidad: ver `contratos.py`. Reglas que `validar()` hace cumplir además de tipos:
claves únicas; FK `cell_id` a `celdas`; `precision_fecha ∈ {dia, mes, anio, desconocida}`;
`usable_temporal ⇒ precision_fecha = dia`; lluvia ≥ 0; exactamente 1 caso por estrato;
`inf ≤ punto ≤ sup` en los tres pares de intervalos; `fecha_ultimo_dato_lluvia ≤ fecha_objetivo`.

### Trampa de SIMMA que el contrato obliga a resolver al ingerir
Registros con solo año se guardan como `AAAA-01-01` y con solo mes como `AAAA-MM-01`. Si la ingesta
no llena `precision_fecha`, el 1 de enero se vuelve el día más peligroso del año. Regla: fecha que cae
en día 1 sin evidencia de precisión diaria → `mes` o `anio`, nunca `dia`.

## Cómo correr

```bash
pip install numpy pandas pyarrow scipy scikit-learn statsmodels
python generar_sintetico.py        # 15 s · data/sintetico/{celdas,eventos,lluvia_diaria}.parquet + verdad
python construir_muestras.py       # 10 s · muestras_susceptibilidad + muestras_case_crossover
python verificar_recuperacion.py   #  5 s · exit 0 = el diseño recupera la señal plantada
python predecir_demo.py            # 20 s · predicciones.parquet para el frontend
```

Cambiar a datos reales = reemplazar los tres parquet crudos con la misma forma; los otros tres scripts no cambian.

## Supuestos marcados (verificar con datos reales)

- **S1 — CERRADO (2 oct):** Catálogo Santander 3.919 registros con fecha, 3.552 desde 1981: hay estratos suficientes sin ampliar a otros departamentos. Ojo: dd/mm/aaaa no prueba precisión diaria; en el Inventario 14% de las fechas son día 1 (esperado 3%) y se reclasifican a `mes`. Ver CLAUDE.md.
- **S2** `precision_espacial_m` nula se trata como ≤1 km. Si SIMMA reporta mayoritariamente a nivel municipio, el modelo B se degrada a nivel municipio.
- **S3 — CERRADO:** ~4.4k registros SIMMA en Santander (Catálogo + Inventario).
- **S4** Coordenadas lon/lat del sintético son aproximadas (no reproyección real); irrelevante para el pipeline, no usar para mapas.

## Pendiente técnico conocido

- La calibración absoluta de `tasa_diaria` falta validarla día a día (observado vs esperado por día y por mes). En el día histórico más extremo del sintético se esperan 2.0 y se observan 12: parte es regresión a la media por elegir el máximo, pero hay que medirlo.
- Venn-ABERS calibra contra la prevalencia del diseño 1:2, no contra la real. Por eso la probabilidad absoluta sale de D9, no de la susceptibilidad.
