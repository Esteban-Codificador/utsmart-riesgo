# UTSmart 2027 — Riesgo de movimientos en masa, Santander

Hackathon UTSmart 2027. Predicción de tasa diaria de movimientos en masa por celda de 1 km en
Santander (Colombia): susceptibilidad estática × detonante de lluvia (case-crossover).

## Reglas del repo
- `contratos.py` es la fuente de verdad de esquemas (v2.1.0). Todo Parquet se escribe con
  `contratos.escribir()`, que valida antes de escribir. `CONTRATOS_v2.md` explica el porqué de cada decisión (D1–D12).
- **Nunca commitear datos** (`data/` está en .gitignore). El repo es público y los datos SIMMA
  son propiedad intelectual del SGC (pie de página de sus reportes).
- Pruebas: `python -m pytest tests/ -q`. Pipeline sintético: ver "Cómo correr" en CONTRATOS_v2.md.
- Scripts de ingesta se corren como módulo desde la raíz: `python -m ingesta.<script>`.

## Estado de los datos SIMMA-Santander (2 oct 2026)

| Fuente | Registros | Fecha | Coordenadas | Archivo |
|---|---|---|---|---|
| Catálogo (portal, PDF) | 3.919 (1.676 desl., 1.160 flujos, 1.019 caídas, 63 rept., 1 volc.); 3.552 desde 1981 | sí | **no en PDF**; sí en tabla del portal (DMS) | PDF pendiente de subir al entorno |
| Inventario (portal, PDF) | 469 (245 deslizamientos) | sí | no en PDF | `data/crudo/inventario.parquet` |
| CSV datos.gov.co | 6.826, todo Colombia | **no** | sí, EPSG:3857 | — |

- **El CSV de datos.gov.co ES el Inventario SIMMA con coordenadas** (verificado: el punto de
  Carrizal/Cortaderas del portal coincide a 0.13 m con `Inentario_Movimiento=34923`).
  `Inentario_Movimiento` parece ser el ID del inventario. Útil para susceptibilidad.
- El problema de coordenadas real es el **Catálogo**: solo el portal las tiene.

## Decisiones tomadas
- **S1 cerrado (decisión de Esteban, 2 oct):** hay >600 casos con fecha diaria solo en Santander;
  no se amplía a otros departamentos.
  - Matiz verificado: el formato dd/mm/aaaa del PDF **no prueba** precisión diaria. En el
    Inventario hay 79 registros en día 1 vs ~15 esperados (01/07/2013 en 4 municipios distantes).
    Regla en ingesta: día 1 → `mes`, 01/01 → `anio`. Inventario queda: 390 dia / 67 mes / 12 anio.
    **Falta correr la misma auditoría sobre el Catálogo**; con 3.919 registros S1 sigue cerrado
    aunque ~15% resulte mensual.
  - Deslizamientos del Inventario con fecha diaria desde 1981: 189, concentrados en 2010 (31) y
    2013 (58) — posibles campañas de campo, no tormentas. Las fechas del Inventario podrían ser
    fecha de visita; el Catálogo es la fuente principal para el case-crossover.
- Ubicación de eventos, en orden de preferencia (`eventos_crudos.ubicacion_metodo`):
  1. `portal_dms`: lat/lon del portal SIMMA (script `ingesta/extraer_portal_simma.py`, corre local).
  2. `centroide_vereda`: respaldo SOLO si falla el portal (mapa de veredas DANE). Marca ubicación aproximada.
  Eventos con centroide de vereda no entran al case-crossover si la vereda es más grande que ~1 km² de
  imprecisión efectiva (regla D7: precisión espacial ≤ 1 km).

## Pendientes
- [Esteban] Subir el PDF del Catálogo al entorno → `python -m ingesta.parsear_simma_pdf --fuente simma_catalogo ...`
  y auditar día 1.
- [Esteban] Correr `python -m ingesta.extraer_portal_simma extraer --pestana catalogo` en local y
  devolver `data/crudo/portal_catalogo.csv` + la carpeta `portal_red_catalogo/`.
- [Claude] Si `portal_red_catalogo/` trae JSON con registros, reemplazar el scraping por cliente de API.
- [Claude] Ingestar el CSV nacional (3857 → WGS84) como capa de presencias para susceptibilidad.

## Entorno
- La sesión en la nube no tiene salida a `*.sgc.gov.co` ni `datos.gov.co` (proxy 403).
- Owner de GitHub: `Esteban-Codificador` (no `esteban21avila`).
