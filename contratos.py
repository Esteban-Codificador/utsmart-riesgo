"""Contratos de datos v2 — UTSmart 2027, riesgo de movimientos en masa (Santander).

Fuente única de verdad para esquemas. Todo productor y consumidor importa de aquí.
Formato de intercambio: Parquet. CRS proyectado: EPSG:9377 (MAGNA-SIRGAS Origen Nacional).
"""
from __future__ import annotations

import pyarrow as pa
import pandas as pd

VERSION_CONTRATO = "2.1.0"   # 2.1: + eventos_crudos (staging de ingesta SIMMA)
CRS_PROYECTADO = "EPSG:9377"
TAMANO_CELDA_M = 1000          # celda regular 1 km
TAMANO_PIXEL_LLUVIA_M = 5000   # aprox. CHIRPS 0.05° (~5.5 km); 5x5 celdas por píxel

PRECISION_FECHA = ["dia", "mes", "anio", "desconocida"]
CATEGORIA_RIESGO = ["bajo", "medio", "alto", "muy_alto"]
USO_VARIABLE = {"modelo", "sesgo", "descriptiva"}

# ---------------------------------------------------------------- 1. celdas
# Una fila por celda de 1 km. Estática. ~31k filas para Santander.
CELDAS = pa.schema([
    ("cell_id", pa.string()),            # "C{col:04d}_{row:04d}" sobre malla EPSG:9377
    ("pixel_lluvia_id", pa.string()),    # FK -> lluvia_diaria.pixel_lluvia_id
    ("x_m", pa.float64()), ("y_m", pa.float64()),        # centroide EPSG:9377
    ("lon", pa.float64()), ("lat", pa.float64()),        # centroide WGS84 (solo visualización)
    ("cod_dane_mpio", pa.string()),      # 5 dígitos, municipio mayoritario
    ("area_km2", pa.float32()),          # < 1 en bordes del departamento
    # --- variables del modelo de susceptibilidad (uso = "modelo")
    ("elev_media_m", pa.float32()),
    ("pendiente_media_deg", pa.float32()),
    ("pendiente_p90_deg", pa.float32()),
    ("twi_medio", pa.float32()),
    ("curvatura_plan_media", pa.float32()),
    ("litologia_cod", pa.string()),      # categórica, unidad mayoritaria SGC 1:100k
    ("cobertura_clc_cod", pa.string()),  # categórica, Corine Land Cover nivel 2
    ("dist_drenaje_m", pa.float32()),
    ("precip_media_anual_mm", pa.float32()),
    # --- variables de sesgo de reporte (uso = "sesgo"): NO entran al modelo
    ("dist_via_m", pa.float32()),
    ("poblacion", pa.float32()),
])

USO_COLUMNAS_CELDAS = {
    **{c: "modelo" for c in [
        "elev_media_m", "pendiente_media_deg", "pendiente_p90_deg", "twi_medio",
        "curvatura_plan_media", "litologia_cod", "cobertura_clc_cod",
        "dist_drenaje_m", "precip_media_anual_mm"]},
    "dist_via_m": "sesgo",
    "poblacion": "sesgo",
}

# ---------------------------------------------------------------- 2. eventos
# Una fila por registro SIMMA dentro de Santander. Crudo + normalizado.
EVENTOS = pa.schema([
    ("evento_id", pa.string()),          # id interno estable
    ("simma_id", pa.string()),           # id original SIMMA (nullable)
    ("tipo_movimiento", pa.string()),    # deslizamiento | caida | flujo | reptacion | otro
    ("fecha_evento", pa.date32()),       # nullable si precision_fecha = desconocida
    ("precision_fecha", pa.string()),    # enum PRECISION_FECHA
    ("fecha_reporte", pa.date32()),      # nullable
    ("lon", pa.float64()), ("lat", pa.float64()),
    ("precision_espacial_m", pa.float32()),  # nullable; radio de incertidumbre
    ("cell_id", pa.string()),            # FK -> celdas
    ("usable_temporal", pa.bool_()),     # precision_fecha == "dia" y fecha en rango de lluvia
])

# ---------------------------------------------------------------- 2b. eventos_crudos (staging)
# Salida de la ingesta SIMMA antes de geolocalizar. lat/lon nulos hasta que se resuelva la
# ubicación; de aquí se deriva `eventos` (que exige cell_id) con las filas ubicadas.
UBICACION_METODO = ["portal_dms", "centroide_vereda", "centroide_municipio"]
EVENTOS_CRUDOS = pa.schema([
    ("evento_id", pa.string()),
    ("fuente", pa.string()),             # simma_catalogo | simma_inventario
    ("simma_id", pa.string()),           # nullable: los PDF no lo traen
    ("tipo_movimiento", pa.string()),    # canónico: deslizamiento | caida | flujo | reptacion | otro
    ("tipo_movimiento_original", pa.string()),
    ("tipo_secundario", pa.string()),
    ("subtipo", pa.string()),
    ("subtipo_secundario", pa.string()),
    ("material", pa.string()),
    ("material_secundario", pa.string()),
    ("fecha_evento_txt", pa.string()),   # tal como viene, para auditoría
    ("fecha_evento", pa.date32()),
    ("precision_fecha", pa.string()),    # enum PRECISION_FECHA
    ("departamento", pa.string()),
    ("municipio", pa.string()),
    ("vereda", pa.string()),
    ("lat", pa.float64()), ("lon", pa.float64()),   # WGS84, nullable
    ("ubicacion_metodo", pa.string()),   # enum UBICACION_METODO, nulo si lat/lon nulos
    ("precision_espacial_m", pa.float32()),
    ("pagina_pdf", pa.int32()),          # nullable si no viene de PDF
])

# ---------------------------------------------------------------- 3. lluvia
# Nivel píxel CHIRPS, NO nivel celda: la celda no agrega resolución de lluvia.
# ~1.2k píxeles x ~9k días = ~11M filas. Variables antecedentes se derivan, no se guardan.
LLUVIA_DIARIA = pa.schema([
    ("pixel_lluvia_id", pa.string()),
    ("fecha", pa.date32()),
    ("precip_mm", pa.float32()),
    ("fuente", pa.string()),             # "CHIRPS-2.0" | "CHIRPS-prelim" | "IDEAM" | "sintetico"
])

# ---------------------------------------------------------------- 4. muestras susceptibilidad
# Modelo A (estático): presencia vs pseudo-ausencia a nivel celda.
MUESTRAS_SUSC = pa.schema([
    ("cell_id", pa.string()),
    ("etiqueta", pa.int8()),             # 1 = celda con >=1 evento; 0 = pseudo-ausencia
    ("estrategia", pa.string()),         # "presencia" | "pseudoausencia_buffer_estratificada"
    ("bloque_espacial", pa.int16()),     # fold para CV espacial (bloques 20 km)
    ("peso", pa.float32()),
])

# ---------------------------------------------------------------- 5. muestras case-crossover
# Modelo B (detonante): por evento con fecha diaria, 1 día caso + controles del mismo estrato.
MUESTRAS_CC = pa.schema([
    ("estrato_id", pa.string()),         # = evento_id
    ("cell_id", pa.string()),
    ("pixel_lluvia_id", pa.string()),
    ("fecha", pa.date32()),
    ("es_caso", pa.int8()),
    ("p1d_mm", pa.float32()), ("p3d_mm", pa.float32()), ("p7d_mm", pa.float32()),
    ("p15d_mm", pa.float32()), ("p30d_mm", pa.float32()),
    ("api_k090", pa.float32()),          # índice de precipitación antecedente, k = 0.90, 30 días
])

# ---------------------------------------------------------------- 6. predicciones (salida -> frontend)
PREDICCIONES = pa.schema([
    ("cell_id", pa.string()),
    ("fecha_objetivo", pa.date32()),     # día para el que se predice
    ("fecha_ultimo_dato_lluvia", pa.date32()),  # transparencia sobre el rezago de CHIRPS
    ("susceptibilidad", pa.float32()),   # P(celda con evento reportado), calibrada Venn-ABERS
    ("susceptibilidad_inf", pa.float32()),
    ("susceptibilidad_sup", pa.float32()),
    ("rr_lluvia", pa.float32()),         # riesgo relativo del detonante vs día típico (exp(b·x))
    ("rr_lluvia_inf", pa.float32()),
    ("rr_lluvia_sup", pa.float32()),
    ("tasa_diaria", pa.float32()),       # P(evento reportado en la celda ese día)
    ("tasa_diaria_inf", pa.float32()),
    ("tasa_diaria_sup", pa.float32()),
    ("categoria", pa.string()),          # enum CATEGORIA_RIESGO, por cuantiles fijos de tasa_diaria
    ("version_modelo", pa.string()),
])

ESQUEMAS = {
    "celdas": CELDAS,
    "eventos": EVENTOS,
    "eventos_crudos": EVENTOS_CRUDOS,
    "lluvia_diaria": LLUVIA_DIARIA,
    "muestras_susceptibilidad": MUESTRAS_SUSC,
    "muestras_case_crossover": MUESTRAS_CC,
    "predicciones": PREDICCIONES,
}

CLAVES = {
    "celdas": ["cell_id"],
    "eventos": ["evento_id"],
    "eventos_crudos": ["evento_id"],
    "lluvia_diaria": ["pixel_lluvia_id", "fecha"],
    "muestras_susceptibilidad": ["cell_id"],
    "muestras_case_crossover": ["estrato_id", "fecha"],
    "predicciones": ["cell_id", "fecha_objetivo"],
}

NO_NULOS = {
    "celdas": [f.name for f in CELDAS],
    "eventos": ["evento_id", "tipo_movimiento", "precision_fecha", "lon", "lat",
                "cell_id", "usable_temporal"],
    "eventos_crudos": ["evento_id", "fuente", "tipo_movimiento", "fecha_evento_txt",
                       "precision_fecha", "departamento", "municipio"],
    "lluvia_diaria": [f.name for f in LLUVIA_DIARIA],
    "muestras_susceptibilidad": [f.name for f in MUESTRAS_SUSC],
    "muestras_case_crossover": [f.name for f in MUESTRAS_CC],
    "predicciones": [f.name for f in PREDICCIONES],
}


def validar(nombre: str, df: pd.DataFrame, celdas: pd.DataFrame | None = None) -> list[str]:
    """Devuelve lista de violaciones del contrato (vacía = OK)."""
    errores: list[str] = []
    esquema = ESQUEMAS[nombre]
    faltan = set(esquema.names) - set(df.columns)
    sobran = set(df.columns) - set(esquema.names)
    if faltan:
        errores.append(f"{nombre}: faltan columnas {sorted(faltan)}")
    if sobran:
        errores.append(f"{nombre}: columnas no contratadas {sorted(sobran)}")
    if faltan:
        return errores
    try:
        pa.Table.from_pandas(df[esquema.names], schema=esquema, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
        errores.append(f"{nombre}: tipos inválidos: {e}")
    for c in NO_NULOS[nombre]:
        n = int(df[c].isna().sum())
        if n:
            errores.append(f"{nombre}.{c}: {n} nulos en columna no nula")
    dup = int(df.duplicated(CLAVES[nombre]).sum())
    if dup:
        errores.append(f"{nombre}: {dup} claves duplicadas {CLAVES[nombre]}")

    if nombre == "eventos":
        malos = ~df["precision_fecha"].isin(PRECISION_FECHA)
        if malos.any():
            errores.append(f"eventos.precision_fecha: {int(malos.sum())} valores fuera de enum")
        sin_fecha = df["fecha_evento"].isna() & (df["precision_fecha"] != "desconocida")
        if sin_fecha.any():
            errores.append(f"eventos: {int(sin_fecha.sum())} sin fecha pero con precisión declarada")
        incoh = df["usable_temporal"] & (df["precision_fecha"] != "dia")
        if incoh.any():
            errores.append(f"eventos: {int(incoh.sum())} usable_temporal sin precisión diaria")
    if nombre == "eventos_crudos":
        malos = ~df["precision_fecha"].isin(PRECISION_FECHA)
        if malos.any():
            errores.append(f"eventos_crudos.precision_fecha: {int(malos.sum())} valores fuera de enum")
        ubic = df["lat"].notna() & df["lon"].notna()
        if (ubic != df["ubicacion_metodo"].notna()).any():
            errores.append("eventos_crudos: ubicacion_metodo debe estar lleno si y solo si hay lat/lon")
        m = df["ubicacion_metodo"].notna() & ~df["ubicacion_metodo"].isin(UBICACION_METODO)
        if m.any():
            errores.append(f"eventos_crudos.ubicacion_metodo: {int(m.sum())} valores fuera de enum")
        fuera = ubic & ~(df["lat"].between(5.6, 8.2) & df["lon"].between(-74.6, -72.4))
        if fuera.any():
            errores.append(f"eventos_crudos: {int(fuera.sum())} coordenadas fuera de la caja de Santander")
    if nombre == "lluvia_diaria" and (df["precip_mm"] < 0).any():
        errores.append("lluvia_diaria.precip_mm: valores negativos")
    if nombre == "muestras_case_crossover":
        casos = df.groupby("estrato_id")["es_caso"].sum()
        if (casos != 1).any():
            errores.append(f"case-crossover: {int((casos != 1).sum())} estratos sin exactamente 1 caso")
    if nombre == "predicciones":
        for base in ["susceptibilidad", "rr_lluvia", "tasa_diaria"]:
            mal = (df[f"{base}_inf"] > df[base]) | (df[base] > df[f"{base}_sup"])
            if mal.any():
                errores.append(f"predicciones.{base}: {int(mal.sum())} intervalos que no contienen el punto")
        if (df["fecha_ultimo_dato_lluvia"] > df["fecha_objetivo"]).any():
            errores.append("predicciones: fecha_ultimo_dato_lluvia posterior a fecha_objetivo")
        if not df["categoria"].isin(CATEGORIA_RIESGO).all():
            errores.append("predicciones.categoria: valores fuera de enum")
    if celdas is not None and "cell_id" in df.columns and nombre != "celdas":
        huerf = ~df["cell_id"].isin(celdas["cell_id"])
        if huerf.any():
            errores.append(f"{nombre}: {int(huerf.sum())} cell_id sin celda (FK rota)")
    return errores


def escribir(nombre: str, df: pd.DataFrame, ruta: str) -> None:
    errores = validar(nombre, df)
    if errores:
        raise ValueError("\n".join(errores))
    tabla = pa.Table.from_pandas(df[ESQUEMAS[nombre].names], schema=ESQUEMAS[nombre],
                                 preserve_index=False)
    tabla = tabla.replace_schema_metadata({"contrato": nombre, "version": VERSION_CONTRATO})
    import pyarrow.parquet as pq
    pq.write_table(tabla, ruta, compression="zstd")
