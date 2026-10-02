"""Parsea los PDF de reporte SIMMA (Catálogo / Inventario) a Parquet `eventos_crudos`.

Los PDF del portal no traen coordenadas: lat/lon quedan nulos y se llenan después con
`extraer_portal_simma.py` (DMS del portal) o, como respaldo, con centroide de vereda.

Uso (desde la raíz del repo):
  python -m ingesta.parsear_simma_pdf --fuente simma_inventario --salida data/crudo/inventario.parquet RUTA.pdf [...]
  python -m ingesta.parsear_simma_pdf --fuente simma_catalogo   --salida data/crudo/catalogo.parquet   RUTA.pdf [...]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pdfplumber

import contratos as C

# 87 municipios de Santander: vocabulario para reconstruir nombres partidos por el PDF.
MUNICIPIOS_SANTANDER = """AGUADA ALBANIA ARATOCA BARBOSA BARICHARA BARRANCABERMEJA BETULIA BOLÍVAR CABRERA
CALIFORNIA CAPITANEJO CARCASÍ CEPITÁ CERRITO CHARALÁ CHARTA CHIMA CHIPATÁ CIMITARRA CONCEPCIÓN
CONFINES CONTRATACIÓN COROMORO CURITÍ CARMEN CHUCURÍ GUACAMAYO PEÑÓN PLAYÓN ENCINO ENCISO FLORIÁN
FLORIDABLANCA GALÁN GÁMBITA GIRÓN GUACA GUADALUPE GUAPOTÁ GUAVATÁ GÜEPSA HATO JESÚS MARÍA JORDÁN
BELLEZA LANDÁZURI PAZ LEBRIJA SANTOS MACARAVITA MÁLAGA MATANZA MOGOTES MOLAGAVITA OCAMONTE OIBA
ONZAGA PALMAR PALMAS SOCORRO PÁRAMO PIEDECUESTA PINCHOTE PUENTE NACIONAL PUERTO PARRA WILCHES
RIONEGRO SABANA TORRES ANDRÉS BENITO GIL JOAQUÍN JOSÉ MIRANDA MIGUEL VICENTE SANTA BÁRBARA HELENA
OPÓN SIMACOTA SUAITA SUCRE SURATÁ TONA VALLE VÉLEZ VETAS VILLANUEVA ZAPATOCA BUCARAMANGA SANTANDER""".split()
SEMILLA = """deslizamiento deslizamientos traslacional rotacional cuña planar flujo flujos detritos lodo
tierra tierras roca rocas caida caída caidas reptación reptacion suelos volcamiento propagación propagacion
lateral lenta rápida avalancha movimiento material de del la el los las en y o""".split()
STOP = {"DE", "LA", "EL", "LAS", "LOS", "DEL", "Y", "O", "EN", "SAN"}

COLUMNAS = ["tipo_movimiento_original", "fecha_evento_txt", "departamento", "municipio", "vereda",
            "tipo_secundario", "subtipo", "subtipo_secundario", "material", "material_secundario"]

TIPO_CANONICO = {"deslizamiento": "deslizamiento", "caida": "caida", "flujo": "flujo",
                 "reptacion": "reptacion"}  # volcamiento, propagación lateral, etc. -> "otro"


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


FINALES_VALIDOS = set("AEIOULNRSDZYJX")


def termina_imposible(palabra: str) -> bool:
    """True si la palabra no puede terminar ahí en español (consonante rara o grupo consonántico)."""
    if len(palabra) < 3 or not palabra.isalpha():
        return False
    a, b = palabra[-2], palabra[-1]
    if b not in FINALES_VALIDOS:
        return True
    if b in "AEIOU":
        return False
    return a not in "AEIOU" and b != "S" and a not in "LNR"   # "GR", "ND"... al final


def unir_lineas(celda: str | None, vocab: set[str], ancho: int) -> str | None:
    """Reconstruye una celda partida en líneas: decide si cada salto es dentro o entre palabras."""
    if celda is None:
        return None
    celda = celda.replace("#", "Ñ")   # el PDF del portal codifica Ñ como '#'
    lineas = [l.strip() for l in celda.split("\n") if l.strip()]
    if not lineas:
        return None
    out = lineas[0]
    for prev, sig in zip(lineas, lineas[1:]):
        x = out.split(" ")[-1]
        y = sig.split(" ")[0]
        ux, uy, uxy = (sin_tildes(t).upper() for t in (x, y, x + y))
        if uxy in vocab:
            sep = ""
        elif ux in vocab and uy in vocab:
            sep = " "
        elif len(y) <= 2 and uy not in STOP:
            sep = ""
        elif termina_imposible(ux):
            sep = ""   # "PANTANOGR": ninguna palabra en español termina así
        elif " " not in prev and len(prev) >= ancho - 1:
            sep = ""   # una sola palabra que llenó el ancho de la columna: se cortó adentro
        else:
            sep = " "
        out = out + sep + sig
    return out


def leer_tablas(rutas: list[Path]) -> list[tuple[int, list[str | None]]]:
    filas = []
    for ruta in rutas:
        with pdfplumber.open(ruta) as pdf:
            for n, pag in enumerate(pdf.pages, 1):
                for tabla in pag.extract_tables():
                    for fila in tabla:
                        if fila and fila[1] and re.match(r"\d{2}/\d{2}/\d{4}", fila[1].strip()):
                            filas.append((n, fila))
    return filas


def construir_vocab(filas) -> tuple[set[str], list[int]]:
    vocab = {sin_tildes(w).upper() for w in MUNICIPIOS_SANTANDER + SEMILLA}
    anchos = [0] * len(COLUMNAS)
    for _, fila in filas:
        for j, celda in enumerate(fila[: len(COLUMNAS)]):
            if not celda:
                continue
            lineas = celda.split("\n")
            anchos[j] = max(anchos[j], *(len(l) for l in lineas))
            for i, l in enumerate(lineas):
                toks = l.split()
                # un token pudo ser cortado solo si es el último antes de un salto o el primero
                # después de uno; los demás son palabras completas
                ini = 1 if i > 0 else 0
                fin = len(toks) - 1 if i < len(lineas) - 1 else len(toks)
                vocab |= {sin_tildes(t).upper() for t in toks[ini:fin] if len(t) > 2}
    return vocab, anchos


def parsear(rutas: list[Path], fuente: str) -> pd.DataFrame:
    filas = leer_tablas(rutas)
    vocab, anchos = construir_vocab(filas)
    registros = []
    for pagina, fila in filas:
        valores = [unir_lineas(c, vocab, anchos[j]) if c else None for j, c in enumerate(fila[: len(COLUMNAS)])]
        r = dict(zip(COLUMNAS, valores))
        r["pagina_pdf"] = pagina
        registros.append(r)
    df = pd.DataFrame(registros)
    for c in COLUMNAS:
        df[c] = df[c].replace("", None)

    df["fecha_evento"] = pd.to_datetime(df["fecha_evento_txt"], format="%d/%m/%Y").dt.date
    # El PDF siempre muestra dd/mm/aaaa: el formato NO prueba precisión diaria. SIMMA guarda
    # "solo año" como 01/01 y "solo mes" como 01/mm. Inventario Santander: 79 registros en día 1
    # vs ~15 esperados, y el 01/07/2013 aparece en 4 municipios distantes. Regla conservadora:
    # día 1 -> "mes", 01/01 -> "anio". Se pierden ~3% de eventos reales de día 1; se evita
    # contaminar el case-crossover con ~14% de fechas falsas.
    df["precision_fecha"] = "dia"
    df.loc[df["fecha_evento_txt"].str.startswith("01/"), "precision_fecha"] = "mes"
    df.loc[df["fecha_evento_txt"].str.startswith("01/01/"), "precision_fecha"] = "anio"

    tipo = df["tipo_movimiento_original"].map(lambda s: sin_tildes(s).lower() if s else "")
    df["tipo_movimiento"] = tipo.map(TIPO_CANONICO).fillna("otro")

    # id estable: hash del contenido + ordinal para duplicados exactos (que se conservan: ver D8)
    clave = df[COLUMNAS].fillna("").astype(str).agg("|".join, axis=1)
    ordinal = clave.groupby(clave).cumcount().astype(str)
    df["evento_id"] = [f"{fuente[6:9].upper()}-" + hashlib.sha1((k + "#" + o).encode()).hexdigest()[:12]
                       for k, o in zip(clave, ordinal)]
    df["fuente"] = fuente
    df["simma_id"] = None
    df["lat"] = None
    df["lon"] = None
    df["ubicacion_metodo"] = None
    df["precision_espacial_m"] = None
    return df[C.EVENTOS_CRUDOS.names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--fuente", required=True, choices=["simma_catalogo", "simma_inventario"])
    ap.add_argument("--salida", required=True, type=Path)
    a = ap.parse_args()
    df = parsear(a.pdfs, a.fuente)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    C.escribir("eventos_crudos", df, a.salida)
    print(f"{len(df)} registros -> {a.salida}")


if __name__ == "__main__":
    main()
