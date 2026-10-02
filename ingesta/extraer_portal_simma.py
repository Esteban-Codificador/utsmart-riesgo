"""Extrae la tabla de RESULTADOS del portal SIMMA (simma.sgc.gov.co), página por página.

Correr LOCALMENTE (el entorno de nube no tiene salida a sgc.gov.co). Semi-automático a
propósito: tú aplicas el filtro en el navegador (robusto a cambios de la UI) y el script
pagina, lee y guarda. Además graba toda respuesta JSON de la red: si el portal carga la
tabla desde una API, con esos archivos se escribe un cliente directo, mucho más rápido.

Instalación:
  pip install playwright pandas pyarrow && python -m playwright install chromium

Uso:
  python -m ingesta.extraer_portal_simma extraer --pestana catalogo
      1. Se abre Chromium en el portal.
      2. Tú: Departamento = SANTANDER, pestaña Catálogo, activas TODAS las columnas
         (incluidas Latitud, Longitud y el ID/código si existe), dejas la tabla en la página 1.
      3. Enter en la terminal. El script recorre todas las páginas.
      Salida: data/crudo/portal_catalogo.csv (checkpoint por página; si se corta, vuelve a
      correrlo con --reanudar y continúa desde la última página guardada).

  python -m ingesta.extraer_portal_simma convertir --pestana catalogo
      CSV crudo -> data/crudo/catalogo_portal.parquet con contrato eventos_crudos
      (lat/lon decimales desde grados-minutos-segundos, validados contra la caja de Santander).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

URL = "https://simma.sgc.gov.co/"
DIR = Path("data/crudo")

# ---------------------------------------------------------------- grados-minutos-segundos
_DMS = re.compile(
    r"""^\s*(?P<signo>[-−])?\s*(?P<g>\d+(?:[.,]\d+)?)\s*[°º˚]?\s*
        (?:(?P<m>\d+(?:[.,]\d+)?)\s*['′’]\s*)?
        (?:(?P<s>\d+(?:[.,]\d+)?)\s*(?:["″”]|'')?\s*)?
        (?P<hemi>[NSEWOnsewo])?\s*$""", re.X)


def dms_a_decimal(texto: str | float | None, es_longitud: bool) -> float | None:
    """'6° 38' 55.35"' -> 6.648708 ; '-72° 41' 7.81"' -> -72.685503 ; '72°41'7.81" W' -> -72.68.

    Acepta coma decimal, símbolos tipográficos, minutos/segundos opcionales y hemisferio.
    En Colombia toda longitud es oeste: si llega positiva se niega (el portal a veces omite el signo).
    """
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return None
    t = str(texto).strip().replace(" ", " ")
    if not t:
        return None
    m = _DMS.match(t)
    if not m:
        raise ValueError(f"coordenada no reconocida: {texto!r}")
    num = lambda k: float(m[k].replace(",", ".")) if m[k] else 0.0
    if m["m"] is None and m["s"] is None and "°" not in t:
        valor = num("g")                       # ya venía en decimal
    else:
        if num("m") >= 60 or num("s") >= 60:
            raise ValueError(f"minutos/segundos >= 60: {texto!r}")
        valor = num("g") + num("m") / 60 + num("s") / 3600
    negativo = bool(m["signo"]) or (m["hemi"] or "").upper() in {"S", "W", "O"}
    valor = -valor if negativo else valor
    if es_longitud and valor > 0:
        valor = -valor
    return round(valor, 7)


# ---------------------------------------------------------------- extracción
JS_LEER_TABLA = r"""
() => {
  // la tabla de resultados es la visible cuyo encabezado contiene "Latitud" o "Fecha"
  const tablas = [...document.querySelectorAll('table')].filter(t => t.offsetParent !== null);
  const txt = el => (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
  let mejor = null, puntaje = -1;
  for (const t of tablas) {
    const h = txt(t.querySelector('thead') || t.rows[0] || t);
    const p = (/Latitud/i.test(h) ? 2 : 0) + (/Fecha/i.test(h) ? 1 : 0) + Math.min(t.rows.length, 50) / 100;
    if (p > puntaje) { puntaje = p; mejor = t; }
  }
  if (!mejor) return null;
  const filas = [...mejor.rows].map(r => [...r.cells].map(c => txt(c)));
  let enc = mejor.tHead ? [...mejor.tHead.rows].pop() : null;
  let encabezados = enc ? [...enc.cells].map(c => txt(c)) : filas.shift();
  if (mejor.tHead) filas.splice(0, mejor.tHead.rows.length);
  // tablas con encabezado fijo en un <table> aparte (scroll horizontal): buscar encabezado gemelo
  if (encabezados.every(e => !e)) {
    const otra = tablas.find(t => t !== mejor && t.rows.length <= 2 && /Latitud|Fecha/i.test(txt(t)));
    if (otra) encabezados = [...otra.rows[otra.rows.length - 1].cells].map(c => txt(c));
  }
  return {encabezados, filas: filas.filter(f => f.some(x => x))};
}
"""

JS_SIGUIENTE = r"""
() => {
  // botón "›" (siguiente) de la paginación; null si no existe o está deshabilitado
  // a/button primero: el <li> que envuelve al enlace tiene el mismo texto pero no dispara el clic
  const sel = q => [...document.querySelectorAll(q)]
    .filter(e => e.offsetParent !== null && ['›', '>', 'Siguiente', 'Next'].includes((e.innerText || '').trim()));
  const cands = [...sel('a, button'), ...sel('li, span')];
  for (const e of cands) {
    const li = e.closest('li');
    const off = e.disabled || /disabled/.test(e.className) || (li && /disabled/.test(li.className))
                || e.getAttribute('aria-disabled') === 'true';
    if (!off) { e.click(); return true; }
  }
  return false;
}
"""

JS_PAGINA_ACTUAL = r"""
() => {
  const a = [...document.querySelectorAll('.active, [aria-current="page"], .selected, .current')]
    .map(e => (e.innerText || '').trim()).find(t => /^\d+$/.test(t));
  return a ? parseInt(a) : null;
}
"""


def _firma(datos) -> str:
    return hashlib.md5(json.dumps(datos["filas"]).encode()).hexdigest()


def extraer(pestana: str, reanudar: bool, pausa: float, espera_max: float = 20.0, url: str = URL,
            headless: bool = False) -> None:
    from playwright.sync_api import sync_playwright

    DIR.mkdir(parents=True, exist_ok=True)
    salida = DIR / f"portal_{pestana}.csv"
    red = DIR / f"portal_red_{pestana}"
    red.mkdir(exist_ok=True)
    previas = pd.read_csv(salida, dtype=str) if (reanudar and salida.exists()) else None
    ultima = int(previas["_pagina"].astype(int).max()) if previas is not None else 0

    with sync_playwright() as p:
        # SIMMA_CHROMIUM: ruta a un Chrome/Chromium propio si no quieres `playwright install`
        nav = p.chromium.launch(headless=headless,
                                executable_path=os.environ.get("SIMMA_CHROMIUM") or None)
        ctx = nav.new_context(viewport={"width": 1600, "height": 1000}, locale="es-CO")
        pag = ctx.new_page()

        n_json = 0

        def grabar(resp):
            nonlocal n_json
            try:
                if "json" in (resp.headers.get("content-type") or ""):
                    n_json += 1
                    nombre = f"{n_json:05d}_" + re.sub(r"[^\w]+", "_", resp.url.split("?")[0])[-80:] + ".json"
                    (red / nombre).write_text(json.dumps({"url": resp.url, "body": resp.json()},
                                                         ensure_ascii=False)[:5_000_000])
            except Exception:
                pass

        pag.on("response", grabar)
        pag.goto(url, wait_until="domcontentloaded", timeout=120_000)
        print(f"\n>>> En el navegador: filtro Departamento = SANTANDER, pestaña '{pestana}',"
              f" activa TODAS las columnas y deja la tabla en la página 1.")
        if ultima:
            print(f">>> Reanudando: ya hay {ultima} páginas guardadas; el script avanzará hasta la {ultima + 1}.")
        input(">>> Enter cuando la tabla esté lista... ")

        filas_tot = [] if previas is None else previas.to_dict("records")
        pagina, firma_previa, encabezados = 1, None, None
        while True:
            datos = pag.evaluate(JS_LEER_TABLA)
            if not datos or not datos["filas"]:
                print(f"!! página {pagina}: no encontré la tabla. Revisa el navegador.")
                if input("   Enter para reintentar, 'q' para terminar: ").strip().lower() == "q":
                    break
                continue
            encabezados = encabezados or datos["encabezados"]
            if pagina == 1 and not any(re.search("latitud", e, re.I) for e in encabezados):
                print("   columnas detectadas:", encabezados)
                input("!! No veo columna Latitud: activa todas las columnas y presiona Enter.")
                encabezados = None
                continue
            if pagina == 1:
                print("   columnas detectadas:", encabezados)
            firma_previa = _firma(datos)
            if pagina > ultima:
                for f in datos["filas"]:
                    r = dict(zip(encabezados_unicos(encabezados), f))
                    r["_pagina"] = pagina
                    filas_tot.append(r)
                pd.DataFrame(filas_tot).to_csv(salida, index=False)
            actual = pag.evaluate(JS_PAGINA_ACTUAL)
            print(f"   página {pagina} (portal dice {actual}): {len(datos['filas'])} filas, acumulado {len(filas_tot)}")
            if not pag.evaluate(JS_SIGUIENTE):
                print("   no hay botón 'siguiente' habilitado: fin.")
                break
            # esperar a que la tabla cambie de verdad (la carga puede ser asíncrona)
            limite = time.time() + espera_max
            while time.time() < limite:
                time.sleep(0.3)
                nuevo = pag.evaluate(JS_LEER_TABLA)
                if nuevo and nuevo["filas"] and _firma(nuevo) != firma_previa:
                    break
            else:
                print(f"   la tabla no cambió en {espera_max:.0f}s tras 'siguiente': fin en página {pagina}.")
                break
            pagina += 1
            time.sleep(pausa)   # no martillar el servidor del SGC
        nav.close()
    print(f"\nListo: {len(filas_tot)} filas -> {salida}; {n_json} respuestas JSON en {red}/")
    print("Si hay JSON con los registros en esa carpeta, mándamelos: se reemplaza el scraping por la API.")


def encabezados_unicos(enc: list[str]) -> list[str]:
    """El portal repite títulos ('Tipo movimiento' x2): sufijo _2, _3."""
    vistos: dict[str, int] = {}
    out = []
    for e in enc:
        e = e or "col"
        vistos[e] = vistos.get(e, 0) + 1
        out.append(e if vistos[e] == 1 else f"{e}_{vistos[e]}")
    return out


# ---------------------------------------------------------------- conversión a contrato
def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _col(df: pd.DataFrame, *patrones: str, requerida: bool = True) -> str | None:
    for p in patrones:
        for c in df.columns:
            if re.fullmatch(p, _norm(c)):
                return c
    if requerida:
        sys.exit(f"No encuentro columna {patrones} entre: {list(df.columns)}")
    return None


def convertir(pestana: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import contratos as C
    from ingesta.parsear_simma_pdf import TIPO_CANONICO, sin_tildes

    crudo = pd.read_csv(DIR / f"portal_{pestana}.csv", dtype=str)
    c_lat = _col(crudo, r"latitud.*")
    c_lon = _col(crudo, r"longitud.*")
    c_fecha = _col(crudo, r"fecha(_del)?_evento", r"fecha.*")
    c_mpio = _col(crudo, r"municipio")
    c_vda = _col(crudo, r"vereda", requerida=False)
    c_tipo = _col(crudo, r"tipo(_de)?_movimiento", r"tipo")
    c_sub = _col(crudo, r"subtipo.*", requerida=False)
    c_id = _col(crudo, r"id", r"codigo", r"id_.*", r"no", requerida=False)
    c_dep = _col(crudo, r"departamento", requerida=False)

    errores = []

    def conv(v, lon):
        try:
            return dms_a_decimal(v, lon)
        except ValueError as e:
            errores.append(str(e))
            return None

    df = pd.DataFrame({
        "fuente": f"simma_{pestana}",
        "simma_id": crudo[c_id] if c_id else None,
        "tipo_movimiento_original": crudo[c_tipo],
        "subtipo": crudo[c_sub] if c_sub else None,
        "fecha_evento_txt": crudo[c_fecha],
        "departamento": crudo[c_dep] if c_dep else "SANTANDER",
        "municipio": crudo[c_mpio].str.upper(),
        "vereda": crudo[c_vda].str.upper() if c_vda else None,
        "lat": [conv(v, False) for v in crudo[c_lat]],
        "lon": [conv(v, True) for v in crudo[c_lon]],
    })
    for c in ["tipo_secundario", "subtipo_secundario", "material", "material_secundario"]:
        df[c] = None
    df["fecha_evento"] = pd.to_datetime(df["fecha_evento_txt"], dayfirst=True, errors="coerce").dt.date
    df["precision_fecha"] = "dia"     # misma regla que el parser de PDF
    df.loc[df["fecha_evento_txt"].str.match(r"0?1/"), "precision_fecha"] = "mes"
    df.loc[df["fecha_evento_txt"].str.match(r"0?1/0?1/"), "precision_fecha"] = "anio"
    df.loc[df["fecha_evento"].isna(), "precision_fecha"] = "desconocida"
    df["tipo_movimiento"] = df["tipo_movimiento_original"].fillna("").map(
        lambda s: TIPO_CANONICO.get(sin_tildes(s).lower().strip(), "otro"))

    fuera = df["lat"].notna() & ~(df["lat"].between(5.6, 8.2) & df["lon"].between(-74.6, -72.4))
    if fuera.any():
        print(f"!! {int(fuera.sum())} coordenadas fuera de Santander -> se anulan (ver portal_{pestana}_fuera.csv)")
        crudo[fuera.values].to_csv(DIR / f"portal_{pestana}_fuera.csv", index=False)
        df.loc[fuera, ["lat", "lon"]] = None
    ok = df["lat"].notna() & df["lon"].notna()
    df["ubicacion_metodo"] = ok.map({True: "portal_dms", False: None})
    df["precision_espacial_m"] = None
    df["pagina_pdf"] = None

    clave = pd.Series(["|".join("" if pd.isna(v) else str(v) for v in fila)
                       for fila in df.itertuples(index=False)], index=df.index)
    ordinal = clave.groupby(clave).cumcount().astype(str)
    df["evento_id"] = [f"{pestana[:3].upper()}-" + hashlib.sha1((k + "#" + o).encode()).hexdigest()[:12]
                       for k, o in zip(clave, ordinal)]
    df["fecha_evento"] = df["fecha_evento"].where(df["fecha_evento"].notna(), None)
    salida = DIR / f"{pestana}_portal.parquet"
    C.escribir("eventos_crudos", df[C.EVENTOS_CRUDOS.names], salida)
    print(f"{len(df)} registros, {int(ok.sum())} con coordenadas, {len(errores)} coordenadas ilegibles -> {salida}")
    for e in errores[:10]:
        print("   ", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["extraer", "convertir"])
    ap.add_argument("--pestana", default="catalogo", choices=["catalogo", "inventario"])
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--pausa", type=float, default=1.5, help="segundos entre páginas")
    a = ap.parse_args()
    if a.accion == "extraer":
        extraer(a.pestana, a.reanudar, a.pausa)
    else:
        convertir(a.pestana)


if __name__ == "__main__":
    main()
