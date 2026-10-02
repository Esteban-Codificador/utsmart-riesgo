"""Pruebas de la ingesta SIMMA. Correr desde la raíz: python -m pytest tests/ -q

La prueba del extractor usa un portal simulado local (paginación asíncrona, encabezados
repetidos, coordenadas DMS) porque el portal real no es accesible desde CI.
"""
from __future__ import annotations

import builtins
import os
from pathlib import Path

import pandas as pd
import pytest

from ingesta import extraer_portal_simma as E
from ingesta.parsear_simma_pdf import termina_imposible, unir_lineas


@pytest.mark.parametrize("txt,lon,esperado", [
    ("6° 38' 55.35\"", False, 6.6487083),
    ("-72° 41' 7.81\"", True, -72.6855028),
    ("72° 41' 7.81\" W", True, -72.6855028),
    ("72°41'7,81\"", True, -72.6855028),        # coma decimal, sin signo -> oeste
    ("6º 38’ 55.35”", False, 6.6487083),         # símbolos tipográficos
    ("7° 5'", False, 7.0833333),                 # sin segundos
    ("6.648708", False, 6.648708),               # ya decimal
])
def test_dms(txt, lon, esperado):
    assert E.dms_a_decimal(txt, lon) == pytest.approx(esperado, abs=1e-6)


@pytest.mark.parametrize("malo", ["6° 75' 1\"", "abc", "6° 38' 61\""])
def test_dms_invalido(malo):
    with pytest.raises(ValueError):
        E.dms_a_decimal(malo, False)


def test_dms_nulos():
    assert E.dms_a_decimal(None, False) is None and E.dms_a_decimal("  ", True) is None


@pytest.mark.parametrize("celda,esperado", [
    ("CORTADERA\nS", "CORTADERAS"),
    ("SAN VICENTE\nDE CHUCURÍ", "SAN VICENTE DE CHUCURÍ"),
    ("PANTANOGR\nANDE", "PANTANOGRANDE"),
    ("SANTA ROSA\nDEL CARARE", "SANTA ROSA DEL CARARE"),
    ("PE#A COLORADA", "PEÑA COLORADA"),
    ("Deslizamient\no\ntraslacional", "Deslizamiento traslacional"),
    ("TIERRA\nBLANCA", "TIERRA BLANCA"),
])
def test_unir_lineas(celda, esperado):
    vocab = {"DESLIZAMIENTO", "TRASLACIONAL", "VICENTE", "DE", "CHUCURI", "SANTA", "ROSA"}
    assert unir_lineas(celda, vocab, ancho=11) == esperado


def test_termina_imposible():
    assert termina_imposible("PANTANOGR") and termina_imposible("BARRANCAB")
    assert not any(termina_imposible(w) for w in ["ROSA", "SANTA", "TORRES", "ANDRES", "VICENTE"])


PORTAL = """<!doctype html><html><body>
<div id="res"><table><thead><tr>
<th>Tipo movimiento</th><th>Fecha evento</th><th>Municipio</th><th>Vereda</th>
<th>Latitud (°)</th><th>Longitud (°)</th><th>Tipo movimiento</th></tr></thead>
<tbody id="tb"></tbody></table>
<ul class="pagination"><li><a id="sig">›</a></li></ul></div>
<script>
const datos = [];
for (let i = 0; i < 23; i++) datos.push(["Deslizamiento", `${String(2 + i % 27).padStart(2,'0')}/10/2013`,
  "ENCISO", "VDA " + i, `6° ${10 + i}' 5.5"`, `-72° 41' ${i}.81"`, i % 2 ? "Flujo" : ""]);
datos.push(["Caida", "01/01/1990", "MÁLAGA", "X", "4° 0' 0\\"", "-72° 0' 0\\"", ""]);  // fuera de Santander
let p = 0; const N = 10;
function pintar() {
  document.getElementById('tb').innerHTML = datos.slice(p*N, p*N+N)
    .map(f => '<tr>' + f.map(c => `<td>${c}</td>`).join('') + '</tr>').join('');
  if ((p+1)*N >= datos.length) document.getElementById('sig').parentElement.className = 'disabled';
}
document.getElementById('sig').onclick = () => {
  if ((p+1)*N >= datos.length) return;
  setTimeout(() => { p++; pintar(); }, 800);   // carga asíncrona como un portal real
};
pintar();
</script></body></html>"""


@pytest.mark.skipif(not Path("/opt/pw-browsers").exists() and not os.environ.get("CORRER_PLAYWRIGHT"),
                    reason="sin Chromium local")
def test_extractor_portal_simulado(tmp_path, monkeypatch):
    html = tmp_path / "portal.html"
    html.write_text(PORTAL, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(E, "DIR", Path("data/crudo"))
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    if Path("/opt/pw-browsers/chromium").exists():
        monkeypatch.setenv("SIMMA_CHROMIUM", "/opt/pw-browsers/chromium")
    E.extraer("catalogo", reanudar=False, pausa=0, espera_max=5, url=html.as_uri(), headless=True)

    crudo = pd.read_csv("data/crudo/portal_catalogo.csv", dtype=str)
    assert len(crudo) == 24 and crudo["_pagina"].astype(int).max() == 3
    assert "Tipo movimiento_2" in crudo.columns          # encabezado repetido desambiguado

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    E.convertir("catalogo")
    df = pd.read_parquet("data/crudo/catalogo_portal.parquet")
    assert len(df) == 24
    assert df["lat"].notna().sum() == 23                 # la de MÁLAGA 4°N se anula por fuera de caja
    assert df.loc[df.vereda == "VDA 0", "lat"].iloc[0] == pytest.approx(6 + 10 / 60 + 5.5 / 3600)
    assert (df["lon"].dropna() < 0).all()
    assert set(df.loc[df.lat.isna(), "ubicacion_metodo"].fillna("nulo")) == {"nulo"}
    assert df.loc[df.fecha_evento_txt == "01/01/1990", "precision_fecha"].iloc[0] == "anio"
