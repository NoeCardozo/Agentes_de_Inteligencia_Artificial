"""
=============================================================================
APIs PÚBLICAS DE ARGENTINA
=============================================================================

Este módulo consulta datos oficiales y abiertos sobre ciudades argentinas.

APIs utilizadas (gratuitas, sin API key):
  1. Georef (datos.gob.ar) → Datos oficiales del Estado argentino:
     provincias, localidades, coordenadas GPS, departamentos.
  2. Wikipedia en español → Resumen turístico e histórico de cada ciudad.

Para la defensa:
  - Georef es la fuente OFICIAL de división político-territorial de Argentina.
  - Wikipedia complementa con contexto cultural/turístico en lenguaje natural.
  - No guardamos estos datos: se consultan en tiempo real cuando el usuario pregunta.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

# URL base de la API Georef del gobierno argentino (versión 2.0)
GEOREF_BASE = "https://apis.datos.gob.ar/georef/api"

# Wikipedia requiere identificar la aplicación (política de uso)
WIKI_HEADERS = {"User-Agent": "AgenteTurismoArgentina/1.0 (proyecto educativo)"}


def _http_get(url: str, headers: dict | None = None) -> dict:
    """
    Hace una petición GET y devuelve la respuesta como diccionario JSON.

    Si la API falla, lanza una excepción con mensaje claro.
    """
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Error HTTP {exc.code} al consultar: {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con la API: {exc.reason}") from exc


def listar_provincias() -> str:
    """
    Obtiene las 24 provincias argentinas desde Georef.

    Retorna texto formateado listo para mostrar al usuario.
    """
    url = f"{GEOREF_BASE}/provincias?max=30&orden=nombre&campos=id,nombre"
    data = _http_get(url)

    provincias = data.get("provincias", [])
    if not provincias:
        return "No se pudieron obtener las provincias de Argentina."

    lineas = ["**Provincias de Argentina** (fuente: API Georef - datos.gob.ar)", ""]
    for i, prov in enumerate(provincias, 1):
        lineas.append(f"{i}. {prov['nombre']}")

    lineas.append(f"\nTotal: {data.get('total', len(provincias))} provincias.")
    return "\n".join(lineas)


def buscar_localidad(nombre: str, provincia: str = "", max_resultados: int = 5) -> str:
    """
    Busca localidades/ciudades argentinas por nombre en Georef.

    Parámetros:
      nombre    → Nombre de la ciudad a buscar (ej: "Mendoza", "Bariloche")
      provincia → Opcional, filtra por provincia (ej: "Neuquén")
    """
    params = {
        "nombre": nombre,
        "max": max_resultados,
        "campos": "id,nombre,provincia.nombre,departamento.nombre,centroide.lat,centroide.lon",
    }
    if provincia.strip():
        params["provincia"] = provincia.strip()

    url = f"{GEOREF_BASE}/localidades?{urllib.parse.urlencode(params)}"
    data = _http_get(url)

    localidades = data.get("localidades", [])
    if not localidades:
        return f"No se encontraron localidades con el nombre '{nombre}' en Argentina."

    lineas = [
        f"**Localidades encontradas para '{nombre}'** (fuente: API Georef)",
        "",
    ]
    for i, loc in enumerate(localidades, 1):
        prov = loc.get("provincia", {}).get("nombre", "—")
        depto = loc.get("departamento", {}).get("nombre", "—")
        centroide = loc.get("centroide", {})
        lat = centroide.get("lat", "—")
        lon = centroide.get("lon", "—")
        lineas.append(f"{i}. **{loc['nombre']}**")
        lineas.append(f"   - Provincia: {prov}")
        lineas.append(f"   - Departamento: {depto}")
        lineas.append(f"   - Coordenadas: {lat}, {lon}")
        lineas.append("")

    return "\n".join(lineas)


def _resumen_wikipedia(ciudad: str, provincia: str = "") -> str:
    """
    Busca un resumen de la ciudad en Wikipedia en español.

    Prueba varias variantes del título (ej: "Mendoza, Argentina").
    """
    titulos_a_probar = [
        f"{ciudad}, {provincia}" if provincia else "",
        f"{ciudad}, Argentina",
        ciudad,
        f"{ciudad} (Argentina)",
    ]

    for titulo in titulos_a_probar:
        if not titulo.strip():
            continue
        titulo_url = urllib.parse.quote(titulo.replace(" ", "_"))
        url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{titulo_url}"
        try:
            data = _http_get(url, headers=WIKI_HEADERS)
            extracto = data.get("extract", "")
            if extracto:
                titulo_wiki = data.get("title", titulo)
                link = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                resultado = f"**{titulo_wiki}** (fuente: Wikipedia)\n\n{extracto}"
                if link:
                    resultado += f"\n\nMás info: {link}"
                return resultado
        except RuntimeError:
            continue

    return f"No se encontró artículo en Wikipedia para '{ciudad}'."


def consultar_ciudad(ciudad: str, provincia: str = "") -> str:
    """
    Consulta completa de una ciudad argentina: datos oficiales + Wikipedia.

    Combina Georef (ubicación, provincia, coordenadas) con Wikipedia (contexto turístico).
    Esta es la función principal que usa el agente.
    """
    partes = [f"# Información de {ciudad}, Argentina\n"]

    # 1) Datos oficiales de Georef
    try:
        partes.append(buscar_localidad(ciudad, provincia=provincia, max_resultados=3))
    except RuntimeError as exc:
        partes.append(f"(Georef no disponible: {exc})")

    partes.append("")

    # 2) Resumen turístico de Wikipedia
    try:
        partes.append(_resumen_wikipedia(ciudad, provincia))
    except RuntimeError as exc:
        partes.append(f"(Wikipedia no disponible: {exc})")

    return "\n".join(partes)
