"""
Documentos de ejemplo para poblar el vector store RAG.
En producción, estos contenidos vienen de PDFs, sitios oficiales y guías turísticas.
"""
from langchain_core.documents import Document

SAMPLE_DOCUMENTS = [
    Document(
        page_content="""
Bariloche (San Carlos de Bariloche) — Ficha turística
Ubicación: Provincia de Río Negro, Patagonia Argentina.
Mejor época: Invierno (junio-agosto) para ski en Cerro Catedral; verano (diciembre-febrero) para trekking y navegación.
Temperatura promedio: -2°C a 8°C en invierno; 10°C a 25°C en verano.
Atracciones principales: Cerro Catedral (centro de ski más grande de América del Sur), Circuito Chico, Isla Victoria y Bosque de Arrayanes, Centro Cívico, Lago Nahuel Huapi.
Gastronomía: Chocolate artesanal, cerveza artesanal, trucha y jabalí.
Transporte: Vuelo directo desde Buenos Aires (~2hs), bus desde Buenos Aires (~20hs).
Normativa de Parques Nacionales: Para ingresar al Parque Nacional Nahuel Huapi se requiere abonar el arancel de entrada. Algunos senderos requieren registro previo.
Alojamiento: Opciones desde hosteles económicos ($10.000-$15.000/noche) hasta hoteles 5 estrellas ($100.000+/noche). Las cabañas son populares para familias.
        """,
        metadata={"source": "ficha_bariloche", "region": "patagonia", "categoria": "destino"},
    ),
    Document(
        page_content="""
Mendoza — Ficha turística
Ubicación: Provincia de Mendoza, Cuyo, Argentina. Pie de la Cordillera de los Andes.
Mejor época: Marzo (vendimia) y octubre-noviembre para clima ideal. Verano es muy caluroso (35-40°C).
Atracciones: Ruta del Vino (más de 200 bodegas), Aconcagua (base accesible sin equipamiento), Parque General San Martín, Termas de Cacheuta, Cañón del Atuel (San Rafael).
Gastronomía: Asado, malbec, empanadas mendocinas, dulce de leche artesanal.
Bodegas recomendadas: Catena Zapata, Zuccardi, Achaval Ferrer, Trapiche.
Transporte: Vuelo desde Buenos Aires (~1:30hs), bus (~14hs). Se recomienda alquilar auto para la Ruta del Vino.
Aconcagua: El acceso a la base no requiere permiso especial, pero el ascenso a cumbres sí requiere permiso de la provincia. Precio del permiso temporada alta: U$D 800 aprox.
Alojamiento: Variado. Bodegas con alojamiento (ej: Cavas Wine Lodge), hoteles en ciudad, apart-hoteles y hosteles.
        """,
        metadata={"source": "ficha_mendoza", "region": "cuyo", "categoria": "destino"},
    ),
    Document(
        page_content="""
Salta y el Norte Argentino — Ficha turística
Ubicación: Provincia de Salta, Noroeste Argentino (NOA).
Mejor época: Mayo a octubre (estación seca). Evitar enero-febrero (lluvias intensas).
Atracciones: Tren a las Nubes (4.220 m de altura, reserva anticipada obligatoria), Quebrada de Humahuaca (Patrimonio de la Humanidad UNESCO), Cachi y Los Cardones, Cafayate y los vinos de altura, Salinas Grandes.
Importante — Quebrada de Humahuaca: pertenece a Jujuy, no a Salta. Requiere una excursión de día completo desde Salta capital o se puede dormir en Tilcara o Purmamarca.
Gastronomía: Locro, humita, tamales, empanadas salteñas (diferentes a las mendocinas), vino torrontés.
Transporte: Vuelo desde Buenos Aires (~2hs). Bus (~21hs). Para moverse en la región se recomienda contratar excursiones o alquilar 4x4.
Normativa: Algunas comunidades originarias de Humahuaca cobran arancel de ingreso a sus territorios. Respetar las indicaciones locales es obligatorio.
Alojamiento: Posadas y hoteles boutique en Salta capital y Cafayate. Alojamiento más básico en pueblos de la Quebrada.
        """,
        metadata={"source": "ficha_salta", "region": "noroeste", "categoria": "destino"},
    ),
    Document(
        page_content="""
Normativa de Parques Nacionales Argentinos
Los Parques Nacionales en Argentina son administrados por la Administración de Parques Nacionales (APN).
Aranceles 2024: Residentes argentinos pagan entre $2.000 y $8.000 según el parque. Extranjeros pagan tarifas diferenciales más altas.
Senderos con cupo: Algunos senderos de alta demanda (ej: Laguna de los Tres en El Chaltén, Laguna Torre) requieren registro previo gratuito online en el sistema Trekking Patagonia.
El Chaltén: El ingreso al pueblo no tiene costo; el registro de senderos es gratuito. Es considerado la capital del trekking en Argentina.
Iguazú: El Parque Nacional Iguazú tiene uno de los aranceles más altos del país. El acceso incluye transporte interno en tren ecológico. Las cataratas son visitables todo el año, aunque en temporada de lluvias (noviembre-marzo) el caudal es mayor y algunos circuitos pueden cerrar por seguridad.
        """,
        metadata={"source": "normativa_parques", "region": "nacional", "categoria": "normativa"},
    ),
    Document(
        page_content="""
Guía de presupuesto para viajes en Argentina (2024)
Presupuesto mochilero (económico): $15.000-$30.000 ARS por día.
  - Alojamiento: hostel ($8.000-$15.000/noche)
  - Comida: $5.000-$10.000/día (mercados y comedores locales)
  - Actividades: gratuitas o de bajo costo

Presupuesto medio: $50.000-$100.000 ARS por día.
  - Alojamiento: hotel 3 estrellas o apart ($30.000-$60.000/noche)
  - Comida: $15.000-$25.000/día (restaurantes)
  - Actividades: $10.000-$20.000/día

Presupuesto premium: $150.000+ ARS por día.
  - Alojamiento: hotel 4-5 estrellas o lodge ($80.000+/noche)
  - Comida: $40.000+/día
  - Actividades y experiencias exclusivas: variable

Traslados internos:
  - Bus interprovincial: $20.000-$60.000 (según distancia)
  - Vuelos internos: $60.000-$180.000 (según temporada y anticipación)
  - Alquiler de auto: $40.000-$80.000/día (más combustible)

Tip: Reservar con anticipación en temporada alta (enero-febrero y julio) puede reducir costos de alojamiento hasta 40%.
        """,
        metadata={"source": "guia_presupuesto", "region": "nacional", "categoria": "presupuesto"},
    ),
]


def get_sample_documents() -> list[Document]:
    return SAMPLE_DOCUMENTS
