# Agente de Turismo de Argentina

Agente conversacional de turismo argentino con **LangGraph**, **Gemini** y una interfaz gráfica en **Tkinter**. Incluye RAG con guías locales, consultas a APIs públicas (Georef y Wikipedia) y herramientas para itinerarios y vuelos de demostración.

## Requisitos previos

- **Python 3.10 o superior** ([descargar Python](https://www.python.org/downloads/))
- **Git** (opcional, solo si vas a clonar el repositorio)
- Una **API key de Google AI Studio** ([obtener API key](https://aistudio.google.com/apikey))

## Paso a paso para ejecutar la aplicación

### 1. Obtener el código

Si aún no tenés el proyecto en tu máquina, clonalo:

```bash
git clone https://github.com/NoeCardozo/Agentes_de_Inteligencia_Artificial.git
cd Agentes_de_Inteligencia_Artificial
```

Si ya tenés la carpeta del proyecto, entrá a ella:

```bash
cd ruta/a/tu/proyecto
```

### 2. Crear el entorno virtual

El entorno virtual aísla las dependencias del proyecto para no mezclarlas con el resto del sistema.

**Windows (CMD o PowerShell):**

```bash
python -m venv venv
```

**Windows (Git Bash):**

```bash
python -m venv venv
```

**Linux / macOS:**

```bash
python3 -m venv venv
```

Esto crea una carpeta `venv/` dentro del proyecto.

### 3. Activar el entorno virtual

Activá el entorno **antes** de instalar dependencias o ejecutar la app. La terminal debe mostrar `(venv)` al inicio de la línea.

**Windows (CMD):**

```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```bash
venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**

```bash
source venv/Scripts/activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Instalar las dependencias

Con el entorno virtual **activado**, instalá los paquetes del proyecto:

```bash
pip install -r requirements.txt
```

### 5. Configurar variables de entorno

La aplicación lee su configuración desde un archivo `.env` en la raíz del proyecto. Ese archivo **no se sube a GitHub** (está en `.gitignore`) porque puede contener claves privadas.

#### Cómo crear el archivo `.env`

**Opción A — Copiar la plantilla (recomendado):**

**Windows:**

```bash
copy .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

**Opción B — Crearlo manualmente:**

1. En la raíz del proyecto, creá un archivo llamado `.env` (sin nombre antes del punto).
2. Copiá el contenido de `.env.example` dentro.
3. Guardá el archivo.

#### Cómo editar las variables

Abrí `.env` con cualquier editor de texto (Bloc de notas, VS Code, Cursor, etc.) y completá cada línea con el formato `NOMBRE=valor`, **sin comillas** y **sin espacios** alrededor del `=`.

Ejemplo de un `.env` listo para usar:

```env
GOOGLE_API_KEY=AQ.tu_clave_real_aqui
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
```

#### Variables disponibles y para qué sirve cada una

| Variable | ¿Obligatoria? | Para qué sirve |
|----------|---------------|----------------|
| `GOOGLE_API_KEY` | **Sí** | Clave de acceso a Google AI Studio. La usa el agente para **generar respuestas** con Gemini y el módulo RAG para **crear embeddings** de las guías turísticas. Sin esta clave la app no inicia. |
| `GEMINI_MODEL` | No | Modelo de lenguaje que usa el agente conversacional. Por defecto: `gemini-2.5-flash`. Controla la **calidad, velocidad y cuota** de las respuestas del chat. |
| `EMBEDDING_MODEL` | No | Modelo que convierte texto en vectores para el **RAG** (búsqueda en guías de `data/guias/`). Por defecto: `gemini-embedding-001`. Si lo cambiás, ChromaDB se reindexa automáticamente. |

**Detalle de cada variable:**

- **`GOOGLE_API_KEY`** — Obtenela en [Google AI Studio](https://aistudio.google.com/apikey). Puede empezar con `AIza...` (formato antiguo) o `AQ....` (formato nuevo). Es la única variable que **tenés que cambiar sí o sí** antes de ejecutar la app.

- **`GEMINI_MODEL`** — Define qué modelo de Gemini responde al usuario. Recomendado: `gemini-2.5-flash` (rápido y con buena cuota gratuita). Si recibís error 429 (cuota agotada), probá otro modelo disponible en tu cuenta.

- **`EMBEDDING_MODEL`** — Define qué modelo genera los vectores para buscar en las guías turísticas. No afecta las respuestas directas del chat, solo la herramienta `consultar_guia_turistica`. El valor `text-embedding-004` ya no funciona; usá `gemini-embedding-001`.

#### Dónde se usan en el código

- `GOOGLE_API_KEY` y `GEMINI_MODEL` → `app/agent.py` (agente LangGraph)
- `GOOGLE_API_KEY` y `EMBEDDING_MODEL` → `app/rag.py` (índice vectorial ChromaDB)
- Las tres variables se cargan en → `app/config.py` con `python-dotenv`

> **Importante:** no subas el archivo `.env` a GitHub. Contiene tu clave privada. Solo compartí `.env.example`, que es la plantilla sin datos sensibles.

### 6. Ejecutar la aplicación

Con el entorno virtual **activado** y las dependencias **instaladas**, iniciá la app:

```bash
python main.py
```

Se abrirá la ventana del chat. La primera vez puede tardar unos segundos mientras se indexan las guías turísticas en ChromaDB.

### 7. Desactivar el entorno virtual (opcional)

Cuando termines de usar la aplicación:

```bash
deactivate
```

## Resumen rápido

```bash
# 1. Entrar al proyecto
cd Agentes_de_Inteligencia_Artificial

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno (elegí el comando según tu sistema)
source venv/Scripts/activate        # Git Bash / Windows
# source venv/bin/activate          # Linux / macOS

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Configurar .env
copy .env.example .env              # Windows
# cp .env.example .env              # Linux / macOS
# Editar .env y poner tu GOOGLE_API_KEY

# 6. Ejecutar
python main.py
```

## Estructura del proyecto

```
├── main.py              # Punto de entrada
├── requirements.txt     # Dependencias Python
├── .env.example         # Plantilla de configuración
├── app/
│   ├── agent.py         # Agente LangGraph + Gemini
│   ├── tools.py         # Herramientas del agente
│   ├── rag.py           # RAG con ChromaDB
│   ├── argentina_api.py # APIs Georef y Wikipedia
│   ├── ui.py            # Interfaz Tkinter
│   └── chat_widgets.py  # Burbujas de chat
└── data/guias/          # Guías turísticas (Buenos Aires, Mendoza, etc.)
```

## Solución de problemas

| Problema | Posible solución |
|----------|------------------|
| `GOOGLE_API_KEY no configurada` | Creá `.env` desde `.env.example` y agregá tu API key |
| Error 429 / cuota agotada | Probá con `gemini-2.5-flash` en `.env` o esperá unos minutos |
| `python` no reconocido | Usá `py main.py` en Windows o instalá Python y marcá "Add to PATH" |
| PowerShell bloquea `Activate.ps1` | Ejecutá `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` o usá CMD/Git Bash |

## Licencia

Proyecto educativo — Agente de Inteligencia Artificial.
