"""
=============================================================================
INTERFAZ GRÁFICA (Tkinter)
=============================================================================

Ventana principal del chat. Conecta la UI con el agente de IA.

Flujo de la aplicación:
  1. Al abrir: carga el RAG y crea el agente en un hilo secundario
     (para no congelar la ventana mientras carga).
  2. El usuario escribe y presiona Enter o "Enviar".
  3. El mensaje se muestra en el chat y se envía al agente en otro hilo.
  4. Mientras espera, se muestra el indicador "escribiendo..." (tres puntos).
  5. La respuesta aparece en una burbuja del agente (lado izquierdo).

¿Por qué hilos (threading)?
  Las llamadas a Gemini tardan varios segundos. Si se hicieran en el hilo
  principal, la ventana se congelaría. Los hilos permiten que la UI siga
  respondiendo mientras el agente piensa.
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from app.agent import create_tourism_agent, invoke_agent
from app.chat_widgets import CHAT_BG, ScrollableChat
from app.errors import format_agent_error
from app.rag import get_vectorstore

# Mensaje de bienvenida que ve el usuario al iniciar
MENSAJE_BIENVENIDA = (
    "¡Hola! Soy tu guía de turismo de **Argentina**. "
    "Puedo consultar ciudades con datos oficiales (API Georef), "
    "buscar en guías de Buenos Aires, Mendoza, Córdoba, Bariloche y Salta, "
    "armar itinerarios y buscar vuelos."
)


class TourismAgentApp:
    """Ventana principal de la aplicación de turismo."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Agente de Turismo")
        self.root.geometry("780x620")
        self.root.minsize(600, 480)
        self.root.configure(bg=CHAT_BG)

        self.agent = None          # Se crea al iniciar (en segundo plano)
        self.thread_id = "session-1"  # ID de la conversación actual
        self._busy = False         # True mientras espera respuesta del agente

        self._build_ui()
        self._init_agent_async()

    def _build_ui(self):
        """Construye todos los elementos visuales de la ventana."""
        style = ttk.Style()
        style.theme_use("clam")

        # --- Cabecera: título y estado ---
        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill=tk.X)

        ttk.Label(header, text="Agente de Turismo", font=("Segoe UI", 16, "bold")).pack(
            side=tk.LEFT
        )
        self.status_label = ttk.Label(header, text="Inicializando...", foreground="#666")
        self.status_label.pack(side=tk.RIGHT)

        # --- Área de chat con burbujas y scroll ---
        chat_frame = ttk.Frame(self.root, padding=(8, 0))
        chat_frame.pack(fill=tk.BOTH, expand=True)
        self.chat = ScrollableChat(chat_frame)
        self.chat.pack(fill=tk.BOTH, expand=True)

        # --- Campo de texto y botón enviar ---
        input_frame = ttk.Frame(self.root, padding=12)
        input_frame.pack(fill=tk.X)

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            input_frame, textvariable=self.input_var, font=("Segoe UI", 10)
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.input_entry.bind("<Return>", lambda _e: self._on_send())

        self.send_btn = ttk.Button(input_frame, text="Enviar", command=self._on_send)
        self.send_btn.pack(side=tk.RIGHT)
        self.send_btn.state(["disabled"])  # Deshabilitado hasta que el agente esté listo

        # --- Pie con ejemplos de consultas ---
        footer = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        footer.pack(fill=tk.X)
        ttk.Label(
            footer,
            text="Prueba: 'Contame sobre Mendoza' · 'Provincias de Argentina' · "
            "'Vuelos Buenos Aires a Bariloche' · 'Itinerario 4 días en Salta'",
            foreground="#888",
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W)

    # -------------------------------------------------------------------------
    # Helpers de la interfaz
    # -------------------------------------------------------------------------

    def _append_chat(self, role: str, text: str):
        """Agrega un mensaje al chat. role: usuario | agente | sistema | error"""
        self.chat.add_message(role, text)

    def _set_status(self, text: str):
        """Actualiza el texto de estado en la esquina superior derecha."""
        self.status_label.config(text=text)

    def _set_busy(self, busy: bool):
        """Activa/desactiva controles mientras el agente procesa una consulta."""
        self._busy = busy
        if busy:
            self.send_btn.state(["disabled"])
            self.input_entry.config(state=tk.DISABLED)
            if self.status_label.cget("text") == "Listo":
                self._set_status("Pensando...")
        else:
            self.send_btn.state(["!disabled"])
            self.input_entry.config(state=tk.NORMAL)
            self._set_status("Listo")
            self.input_entry.focus_set()

    # -------------------------------------------------------------------------
    # Inicialización del agente (en segundo plano)
    # -------------------------------------------------------------------------

    def _init_agent_async(self):
        """
        Carga el RAG y crea el agente sin bloquear la ventana.

        Se ejecuta en un hilo separado porque indexar ChromaDB y conectar
        a Gemini puede tardar varios segundos.
        """
        def worker():
            try:
                get_vectorstore()                    # Paso 1: preparar RAG
                self.agent = create_tourism_agent()  # Paso 2: crear agente
                # Volver al hilo principal de Tkinter para actualizar la UI
                self.root.after(0, lambda: self._on_ready(MENSAJE_BIENVENIDA))
            except Exception as exc:
                error_msg = format_agent_error(exc)
                self.root.after(0, lambda msg=error_msg: self._on_init_error(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ready(self, welcome: str):
        """Se llama cuando el agente terminó de cargar correctamente."""
        self._append_chat("sistema", welcome)
        self._set_busy(False)
        self.input_entry.focus_set()

    def _on_init_error(self, error: str):
        """Se llama si falló la carga del agente o del RAG."""
        self.status_label.config(text="Error")
        self._append_chat("error", error)
        messagebox.showerror(
            "Error de inicialización",
            f"No se pudo iniciar el agente:\n\n{error}\n\n"
            "Verifica que GOOGLE_API_KEY esté en el archivo .env",
        )

    # -------------------------------------------------------------------------
    # Envío y recepción de mensajes
    # -------------------------------------------------------------------------

    def _on_send(self):
        """Se ejecuta cuando el usuario presiona Enter o el botón Enviar."""
        if self._busy or not self.agent:
            return

        message = self.input_var.get().strip()
        if not message:
            return

        # Mostrar mensaje del usuario y limpiar el campo de entrada
        self.input_var.set("")
        self._append_chat("usuario", message)
        self._set_busy(True)
        self._set_status("Consultando a Gemini...")
        self.chat.show_typing()  # Tres puntos animados mientras espera

        def worker():
            """Llama al agente en segundo plano para no congelar la UI."""
            try:
                response = invoke_agent(self.agent, message, self.thread_id)
                self.root.after(0, lambda r=response: self._on_response(r))
            except Exception as exc:
                error_msg = format_agent_error(exc)
                self.root.after(0, lambda msg=error_msg: self._on_response_error(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_response(self, response: str):
        """Muestra la respuesta del agente en el chat."""
        self.chat.hide_typing()
        self._append_chat("agente", response)
        self._set_busy(False)

    def _on_response_error(self, error: str):
        """Muestra un error en el chat si falló la consulta al agente."""
        self.chat.hide_typing()
        self._append_chat("error", error)
        self._set_busy(False)


def run_app():
    """Crea la ventana principal y entra en el bucle de eventos de Tkinter."""
    root = tk.Tk()
    TourismAgentApp(root)
    root.mainloop()
