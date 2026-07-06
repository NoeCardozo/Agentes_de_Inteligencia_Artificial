"""
=============================================================================
COMPONENTES VISUALES DEL CHAT
=============================================================================

Burbujas de chat sencillas para Tkinter.

Idea principal (simple):
  - Cada burbuja es un tk.Label con wraplength (crece solo según el texto).
  - Detrás del Label hay un Canvas que dibuja el rectángulo redondeado y se
    redibuja cuando la burbuja cambia de tamaño.

Componentes:
  - ChatBubble       → Una burbuja de mensaje (usuario, agente, sistema, error)
  - TypingIndicator  → Tres puntos "escribiendo..."
  - ScrollableChat   → Área de chat con scroll vertical
"""

import re
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Constantes visuales
# ---------------------------------------------------------------------------
CHAT_BG = "#F0F2F5"       # Fondo del área de chat

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 8)

BUBBLE_RADIUS = 16        # Radio de las esquinas
BUBBLE_PAD_X = 12         # Padding horizontal del texto dentro de la burbuja
BUBBLE_PAD_Y = 8          # Padding vertical del texto dentro de la burbuja
WRAP_LENGTH = 380         # Ancho máximo del texto antes de hacer wrap (px)

# Estilo por tipo de mensaje
STYLES = {
    "usuario": {"bg": "#0084FF", "fg": "#FFFFFF", "border": "#006DD4", "anchor": "e"},
    "agente":  {"bg": "#FFFFFF", "fg": "#1C1E21", "border": "#DADDE1", "anchor": "w"},
    "sistema": {"bg": "#E4E6EB", "fg": "#65676B", "border": "#CCD0D5", "anchor": "center"},
    "error":   {"bg": "#FFEBEE", "fg": "#B71C1C", "border": "#FFCDD2", "anchor": "w"},
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _draw_rounded_rect(canvas, x1, y1, x2, y2, r, fill, outline, width=1):
    """Dibuja un rectángulo redondeado usando arcos y rectángulos."""
    canvas.delete("all")
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    points = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1, x1 + r, y1,
    ]
    canvas.create_polygon(
        points, smooth=True, splinesteps=24,
        fill=fill, outline=outline, width=width,
    )


def _clean_markdown(text: str) -> str:
    """Limpia los símbolos de markdown para mostrar texto legible en un Label."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)   # **negrita**
    text = re.sub(r"\*([^*]+)\*", r"\1", text)         # *cursiva*
    text = re.sub(r"`([^`]+)`", r"\1", text)           # `código`
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [texto](url)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # títulos
    text = re.sub(r"^\s*[\*\-]\s+", "• ", text, flags=re.MULTILINE)  # listas
    return text.strip()


# ---------------------------------------------------------------------------
# Burbuja de mensaje
# ---------------------------------------------------------------------------

class ChatBubble(tk.Frame):
    """Una burbuja de chat que crece automáticamente según el texto."""

    def __init__(self, parent, role, text):
        style = STYLES.get(role, STYLES["agente"])
        super().__init__(parent, bg=CHAT_BG)
        self.pack(fill=tk.X, pady=4, padx=8)

        self._style = style
        font = FONT if role != "sistema" else FONT_SMALL

        # Contenedor que se alinea a izquierda / derecha / centro.
        # El holder toma el tamaño del Label; el Canvas de fondo se coloca
        # con place() detrás del Label (que se empaqueta normalmente).
        holder = tk.Frame(self, bg=CHAT_BG)
        holder.pack(anchor=style["anchor"], padx=6)

        self._canvas = tk.Canvas(holder, bg=CHAT_BG, highlightthickness=0, borderwidth=0)
        self._canvas.place(x=0, y=0, relwidth=1, relheight=1)

        self._label = tk.Label(
            holder,
            text=_clean_markdown(text),
            bg=style["bg"], fg=style["fg"], font=font,
            wraplength=WRAP_LENGTH, justify=tk.LEFT,
            padx=BUBBLE_PAD_X, pady=BUBBLE_PAD_Y,
        )
        self._label.pack()
        holder.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        """Redibuja el fondo redondeado al tamaño actual de la burbuja."""
        w, h = event.width, event.height
        if w < 4 or h < 4:
            return
        _draw_rounded_rect(
            self._canvas, 1, 1, w - 1, h - 1, BUBBLE_RADIUS,
            fill=self._style["bg"], outline=self._style["border"], width=1,
        )


# ---------------------------------------------------------------------------
# Indicador "escribiendo..."
# ---------------------------------------------------------------------------

class TypingIndicator(tk.Frame):
    """Tres puntos que rebotan mientras el agente responde."""

    def __init__(self, parent):
        super().__init__(parent, bg=CHAT_BG)
        self.pack(fill=tk.X, pady=4, padx=8)

        holder = tk.Frame(self, bg=CHAT_BG)
        holder.pack(anchor="w", padx=6)

        w, h = 64, 34
        self._canvas = tk.Canvas(holder, width=w, height=h, bg=CHAT_BG, highlightthickness=0)
        self._canvas.pack()
        _draw_rounded_rect(self._canvas, 1, 1, w - 1, h - 1, 14,
                           fill="#FFFFFF", outline="#DADDE1", width=1)

        self._dots = [
            self._canvas.create_oval(x - 3, 15, x + 3, 21, fill="#90A4AE", outline="")
            for x in (20, 32, 44)
        ]
        self._frame = 0
        self._animate()

    def _animate(self):
        if not self.winfo_exists():
            return
        for i, dot in enumerate(self._dots):
            phase = (self._frame + i * 3) % 12
            dy = -3 if phase < 6 else 0
            x = (20, 32, 44)[i]
            self._canvas.coords(dot, x - 3, 15 + dy, x + 3, 21 + dy)
            self._canvas.itemconfig(dot, fill="#546E7A" if phase < 6 else "#90A4AE")
        self._frame += 1
        self.after(120, self._animate)


# ---------------------------------------------------------------------------
# Área de chat con scroll
# ---------------------------------------------------------------------------

class ScrollableChat(tk.Frame):
    """Historial de mensajes con scroll vertical."""

    def __init__(self, parent):
        super().__init__(parent, bg=CHAT_BG)
        self._typing = None

        self._canvas = tk.Canvas(self, bg=CHAT_BG, highlightthickness=0, borderwidth=0)
        self._scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._inner = tk.Frame(self._canvas, bg=CHAT_BG)

        self._inner_window = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(
            self._inner_window, width=e.width))
        self._canvas.bind("<Enter>", lambda e: self._canvas.bind_all(
            "<MouseWheel>", self._on_mousewheel))
        self._canvas.bind("<Leave>", lambda e: self._canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def _scroll_to_bottom(self):
        self._inner.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._canvas.yview_moveto(1.0)

    def add_message(self, role, text, animate=True):
        """Agrega una burbuja de mensaje al chat."""
        ChatBubble(self._inner, role, text)
        self._scroll_to_bottom()

    def show_typing(self):
        self.hide_typing()
        self._typing = TypingIndicator(self._inner)
        self._scroll_to_bottom()

    def hide_typing(self):
        if self._typing is not None:
            self._typing.destroy()
            self._typing = None
            self._scroll_to_bottom()
