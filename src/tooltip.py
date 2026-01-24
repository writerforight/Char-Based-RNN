# src/tooltip.py
import tkinter as tk
from tkinter import ttk


class HoverHistogramTooltip:
    """
    Tooltip bubble that shows a Top-K next-character probability histogram.

    Requirements implemented:
    - Appears ABOVE the cursor (falls back below if there is no space).
    - Green bars proportional to probability (relative to max in Top-K).
    - Shows characters (with visible symbols for whitespace).
    - Shows percentages and keeps tooltip inside screen bounds.
    """

    def __init__(self, root: tk.Tk, width: int = 380, height: int = 180):
        self.root = root
        self.width = width
        self.height = height

        self.tip: tk.Toplevel | None = None
        self.frame: ttk.Frame | None = None
        self.label: ttk.Label | None = None
        self.canvas: tk.Canvas | None = None

    # ----------------------------
    # Lifecycle
    def ensure(self):
        if self.tip is not None:
            return

        self.tip = tk.Toplevel(self.root)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)

        self.frame = ttk.Frame(self.tip, padding=8, relief="solid", borderwidth=1)
        self.frame.grid(row=0, column=0, sticky="nsew")

        self.label = ttk.Label(self.frame, text="", justify="left")
        self.label.grid(row=0, column=0, sticky="w")

        self.canvas = tk.Canvas(
            self.frame,
            width=self.width,
            height=self.height,
            highlightthickness=0,
            borderwidth=0,
        )
        self.canvas.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # Make sure dimensions are computed
        self.tip.update_idletasks()

    def hide(self):
        if self.tip is not None:
            try:
                self.tip.destroy()
            except Exception:
                pass
        self.tip = None
        self.frame = None
        self.label = None
        self.canvas = None

    # ----------------------------
    # Positioning (ABOVE cursor)
    def move_to(self, x_root: int, y_root: int, dx: int = 16, dy: int = 10):
        """
        Position tooltip ABOVE the cursor using screen coordinates.
        If there is no room above, it falls back below.
        Also clamps within screen bounds.
        """
        self.ensure()
        assert self.tip is not None

        # Get actual tooltip size (fallback to configured size)
        try:
            self.tip.update_idletasks()
            tip_w = self.tip.winfo_width() or (self.width + 20)
            tip_h = self.tip.winfo_height() or (self.height + 40)
        except Exception:
            tip_w = self.width + 20
            tip_h = self.height + 40

        screen_w = self.tip.winfo_screenwidth()
        screen_h = self.tip.winfo_screenheight()

        # Preferred: ABOVE cursor
        x = x_root + dx
        y = y_root - tip_h - dy

        # Clamp horizontally
        if x + tip_w > screen_w - 5:
            x = screen_w - tip_w - 5
        if x < 5:
            x = 5

        # If above goes off-screen, fall back below
        if y < 5:
            y = y_root + dy

        # Clamp vertically
        if y + tip_h > screen_h - 5:
            y = screen_h - tip_h - 5
        if y < 5:
            y = 5

        try:
            self.tip.geometry(f"+{x}+{y}")
        except Exception:
            pass

    # ----------------------------
    # Rendering
    def render_topk(self, current_char: str, items):
        """
        Render the histogram.

        Args:
            current_char: hovered character
            items: list of (char, prob) sorted descending
        """
        self.ensure()
        assert self.label is not None and self.canvas is not None

        self.label.config(text=f"'{self._pretty_char(current_char)}' sonrası (Top {len(items)}):")

        c = self.canvas
        c.delete("all")

        if not items:
            return

        W = int(c["width"])
        H = int(c["height"])

        # Layout
        left = 78   # space for char labels
        right = 14  # space for percent labels
        top = 12
        bottom = 12

        bar_area_w = max(10, W - left - right)
        n = len(items)

        gap = 6
        # distribute bars with gaps
        usable_h = max(10, H - top - bottom - (n - 1) * gap)
        bar_h = max(10, usable_h // n)

        # Scale by max prob among Top-K
        max_p = max(p for _, p in items)
        if max_p <= 0.0:
            max_p = 1e-12

        # Colors
        green = "#2ecc71"
        baseline = "#999999"
        text_col = "#111111"

        # Baseline
        c.create_line(left, top - 4, left, H - bottom + 4, fill=baseline)

        for i, (ch, p) in enumerate(items):
            y0 = top + i * (bar_h + gap)
            y1 = y0 + bar_h

            frac = float(p) / float(max_p)
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

            x0 = left
            x1 = left + int(bar_area_w * frac)

            # Character label
            c.create_text(
                left - 52,
                (y0 + y1) // 2,
                text=self._pretty_char(ch),
                anchor="w",
                fill=text_col
            )

            # Green bar proportional to probability
            c.create_rectangle(
                x0, y0, x1, y1,
                fill=green,
                outline=""
            )

            # Percentage label
            c.create_text(
                W - right,
                (y0 + y1) // 2,
                text=f"{p * 100:.2f}%",
                anchor="e",
                fill=text_col
            )

    # ----------------------------
    # Helpers
    @staticmethod
    def _pretty_char(ch: str) -> str:
        """
        Make whitespace and control characters visible.
        """
        if ch == " ":
            return "␠"
        if ch == "\n":
            return "↵"
        if ch == "\t":
            return "↹"
        if ch == "\r":
            return "␍"
        return ch
