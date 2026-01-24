import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import queue
import time

from .model import OnlineCharRNN
from .tooltip import HoverHistogramTooltip


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Online Char-RNN (1-step learning) — NumPy + Tkinter")
        self.root.geometry("1100x720")

        self.msg_q = queue.Queue()

        self.model = None
        self.char2id = None
        self.id2char = None
        self.train_thread = None
        self.stop_flag = False

        # tooltip
        self.tooltip = HoverHistogramTooltip(self.root)
        self.tip_last_idx = None
        self.tip_after_id = None

        # UI vars
        self.embed_dim = tk.IntVar(value=32)
        self.hidden_size = tk.IntVar(value=256)
        self.epochs = tk.IntVar(value=10)
        self.lr = tk.DoubleVar(value=0.005)
        self.activation = tk.StringVar(value="tanh")
        self.temperature = tk.DoubleVar(value=1.0)
        self.gen_len = tk.IntVar(value=100)

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        self.frm = ttk.Frame(self.root, padding=12)
        self.frm.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.frm.columnconfigure(0, weight=1)

        ttk.Label(self.frm, text="Eğitim metnini buraya yapıştır:").grid(row=0, column=0, sticky="w")
        self.txt = tk.Text(self.frm, height=14, wrap="word")
        self.txt.grid(row=1, column=0, sticky="nsew", pady=(6, 10))
        self.frm.rowconfigure(1, weight=1)

        ctrl = ttk.LabelFrame(self.frm, text="Ayarlar (online learning: her adım 1 karakter)", padding=10)
        ctrl.grid(row=2, column=0, sticky="ew")
        for c in range(10):
            ctrl.columnconfigure(c, weight=1)

        ttk.Label(ctrl, text="Embedding d:").grid(row=0, column=0, sticky="e")
        ttk.Combobox(ctrl, textvariable=self.embed_dim, values=[8, 16, 32, 64, 128],
                     width=7, state="readonly").grid(row=0, column=1, sticky="w")

        ttk.Label(ctrl, text="Hidden H:").grid(row=0, column=2, sticky="e")
        ttk.Combobox(ctrl, textvariable=self.hidden_size, values=[64, 128, 256, 512],
                     width=7, state="readonly").grid(row=0, column=3, sticky="w")

        ttk.Label(ctrl, text="Epochs (max 100):").grid(row=0, column=4, sticky="e")
        ttk.Combobox(ctrl, textvariable=self.epochs, values=[1, 2, 3, 5, 10, 20, 50, 100],
                     width=7, state="readonly").grid(row=0, column=5, sticky="w")

        ttk.Label(ctrl, text="LR:").grid(row=0, column=6, sticky="e")
        ttk.Entry(ctrl, textvariable=self.lr, width=10).grid(row=0, column=7, sticky="w")

        ttk.Label(ctrl, text="Activation h:").grid(row=1, column=0, sticky="e", pady=(8, 0))
        ttk.Combobox(ctrl, textvariable=self.activation, values=["tanh", "relu", "sigmoid"],
                     width=9, state="readonly").grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(ctrl, text="Softmax T (sampling):").grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(ctrl, textvariable=self.temperature, width=10).grid(row=1, column=3, sticky="w", pady=(8, 0))

        btns = ttk.Frame(self.frm)
        btns.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        btns.columnconfigure(2, weight=1)

        self.btn_train = ttk.Button(btns, text="Eğitimi Başlat", command=self.start_training)
        self.btn_train.grid(row=0, column=0, sticky="w")

        self.btn_stop = ttk.Button(btns, text="Durdur", command=self.stop_training, state="disabled")
        self.btn_stop.grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.pbar = ttk.Progressbar(btns, orient="horizontal", mode="determinate")
        self.pbar.grid(row=0, column=2, sticky="ew", padx=10)

        self.lbl_status = ttk.Label(btns, text="Hazır.")
        self.lbl_status.grid(row=0, column=3, sticky="e")

        self.gen = ttk.LabelFrame(self.frm, text="Metin Üretimi", padding=10)
        self.gen.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.gen.columnconfigure(1, weight=1)

        ttk.Label(self.gen, text="Seed (başlangıç):").grid(row=0, column=0, sticky="e")
        self.ent_seed = ttk.Entry(self.gen)
        self.ent_seed.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        ttk.Label(self.gen, text="Üretim uzunluğu (max 1000):").grid(row=0, column=2, sticky="e")
        ttk.Combobox(self.gen, textvariable=self.gen_len, values=[50, 100, 200, 300, 500, 800, 1000],
                     width=8, state="readonly").grid(row=0, column=3, sticky="w", padx=(6, 8))

        self.btn_gen = ttk.Button(self.gen, text="Üret", command=self.generate, state="disabled")
        self.btn_gen.grid(row=0, column=4, sticky="w")

        self.out = tk.Text(self.gen, height=7, wrap="word", state="disabled")
        self.out.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(8, 0))

        # tooltip bindings
        self.out.bind("<Motion>", self._on_out_motion)
        self.out.bind("<Leave>", self._on_out_leave)

        note = ttk.Label(self.frm, text="Not: Online learning (seq_len=1) daha gürültülü öğrenir; LR'yi küçük tutmak genelde daha iyi.")
        note.grid(row=5, column=0, sticky="w", pady=(8, 0))

    def stop_training(self):
        self.stop_flag = True
        self.btn_stop.config(state="disabled")
        self.lbl_status.config(text="Durduruluyor...")

    def _set_controls(self, training: bool):
        if training:
            self.btn_train.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.btn_gen.config(state="disabled")
        else:
            self.btn_train.config(state="normal")
            self.btn_stop.config(state="disabled")

    def start_training(self):
        text = self.txt.get("1.0", "end").rstrip("\n")
        if len(text) < 200:
            messagebox.showerror("Metin çok kısa", "Daha uzun bir eğitim metni yapıştır (en az ~200 karakter).")
            return

        chars = sorted(list(set(text)))
        V = len(chars)
        if V < 5:
            messagebox.showerror("Vocab çok küçük", "Metinde çok az farklı karakter var.")
            return

        self.char2id = {ch: i for i, ch in enumerate(chars)}
        self.id2char = {i: ch for ch, i in self.char2id.items()}

        d = int(self.embed_dim.get())
        H = int(self.hidden_size.get())
        act = str(self.activation.get())

        epochs = int(self.epochs.get())
        if epochs < 1:
            epochs = 1
        if epochs > 100:
            epochs = 100
            self.epochs.set(100)

        lr = float(self.lr.get())
        if lr <= 0:
            messagebox.showerror("LR hatası", "LR pozitif olmalı.")
            return

        self.model = OnlineCharRNN(V=V, d=d, H=H, act=act, seed=42)

        steps_per_epoch = len(text) - 1
        total_steps = epochs * steps_per_epoch

        self.pbar["value"] = 0
        self.pbar["maximum"] = total_steps

        self.stop_flag = False
        self._set_controls(training=True)
        self.lbl_status.config(text=f"Vocab={V} | Online eğitim başlıyor...")
        self.btn_gen.config(state="disabled")

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(text, epochs, lr, total_steps),
            daemon=True
        )
        self.train_thread.start()

    def _train_worker(self, text: str, epochs: int, lr: float, total_steps: int):
        try:
            xs_all = [self.char2id[ch] for ch in text[:-1]]
            ys_all = [self.char2id[ch] for ch in text[1:]]

            h = np.zeros((self.model.H,), dtype=np.float32)

            step = 0
            t0 = time.time()
            losses_window = []

            for _ep in range(epochs):
                for i in range(len(xs_all)):
                    if self.stop_flag:
                        self.msg_q.put(("stopped", step))
                        return

                    loss, grads, h = self.model.step_loss_grads(xs_all[i], ys_all[i], h)
                    self.model.sgd_apply(grads, lr=lr, clip=5.0)

                    losses_window.append(loss)
                    step += 1

                    if step % 200 == 0 or step == total_steps:
                        avg_loss = float(np.mean(losses_window[-2000:])) if losses_window else float(loss)
                        elapsed = time.time() - t0
                        self.msg_q.put(("progress", step, total_steps, avg_loss, elapsed))

            self.msg_q.put(("done",))
        except Exception as e:
            self.msg_q.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                kind = msg[0]

                if kind == "progress":
                    _, step, total_steps, avg_loss, elapsed = msg
                    self.pbar["value"] = step
                    self.lbl_status.config(text=f"Step {step}/{total_steps} | loss≈{avg_loss:.4f} | {elapsed:.1f}s")

                elif kind == "done":
                    self.pbar["value"] = self.pbar["maximum"]
                    self.lbl_status.config(text="Eğitim bitti ✅")
                    self._set_controls(training=False)
                    self.btn_gen.config(state="normal")

                elif kind == "stopped":
                    _, step = msg
                    self.lbl_status.config(text=f"Durduruldu. Step={step}")
                    self._set_controls(training=False)
                    self.btn_gen.config(state="normal" if self.model is not None else "disabled")

                elif kind == "error":
                    _, err = msg
                    self._set_controls(training=False)
                    messagebox.showerror("Hata", err)

        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def generate(self):
        if self.model is None or self.char2id is None:
            return

        n = int(self.gen_len.get())
        if n < 1:
            n = 1
        if n > 1000:
            n = 1000
            self.gen_len.set(1000)

        seed = self.ent_seed.get() or ""

        if len(seed) == 0:
            seed_ids = [np.random.randint(0, self.model.V)]
        else:
            seed_ids = []
            for ch in seed:
                if ch in self.char2id:
                    seed_ids.append(self.char2id[ch])
                else:
                    seed_ids.append(np.random.randint(0, self.model.V))

        T = float(self.temperature.get())
        out_ids = self.model.sample(seed_ids, length=n, temperature=T)
        out_text = "".join(self.id2char[i] for i in out_ids)

        self.out.config(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", out_text)
        self.out.config(state="disabled")

    # ----------------------------
    # Tooltip hover logic
    def _on_out_leave(self, event=None):
        if self.tip_after_id is not None:
            try:
                self.root.after_cancel(self.tip_after_id)
            except Exception:
                pass
            self.tip_after_id = None

        self.tip_last_idx = None
        self.tooltip.hide()

    def _on_out_motion(self, event):
        if self.model is None or self.char2id is None:
            self._on_out_leave()
            return

        if self.tip_after_id is not None:
            try:
                self.root.after_cancel(self.tip_after_id)
            except Exception:
                pass

        self.tip_after_id = self.root.after(60, lambda e=event: self._show_probs_tooltip(e))

    def _show_probs_tooltip(self, event):
        self.tip_after_id = None

        idx = self.out.index(f"@{event.x},{event.y}")
        ch = self.out.get(idx)
        if not ch or ch == "\n":
            self._on_out_leave()
            return

        if self.tip_last_idx == idx:
            self.tooltip.move_to(event.x_root, event.y_root)
            return
        self.tip_last_idx = idx

        out_text = self.out.get("1.0", "end-1c")
        if not out_text:
            self._on_out_leave()
            return

        abs_pos = self._tk_index_to_abs(out_text, idx)
        if abs_pos is None or abs_pos < 0 or abs_pos >= len(out_text):
            self._on_out_leave()
            return

        cur = out_text[abs_pos]
        if cur not in self.char2id:
            self._on_out_leave()
            return

        prefix_ids = []
        for c in out_text[:abs_pos]:
            if c in self.char2id:
                prefix_ids.append(self.char2id[c])

        probs = self.model.next_char_probs_after_prefix(prefix_ids, self.char2id[cur])

        topk = 10
        top_ids = np.argsort(-probs)[:topk]
        items = [(self.id2char[i], float(probs[i])) for i in top_ids]

        self.tooltip.move_to(event.x_root, event.y_root)
        self.tooltip.render_topk(cur, items)

    def _tk_index_to_abs(self, text: str, tk_index: str):
        try:
            line_s, col_s = tk_index.split(".")
            line = int(line_s)
            col = int(col_s)
        except Exception:
            return None

        lines = text.split("\n")
        if line < 1 or line > len(lines):
            return None

        abs_pos = 0
        for i in range(line - 1):
            abs_pos += len(lines[i]) + 1
        abs_pos += col
        return abs_pos
