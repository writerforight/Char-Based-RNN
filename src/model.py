import numpy as np

# ----------------------------
# Math helpers
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    ez = np.exp(z)
    return ez / (np.sum(ez) + 1e-12)

def clip_(arr: np.ndarray, lo: float, hi: float):
    np.clip(arr, lo, hi, out=arr)

def act_forward(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Unknown activation: {name}")

def act_backward(name: str, z: np.ndarray, h: np.ndarray, dh: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return dh * (1.0 - h * h)
    if name == "relu":
        return dh * (z > 0.0).astype(np.float32)
    if name == "sigmoid":
        return dh * (h * (1.0 - h))
    raise ValueError(f"Unknown activation: {name}")


# ----------------------------
# Online (1-step) Char RNN LM
class OnlineCharRNN:
    """
    Online learning: one character step -> one update.
    No seq_len, no truncated BPTT. Gradient flows only through current step.
    """

    def __init__(self, V: int, d: int, H: int, act: str, seed: int = 42):
        self.V = V
        self.d = d
        self.H = H
        self.act = act

        rng = np.random.default_rng(seed)

        # Embedding: E[V, d]
        self.E = (rng.normal(0, 0.01, size=(V, d))).astype(np.float32)

        # RNN core
        self.Wxh = (rng.normal(0, 0.02, size=(H, d))).astype(np.float32)  # (H, d)
        self.Whh = (rng.normal(0, 0.02, size=(H, H))).astype(np.float32)  # (H, H)
        self.bh  = np.zeros((H,), dtype=np.float32)                       # (H,)

        # Output projection
        self.Why = (rng.normal(0, 0.02, size=(V, H))).astype(np.float32)  # (V, H)
        self.by  = np.zeros((V,), dtype=np.float32)                       # (V,)

    def step_loss_grads(self, x_id: int, y_id: int, h_prev: np.ndarray):
        """
        One step:
          x_id -> embedding x
          z = Wxh x + Whh h_prev + bh
          h = act(z)
          logits = Why h + by
          probs = softmax(logits)
          loss = -log probs[y_id]
        Backprop only through this single step.
        """
        x = self.E[x_id]  # (d,)

        z = self.Wxh @ x + self.Whh @ h_prev + self.bh  # (H,)
        h = act_forward(self.act, z)                    # (H,)

        logits = self.Why @ h + self.by                 # (V,)
        probs = softmax(logits)                         # (V,)

        loss = -np.log(probs[y_id] + 1e-12)

        # dL/dlogits
        dlogits = probs.copy()
        dlogits[y_id] -= 1.0  # (V,)

        # Output grads
        dWhy = np.outer(dlogits, h).astype(np.float32)  # (V, H)
        dby  = dlogits.astype(np.float32)               # (V,)

        # Backprop into h
        dh = (self.Why.T @ dlogits).astype(np.float32)  # (H,)

        # Backprop through activation
        dz = act_backward(self.act, z, h, dh).astype(np.float32)  # (H,)

        # RNN grads
        dWxh = np.outer(dz, x).astype(np.float32)       # (H, d)
        dWhh = np.outer(dz, h_prev).astype(np.float32)  # (H, H)
        dbh  = dz.astype(np.float32)                    # (H,)

        # Embedding grads (only row x_id)
        dE_row = (self.Wxh.T @ dz).astype(np.float32)   # (d,)

        grads = {
            "dWhy": dWhy, "dby": dby,
            "dWxh": dWxh, "dWhh": dWhh, "dbh": dbh,
            "x_id": x_id, "dE_row": dE_row
        }
        return float(loss), grads, h

    def sgd_apply(self, grads, lr: float, clip: float = 5.0):
        # Clip big matrices
        clip_(grads["dWhy"], -clip, clip)
        clip_(grads["dby"],  -clip, clip)
        clip_(grads["dWxh"], -clip, clip)
        clip_(grads["dWhh"], -clip, clip)
        clip_(grads["dbh"],  -clip, clip)
        clip_(grads["dE_row"], -clip, clip)

        # Update
        self.Why -= lr * grads["dWhy"]
        self.by  -= lr * grads["dby"]

        self.Wxh -= lr * grads["dWxh"]
        self.Whh -= lr * grads["dWhh"]
        self.bh  -= lr * grads["dbh"]

        self.E[grads["x_id"]] -= lr * grads["dE_row"]

    def sample(self, seed_ids, length: int, temperature: float = 1.0):
        if temperature <= 0:
            temperature = 1e-6

        h = np.zeros((self.H,), dtype=np.float32)

        # Warm-up all seed chars except last
        if seed_ids:
            for x_id in seed_ids[:-1]:
                x = self.E[x_id]
                z = self.Wxh @ x + self.Whh @ h + self.bh
                h = act_forward(self.act, z)

        current = seed_ids[-1] if seed_ids else np.random.randint(0, self.V)
        out = list(seed_ids) if seed_ids else [current]

        for _ in range(length):
            x = self.E[current]
            z = self.Wxh @ x + self.Whh @ h + self.bh
            h = act_forward(self.act, z)
            logits = (self.Why @ h + self.by) / float(temperature)
            p = softmax(logits)
            current = int(np.random.choice(self.V, p=p))
            out.append(current)

        return out

    def next_char_probs_after_prefix(self, prefix_ids, current_id: int):
        """
        For tooltip: compute next-char probs after consuming prefix, then current char.
        """
        h = np.zeros((self.H,), dtype=np.float32)

        for x_id in prefix_ids:
            x = self.E[x_id]
            z = self.Wxh @ x + self.Whh @ h + self.bh
            h = act_forward(self.act, z)

        x = self.E[current_id]
        z = self.Wxh @ x + self.Whh @ h + self.bh
        h = act_forward(self.act, z)

        logits = self.Why @ h + self.by
        return softmax(logits).astype(np.float32)
