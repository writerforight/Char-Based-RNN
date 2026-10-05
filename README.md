# Char-Based-RNN

**Char-Based-RNN is a character-level recurrent neural network (RNN) language model written from
scratch in NumPy, with a Tkinter desktop interface.** You paste any text, it learns to predict the next
character *online* — one character, one gradient step — and then generates new text in the same style.
Hovering over the generated text shows the model's top-10 next-character probabilities as a histogram,
so you can see what it "thought" at every position. It was built by
[Eren Can Almaz](https://writerforight.github.io), an Electrical Engineering student at RWTH Aachen
University, as a from-first-principles study of how RNNs learn.

*The interface is in Turkish; a short glossary is below.*

## Features

- Language model from first principles: embedding, recurrent layer, softmax output, cross-entropy loss
  and backpropagation — no deep-learning framework (`src/model.py`).
- Settings: embedding size *d* (8–128), hidden size *H* (64–512), epochs (1–100), learning rate,
  activation (tanh, ReLU or sigmoid).
- Training runs in a background thread with a live step counter and running loss; it can be stopped.
- Text generation from a seed string, with a sampling **temperature** (low = safe and repetitive,
  high = creative and noisy) and adjustable length (up to 1000 characters).
- **Probability tooltip**: hover over any generated character to see the top-10 candidates for the
  next character and their probabilities.

## How it works

For each position in the text, with characters as integer ids:

```
x_t   = E[c_t]                          embedding of the current character
z_t   = W_xh x_t + W_hh h_{t-1} + b_h
h_t   = act(z_t)                        new hidden state (carried to the next character)
p_t   = softmax(W_hy h_t + b_y)         distribution over the next character
loss  = −log p_t[c_{t+1}]               cross-entropy
```

Training is **online with a one-step gradient**: after every character the gradient of that single
loss is computed and applied immediately with SGD (gradients clipped to ±5). The hidden state is
carried from step to step, but the gradient does not flow back into earlier steps (no backpropagation
through time). This makes every update cheap and the learning easy to follow. The trade-off is that
long-range structure is learned only weakly and the loss is noisy — small learning rates work best.
Generation samples from `softmax(logits / T)` one character at a time.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy
python main.py
```

Tkinter ships with Python on Windows and macOS. On Linux you may need the system package, for example
`sudo apt install python3-tk`.

## Interface glossary (Turkish → English)

| Turkish | English |
|---|---|
| Eğitim metnini buraya yapıştır | Paste the training text here |
| Eğitimi Başlat / Durdur | Start / stop training |
| Metin Üretimi · Üret | Text generation · Generate |
| Seed (başlangıç) | Seed text (start) |
| Üretim uzunluğu | Generation length |
| Softmax T (sampling) | Sampling temperature |

## Project structure

```text
main.py          entry point
src/model.py     the RNN: forward step, one-step backprop, SGD, sampling, next-character probabilities
src/app.py       Tkinter interface, background training thread, generation
src/tooltip.py   hover histogram of the top-10 next-character probabilities
```

## Author

**Eren Can Almaz** — Electrical Engineering (Elektrotechnik) student at RWTH Aachen University.
GitHub [@writerforight](https://github.com/writerforight) · website
[writerforight.github.io](https://writerforight.github.io). Related project:
[Neural Space Deformation](https://github.com/writerforight/mlp-space-deformation), an interactive
visualization of how neural networks deform space, also written from first principles.

## License

MIT — see [LICENSE](LICENSE).
