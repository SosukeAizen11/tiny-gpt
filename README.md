# TinyGPT

A small GPT-style language model built from scratch for learning and demonstrating how modern LLMs work internally. It uses a real Transformer architecture with multi-head attention, positional embeddings, and autoregressive text generation — all in pure PyTorch.

---

## Requirements

- Python **3.9 or higher** (developed and tested on Python 3.11.9)
  - The code uses `list[str]` / `list[int]` type hint syntax that requires Python 3.9+.
  - If you don't have Python 3.9+, download it from: https://www.python.org/downloads/
- pip

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SosukeAizen11/tiny-gpt.git
cd tiny-gpt
```

### 2. Create a virtual environment

If you only have one Python version installed, just run:

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**If you have multiple Python versions installed**, be explicit so you don't accidentally use an old one:

**Windows** (uses the `py` launcher):
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

> Replace `3.11` with whichever version you have (3.9, 3.10, 3.12 — all work).
> To see all installed versions on Windows, run: `py --list`
> To see all installed versions on macOS/Linux, run: `ls /usr/bin/python3*`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> This installs `torch==2.14.0` and `numpy==2.4.6`. No GPU required — everything runs on CPU.

---

## Running TinyGPT

### Option A — Train + Chat (recommended first time)

This trains the model from scratch and then opens an interactive chat:

```bash
python main.py
```

Training takes about **30–60 seconds** on CPU. The model early-stops once the loss drops below 0.3 (usually around epoch 55–60). The trained model is saved to `checkpoints/tinygpt_v2.pt`.

**Example output:**
```
Epoch   1 | Loss: 6.2929 | LR: 0.001000
Epoch  10 | Loss: 3.7531 | LR: 0.000997
...
Early stop at epoch 59 (loss=0.2941 < 0.3)

Training complete!
Model saved to: checkpoints/tinygpt_v2.pt
```

### Option B — Chat only (skip retraining)

If `checkpoints/tinygpt_v2.pt` already exists (either you trained before, or it was included in the repo), you can jump straight into chat:

```bash
python chat_v2.py
```

---

## Chatting with TinyGPT

Once the chat starts, just type a message and press Enter:

```
You: hello
TinyGPT: hello ! it is nice to talk with you .

You: what is python?
TinyGPT: python is a programming language known for its simple and readable syntax .

You: what is machine learning?
TinyGPT: machine learning is a method that allows computers to learn patterns from data .

You: quit
Goodbye!
```

**Available commands inside the chat:**

| Command | Description |
|---|---|
| `quit` | Exit the chat |
| `debug <text>` | Inspect model internals (attention weights, logits, top predictions) |

---

## Project Structure

```
tiny-gpt/
├── main.py               # Train the model + interactive chat
├── chat_v2.py            # Chat using a saved checkpoint (no retraining)
├── chat.py               # Legacy chat loader (loads tinygpt.pt)
├── requirements.txt      # Python dependencies
├── data/
│   └── conversations.txt # Training data (Q&A conversation pairs)
├── checkpoints/
│   └── tinygpt_v2.pt     # Saved model checkpoint (after training)
└── src/
    ├── model.py              # TinyGPT model definition
    ├── tokenizer.py          # Word-level tokenizer
    ├── dataset.py            # Dataset builder
    ├── generation.py         # Autoregressive text generation
    ├── attention.py          # Multi-head causal self-attention
    ├── transformer_block.py  # Transformer block (attention + FFN)
    ├── feed_forward.py       # Feed-forward network
    ├── embeddings.py         # Token embeddings
    ├── positional_embedding.py # Learned positional embeddings
    └── checkpoint.py         # Save / load model checkpoints
```

---

## How It Works

TinyGPT is a decoder-only Transformer (same family as GPT). Here's the pipeline:

1. **Tokenization** — Input text is split into word-level tokens. Special tokens like `<BOS>`, `<EOS>`, `<USER>`, `<ASSISTANT>` mark conversation structure.
2. **Embeddings** — Each token is mapped to a 64-dimensional vector, then positional embeddings are added.
3. **Transformer blocks** — 2 blocks, each with 4-head causal self-attention + a feed-forward network.
4. **Language model head** — A linear layer maps hidden states to logits over the vocabulary.
5. **Generation** — At inference time, tokens are sampled one at a time (top-k sampling with temperature) until `<EOS>` is reached.

**Model config:**

| Setting | Value |
|---|---|
| Embedding dim | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feed-forward dim | 256 |
| Context length | 64 tokens |
| Vocabulary size | ~653 tokens |
| Total parameters | ~188,000 |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
→ Make sure you activated your virtual environment before running:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

**Wrong Python version (older than 3.9)**
→ If you see an error like `TypeError: 'type' object is not subscriptable`, your Python is too old.
Check your version:
```bash
python --version
```
If it says anything below `3.9`, download a newer version from https://www.python.org/downloads/ and recreate the virtual environment:
```bash
# After installing Python 3.9+
python3.11 -m venv .venv        # use whichever version you just installed
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

**`UnicodeEncodeError` on Windows**
→ Set your terminal encoding before running:
```bash
set PYTHONIOENCODING=utf-8
python main.py
```

**`FileNotFoundError: checkpoints/tinygpt_v2.pt`**
→ Run `python main.py` first to train and save the model, then use `chat_v2.py`.

**Model gives weird answers**
→ The model is very small (~188K parameters) trained on limited data. It only knows what's in `data/conversations.txt`. For better answers, add more Q&A pairs to that file and retrain with `python main.py`.

