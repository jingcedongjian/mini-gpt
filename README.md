# Mini GPT

A character-level language model built from scratch with PyTorch.

## Current Stage

The project currently implements a complete Bigram language model pipeline:

- Text loading and token encoding
- Training batch generation
- Model training and validation
- Autoregressive token generation
- Token decoding into text

The Bigram model will gradually be upgraded to a Transformer-based mini GPT.

## Project Structure

```text
mini_gpt/
├── data/
│   └── input.txt
├── data.py
├── model.py
├── train.py
└── README.md
```

## Run

```bash
python train.py
```

## Roadmap

- Token and position embeddings
- Causal self-attention with Q, K and V
- Multi-head attention
- Feed-forward network
- Residual connections and LayerNorm
- Complete Transformer blocks