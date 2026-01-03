# SKAD (Knowledge-Constrained Multitask Anomaly Detection)

## Project Structure
```
│  .gitignore
│  LICENSE
│  main.py
│  README.md
│  requirements.txt 
├─comm
│  │  early_stopping.py
│  │  io.py
│  │  utils.py
│  │  __init__.py
│          
├─data
│  │  data.py
│  │  __init__.py
│          
├─eval
│  │  evaluation.py
│  │  __init__.py
│          
├─kmad
│  │  model.py
│  │  __init__.py
│  │  
│  ├─modules
│  │  │  attention.py
│  │  │  decoders.py
│  │  │  film.py
│  │  │  gat.py
│  │  │  spatiotemporal.py
│  │  │  temporal.py
│  │  │  trifactorizer.py
│  │  │  __init__.py
│          
└─train
    │  train.py
    │  __init__.py
```

## Quickstart
```bash
# 1) Create venv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run a tiny sanity-check training loop on random raw
python train.py
```

## Data Shapes
- Input window `X`: `[B, T, M]` (normalized).
- Node types `node_types`: `[B, M]` (int indices, also used as classification labels in the example).
- Optional prior adjacency `A_prior`: `[B, M, M]` or `[M, M]`.

## Notes
- Local spatiotemporal adjacency per segment uses cosine similarity of segment-level temporal features.
- Global adjacency is obtained via Neural-Guided tri-factorization of segment adjacencies.
- The simple Dense-GAT here avoids external graph libraries.
- Losses include: Huber (reconstruction & adjacency), Focal (classification), segment smoothness, and segment adjacency reconstruction.