from __future__ import annotations

import os.path
from typing import Tuple
import json
import numpy as np
import pandas as pd

from comm.utils import remove_constant_columns


def load_graph_json(graph_json_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load graph structure from JSON.
    Expected schema:
        {
            "node_onehot": [[0,1,0,...], ...],   # shape [num_vars, emb_dim]
            "edge_index":  [[src, dst], ...],    # list of pairs
            "edge_weights":[w1, w2, ...]         # list of floats in [-1, 1]
        }
    Returns:
        node_types: [num_vars] int array from argmax of one-hot
        adj_prior: [num_vars, num_vars] float32 in [0,1], diagonal zero (mapped from [-1,1] via (w+1)/2)
        num_types: emb_dim
    """
    with open(graph_json_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    onehot = np.asarray(g.get("node_onehot", []), dtype=np.float32)
    if onehot.ndim != 2:
        raise ValueError("node_onehot must be a 2D list/array of shape [num_vars, emb_dim].")
    num_vars, emb_dim = onehot.shape
    node_types = np.argmax(onehot, axis=1).astype(np.int64)  # [num_vars]

    edges = np.asarray(g.get("edge_index", []), dtype=np.int64)
    weights = np.asarray(g.get("edge_weights", []), dtype=np.float32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edge_index must be a list of [src, dst] pairs.")
    if weights.shape[0] != edges.shape[0]:
        raise ValueError("edge_weights length must match edge_index length.")
    adj = np.zeros((num_vars, num_vars), dtype=np.float32)
    # map weights from [-1,1] -> [0,1] to match the model's current adjacency range
    # weights = (weights + 1.0) / 2.0
    print(f"Total nodes:{num_vars}, total edges:{len(weights)}")

    for (src, dst), w in zip(edges, weights):
        if 0 <= src < num_vars and 0 <= dst < num_vars:
            adj[src, dst] = float(w)
    # zero diagonal
    np.fill_diagonal(adj, 0.0)
    return node_types, adj, int(emb_dim)

def load_train_test_csv(train_csv: str, test_csv: str, drop_constant_cols: bool = True):
    """Load train/test multivariate time series from CSV files.
    - train.csv: each column is a variable; no labels.
    - test.csv:  last column is the label; preceding columns are variables.
    Returns:
        train_data: [T_tr, M] float32
        test_data:  [T_te, M] float32
        test_labels: [T_te] int64 (from the last column)
        train_cols: list of column names in train (variables)
        test_cols:  list of variable column names in test (excluding the label col)
        label_col:  name of the label column in test.csv
    """

    df_tr = pd.read_csv(train_csv)
    # dataset = os.path.basename(os.path.dirname(train_csv))
    # skip_rows = 100000 if dataset in ['swat'] else 0
    # df_tr = df_tr.drop(range(1, skip_rows)).reset_index(drop=True)
    df_tr.columns = df_tr.columns.str.strip()
    df_tr = df_tr.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    train_cols = list(df_tr.columns.str.strip())

    df_te = pd.read_csv(test_csv)
    df_te.columns = df_te.columns.str.strip()
    if df_te.shape[1] < 2:
        raise ValueError("test.csv must contain at least one variable column and one label column.")
    # last column is the label
    label_col = df_te.columns[-1]
    test_cols = list(df_te.columns[:-1])
    df_te_vars = df_te[test_cols]
    df_te_vars = df_te_vars.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    if drop_constant_cols:
        df_tr, train_cols, const_cols = remove_constant_columns(df_tr)
        df_te_vars, test_cols, _ = remove_constant_columns(df_te_vars, drop_columns=const_cols)

    train_data = df_tr.to_numpy(dtype=np.float32)
    test_data = df_te_vars.to_numpy(dtype=np.float32)

    test_labels = df_te.iloc[:, -1].to_numpy()
    # try int labels, fallback to float -> int
    if not np.issubdtype(test_labels.dtype, np.integer):
        test_labels = test_labels.astype(np.float32)
        # if labels are floats like 0.0/1.0; cast to int
        test_labels = test_labels.astype(np.int64)
    else:
        test_labels = test_labels.astype(np.int64)

    return train_data, test_data, test_labels, train_cols, test_cols, label_col
