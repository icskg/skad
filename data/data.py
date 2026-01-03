import torch
import torch.utils.data as tud
import numpy as np
from typing import Optional, List

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

from comm.utils import remove_constant_columns


class WindowedGraphDataset(tud.Dataset):
    """统一窗口数据集，支持 train/val/test"""
    def __init__(
        self,
        data: np.ndarray,           # [T, M]
        node_types: np.ndarray,     # [M]
        adj_prior: np.ndarray,      # [M, M]
        window: int,
        stride: int = 1,
        labels: Optional[np.ndarray] = None  # [T]，测试/验证时提供
    ):
        super().__init__()
        self.data = data.astype(np.float32)
        self.node_types = node_types.astype(np.int64)
        self.adj_prior = adj_prior.astype(np.float32)
        self.window = int(window)
        self.stride = int(stride)
        self.labels = labels.astype(np.int64) if labels is not None else None

        self.n_total, self.n_vars = self.data.shape
        assert self.n_vars == self.node_types.shape[0], "the number of columns does not match the node types."

        # 预计算窗口起始位置
        self.starts: List[int] = []
        t = 0
        while t + self.window <= self.n_total:
            self.starts.append(t)
            t += self.stride
        if not self.starts:
            raise ValueError("the window/stride being too large results in no samples.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = s + self.window
        x = torch.from_numpy(self.data[s:e, :])
        y_types = torch.from_numpy(self.node_types)
        adj = torch.from_numpy(self.adj_prior)
        if self.labels is not None:
            y_labels = torch.from_numpy(self.labels[s:e])
        else:
            y_labels = torch.zeros(e-s, dtype=torch.int)
        return x, y_types, adj, y_labels, s


class KMADDataModule:
    def __init__(
        self,
        dataset: str,
        window: int,
        stride: int = 1,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        drop_constant_cols: bool = True,
        scaling: str = "minmax",
        limit_train_ratio: float = 1.0
    ):
        from comm.io import load_train_test_csv, load_graph_json

        self.window = int(window)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.scaling = scaling.lower()

        # Load CSV & graph
        script_path = os.path.dirname(os.path.abspath(__file__))
        train_csv = os.path.join(script_path, '..', 'raw', dataset, 'train.csv')
        test_csv = os.path.join(script_path, '..', 'raw', dataset, 'test.csv')
        graph_json = os.path.join(script_path, '..', 'raw', dataset, 'graph.json')

        tr_data, te_data, te_labels, tr_cols, te_cols, label_col = load_train_test_csv(train_csv, test_csv, drop_constant_cols)
        node_types, adj_prior, num_types = load_graph_json(graph_json)

        limit_train_ratio = min(limit_train_ratio, 1.0)
        if limit_train_ratio < 1.0:
            n_limit = int(tr_data.shape[0] * limit_train_ratio)
            start_idx = np.random.randint(0, tr_data.shape[0] - n_limit + 1)
            tr_data = tr_data[start_idx:start_idx + n_limit, :]
        # Sanity check
        print(f"[INFO] train data dim: {tr_data.shape[1]}, test data dim: {te_data.shape[1]}, node_types: {node_types.shape[0]}")
        assert tr_data.shape[1] == te_data.shape[1] == node_types.shape[0], "the number of items does not match."

        # 归一化
        if self.scaling == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(tr_data)
        tr_n = scaler.transform(tr_data)
        te_n = scaler.transform(te_data)

        # Split train/val
        t_tr = tr_n.shape[0]
        n_train = int(t_tr * (1.0 - val_ratio))
        if n_train < window or (t_tr - n_train) < window:
            n_train = t_tr
            val_n = None
        else:
            val_n = tr_n[n_train:, :]
        train_n = tr_n[:n_train, :]

        # Build datasets（统一类）
        self.train_ds = WindowedGraphDataset(train_n, node_types, adj_prior, window, stride)
        self.val_ds = WindowedGraphDataset(val_n, node_types, adj_prior, window, stride) if val_n is not None else None
        self.test_ds = WindowedGraphDataset(te_n, node_types, adj_prior, window, stride, labels=te_labels)

        self.num_types = int(num_types)
        self.num_vars = tr_data.shape[1]
        self.train_cols = tr_cols
        self.test_cols = te_cols
        self.label_col = label_col
        assert self.train_cols == self.test_cols, "the column names of train and test do not match."

    def make_loader(self, ds, shuffle=True):
        if ds is None:
            return None
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

    def train_val_test_loaders(self):
        return self.make_loader(self.train_ds, True), self.make_loader(self.val_ds, False), self.make_loader(self.test_ds, False)
