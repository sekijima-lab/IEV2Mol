"""Maintained CPU runtime: restricted data/state loading and local-only tracking."""
import json
import platform
from pathlib import Path
import numpy as np
import torch
from iev_math.mode import old_compatible

class LocalTracker:
    def __init__(self):
        self.config = {}
        self.history = []
    def init(self, **kwargs):
        self.config = kwargs
        self.history = []
        return self
    def log(self, values, **kwargs):
        self.history.append(dict(values))
    def finish(self):
        pass
tracker = LocalTracker()

def metadata():
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'device': 'cpu', 'threads': torch.get_num_threads(),
            'old_compatible': old_compatible(), 'tracking': 'local only', 'algorithm': 'original CVAE architecture'}

def load_tensors(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def load_pairs(smiles_path, vectors_path):
    smiles = json.loads(Path(smiles_path).read_text())
    with np.load(vectors_path, allow_pickle=False) as data:
        vectors = data['vectors'].copy()
    if not isinstance(smiles, list) or not smiles or not all(isinstance(s, str) for s in smiles):
        raise ValueError('Expected a nonempty SMILES string list')
    if vectors.ndim != 2 or vectors.shape[0] != len(smiles) or vectors.dtype.kind not in 'fi' or not np.isfinite(vectors).all():
        raise ValueError('Invalid numeric interaction vectors')
    return list(zip(smiles, torch.from_numpy(vectors).float()))
