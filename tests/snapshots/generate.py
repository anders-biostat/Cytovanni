"""Regenerate the snapshot files: ``python tests/snapshots/generate.py [case ...]`` (default: all cases)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases  # noqa: E402

if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    names = sys.argv[1:] or list(cases.CASES)
    for name in names:
        out = cases.CASES[name]()
        np.savez(cases.path(name), **out)
        print(name, {k: np.shape(v) for k, v in out.items()})
