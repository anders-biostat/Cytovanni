"""Cytovanni: computational standardisation of flow cytometry data.

Subpackages: `cytovanni.io` (reading .fcs files into AnnData), `cytovanni.gating`, `cytovanni.rainbow` (rainbow bead
calibration), `cytovanni.spill` (single-stain spectra), `cytovanni.ref` (reference-based batch standardisation),
`cytovanni.utils` and `cytovanni.torch` (helpers), `cytovanni.exceptions`.
"""
from . import exceptions
from . import utils
from . import torch
from . import io
from . import gating
from . import rainbow
from . import spill
from . import ref

from .version import __version__
