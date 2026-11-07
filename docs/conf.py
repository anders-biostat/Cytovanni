import os
import re
import sys

# document the package from the source tree (no install needed); heavy dependencies are mocked below
sys.path.insert(0, os.path.abspath("../src"))

project = "Cytovanni"
author = "Valentin Wüst, Simon Anders"

with open("../src/cytovanni/version.py") as f:
    release = re.search(r'__version__ = ["\']([^"\']+)["\']', f.read()).group(1)

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

# API reference: numpydoc docstrings rendered by napoleon, one generated page per object
autosummary_generate = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_mock_imports = ["geomloss", "pykeops", "flowkit"]  # torch is installed (CPU wheel, see requirements.txt): mocking it breaks scipy
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
exclude_patterns = ["_build"]

nb_execution_mode = "off"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/anders-biostat/Cytovanni",
}
