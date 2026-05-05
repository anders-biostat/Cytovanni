import re

project = "Cytovanni"
author = "Valentin Wüst, Simon Anders"

with open("../src/cytovanni/version.py") as f:
    release = re.search(r'__version__ = ["\']([^"\']+)["\']', f.read()).group(1)

extensions = ["myst_nb"]

nb_execution_mode = "off"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/anders-biostat/Cytovanni",
}
