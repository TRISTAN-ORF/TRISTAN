import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import metadata


# -- Project information

project = "TRISTAN"
copyright = "2025, Jim Clauwaert"
author = "Jim Clauwaert"

# The full version, including alpha/beta/rc tags.
try:
    release = metadata.version("transcript-transformer")
except metadata.PackageNotFoundError:
    release = "0.0.0"

# The short X.Y version.
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    # "sphinx_copybutton",
    "myst_parser",
]

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
