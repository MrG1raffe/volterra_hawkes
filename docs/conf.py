import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # include project root

extensions = [
    "sphinx.ext.autodoc",       # auto-generate docs from docstrings
    "sphinx.ext.napoleon",      # Google / NumPy style docstrings
    "sphinx_autodoc_typehints"  # include type hints in docs
]

autodoc_typehints = "description"  # shows type hints in function signature or description


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Volterra-Hawkes'
copyright = '2025, Dimitri Sotnikov, Elie Attal, Eduardo Abi Jaber'
author = 'Dimitri Sotnikov, Elie Attal, Eduardo Abi Jaber'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
