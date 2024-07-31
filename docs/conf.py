# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../synax'))
sys.path.insert(0, os.path.abspath('../examples/'))
#os.system('ln -s ../examples/Integration.ipynb nb/Integration.ipynb')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Synax'
copyright = '2024, Kangning Diao, Zack Li, Richard D.P. Grumitt'
author = 'Kangning Diao, Zack Li, Richard D.P. Grumitt'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',  # Automatically document docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.mathjax',
    'nbsphinx',
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'inherited-members': False,
    'show-inheritance': False,
    'special-members': '',
}
nbsphinx_execute = 'never'  # or 'never' if you do not want to execute notebooks during build


autodoc_mock_imports = ["jax",'healpy','interpax']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
