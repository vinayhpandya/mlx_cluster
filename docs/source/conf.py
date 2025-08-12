# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

try:
    import mlx_cluster
    print("mlx_cluster imported successfully!")
except ImportError as e:
    print("Failed to import mlx_cluster:", e)

project = 'mlx-cluster'
copyright = '2025, Vinay Pandya'
author = 'Vinay Pandya'
release = '0.0.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx.ext.viewcode",
]
napoleon_use_param = True
napoleon_google_docstring = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True
typehints_use_signature = True
typehints_use_signature_return = True
autosummary_generate = True


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'mlx': ('https://ml-explore.github.io/mlx/build/html/', None)
}
templates_path = ['_templates']
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

language = 'Python'
master_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "mlx-cluster documentation"

html_theme_options = {
    "repository_url": "https://github.com/vinayhpandya/mlx_cluster",  # Replace with your actual GitHub URL
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_navbar_depth": 2,
    "logo": {
        "text": "MLX-Cluster",
    },
    "home_page_in_toc": True,
    "github_url": "https://github.com/vinayhpandya/mlx_cluster",  # Alternative way to add GitHub link
}
