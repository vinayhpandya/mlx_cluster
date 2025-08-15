# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../mlx_cluster"))

# Try importing to verify path is correct
try:
    import mlx_cluster
    print(f"mlx_cluster imported successfully from: {mlx_cluster.__file__}")
except ImportError as e:
    print("Failed to import mlx_cluster:", e)
    print("Python path:", sys.path)

project = 'mlx-cluster'
copyright = '2025, Vinay Pandya'
author = 'Vinay Pandya'
release = '0.0.6'

# -- General configuration ---------------------------------------------------
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

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

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
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "mlx-cluster documentation"

html_theme_options = {
    "repository_url": "https://github.com/vinayhpandya/mlx_cluster",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_navbar_depth": 3,  # Show 3 levels in navbar
    "logo": {
        "text": "MLX-Cluster",
    },
    "home_page_in_toc": False,
    "github_url": "https://github.com/vinayhpandya/mlx_cluster",
    
    # Better sidebar navigation - valid options only
    "collapse_navigation": False,
}