============
Installation
============

Requirements
============

* Python 3.8 or later
* macOS with Apple Silicon (M1/M2/M3) or Intel Mac
* MLX framework

Install from PyPI
=================

The easiest way to install mlx-cluster is using pip:

.. code-block:: bash

   pip install mlx-cluster

Install from Source
===================

To install the latest development version:
You need to install X-code to enable metal code to run on gpu.

.. code-block:: bash

   git clone https://github.com/yourusername/mlx-cluster.git
   cd mlx-cluster
   pip install -e .

Verify Installation
===================

To verify that mlx-cluster is installed correctly:

.. code-block:: python

   import mlx_cluster
   print(mlx_cluster.__version__)

If you see the version number, the installation was successful!

Troubleshooting
===============

ImportError: No module named 'mlx'
-----------------------------------

Make sure you have MLX installed:

.. code-block:: bash

   pip install mlx

Performance Issues
------------------

For optimal performance, ensure you're running on Apple Silicon hardware with sufficient memory.