API Reference
=============

.. autofunction:: mlx_cluster.random_walk

.. autofunction:: mlx_cluster.rejection_sampling

.. autofunction:: mlx_cluster.neighbor_sample

Heterogeneous graphs
--------------------

.. autofunction:: mlx_cluster.hetero_neighbor_sample

.. autoclass:: mlx_cluster.HeteroSamplerOutput
   :no-members:
   :exclude-members: __init__, __new__

Sparse matrix multiplication
----------------------------

See :doc:`sparse` for an overview, conventions and performance notes.

.. autofunction:: mlx_cluster.spmm_coo

.. autofunction:: mlx_cluster.spmm_csr

.. autofunction:: mlx_cluster.coo_to_csr

.. autofunction:: mlx_cluster.sddmm

.. autofunction:: mlx_cluster.build_long_rows

.. autoclass:: mlx_cluster.CSR
   :no-members:
   :exclude-members: __init__, __new__

.. autoclass:: mlx_cluster.CSRTranspose
    :no-members:
    :exclude-members: __init__, __new__

.. autoclass:: mlx_cluster.LongRows
    :no-members:
    :exclude-members: __init__, __new__
