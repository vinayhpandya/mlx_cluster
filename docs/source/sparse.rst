Sparse Matrix Multiplication (SpMM)
===================================

mlx-cluster provides fused sparse × dense matrix multiplication kernels for
graph neural network message passing. They compute

.. math::

   \text{out}[r, f] = \mathrm{reduce}_{\,e \,:\, \text{row}[e] = r}
   \; \text{values}[e] \cdot x[\text{col}[e], f]

where the sparse matrix ``A`` is the graph (one non-zero per edge) and ``x``
holds the node features. In message-passing terms, ``row`` is each edge's
destination node, ``col`` its source node and ``values`` an optional edge
weight.

Why use it
----------

The usual *gather-scatter* message passing first gathers ``x[col]`` into an
array with one row **per edge** (``[num_edges, features]``) and then scatters
it into the destination nodes. On large graphs that per-edge array dominates
memory and time. The SpMM kernels read each neighbour's features and combine
them directly into the output, so only node-sized arrays are ever allocated,
in the forward **and** the backward pass.

Measured on an Apple M4 Pro (one training step = forward + backward, GCN
layer, 1M edges, 256 features): about **4x faster** and **4.6x less memory**
than gather-scatter (2.36 GB → 509 MB). The gain grows with
``num_edges × features``; on very small graphs (e.g. Cora) it is about
break-even.

Quick example
-------------

.. code-block:: python

   import mlx.core as mx
   from mlx_cluster import coo_to_csr, spmm_coo, spmm_csr

   # edge_index = [src, dst], as in mlx-graphs
   edge_index = mx.array([[0, 1, 2, 2], [1, 2, 0, 1]])
   x = mx.random.normal((3, 16))
   src, dst = edge_index[0], edge_index[1]

   # COO: works directly on the edge list
   out = spmm_coo(dst, src, x, reduce="sum")         # == scatter(x[src], dst, "add")

   # CSR: sort once per graph, then reuse
   csr = coo_to_csr(dst, src, num_rows=3)
   out = spmm_csr(
       csr.rowptr, csr.col, x,
       reduce="max",
       long_rows=csr.long_rows,     # load balancing for high-degree nodes
       transpose=csr.transpose,     # fast, deterministic backward pass
   )

   # gradients work as usual
   grad = mx.grad(lambda x: spmm_coo(dst, src, x, reduce="mean").sum())(x)

COO or CSR?
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - :func:`~mlx_cluster.spmm_coo`
     - :func:`~mlx_cluster.spmm_csr`
   * - Input
     - The edge list as-is, any order
     - CSR from :func:`~mlx_cluster.coo_to_csr` (one sort per graph; cache it)
   * - GPU kernel
     - One thread per (edge, feature); atomic adds
     - One thread per (node, feature); no atomics
   * - Determinism
     - Forward ``max``/``min`` are exact; ``sum``/``mean`` and all gradients
       w.r.t. ``x`` may differ in the last bits between runs
     - Bit-identical results and gradients every run (with ``transpose``)
   * - Best for
     - Small graphs, or graphs that change every step
     - Medium and large graphs (fastest in our benchmarks)

.. _reductions:

Reductions
----------

``reduce`` selects how the messages arriving at a node are combined, per
feature:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - ``reduce``
     - Forward
     - Gradient
   * - ``"sum"`` / ``"add"``
     - Sum of the messages
     - ``A^T @ grad`` for ``x``; one dot product per edge for ``values``
   * - ``"mean"``
     - Sum divided by the number of incoming edges
     - As ``sum``, scaled by ``1 / degree``
   * - ``"max"``
     - Largest message
     - Every edge whose message equals the output gets the full gradient
       (ties included)
   * - ``"min"``
     - Smallest message
     - As ``max``

Conventions, matching mlx-graphs' ``scatter``:

* Nodes without incoming edges get ``0`` for every reduction (not ``±inf``).
* ``max``/``min`` are the true maximum/minimum, so an all-negative
  neighbourhood gives a negative result.
* With ``max``/``min``, an exact tie (e.g. a duplicated edge) gives the full
  gradient to **every** tied edge, exactly like MLX's ``scatter`` max/min
  gradient. The backward pass recomputes each message and compares it with the
  saved output, so no argmax array is stored.
* All arithmetic is done in float32. Inputs of other dtypes are converted, and
  the result is returned in the dtype of ``x``.

``mx.grad`` / ``mx.vjp`` are supported for every reduction. ``mx.jvp`` is
supported for ``sum`` and ``mean``. Second-order gradients and ``mx.vmap`` are
not supported.

High-degree nodes (load balancing)
----------------------------------

Real graphs often have a few nodes with a huge number of neighbours. A
one-thread-per-node kernel would leave one thread looping over all of a hub's
edges while the rest of the GPU sits idle. With ``long_rows`` (built by
:func:`~mlx_cluster.coo_to_csr`), nodes with more than ``long_row_threshold``
edges (default 256) are cut into chunks of up to ``chunk_size`` edges
(default 2048). Each chunk is reduced by its own GPU threadgroup, then the
chunk results are combined in a fixed order. This keeps results
deterministic, and for sums it is also more accurate than one long running sum.

On a 1M-edge power-law graph whose biggest node has 134k neighbours, this took
the CSR forward pass from about 50 ms to about 1 ms.

Building ``long_rows`` reads ``rowptr`` back to the host once per graph, so it
cannot run inside ``mx.compile``. Pass ``long_row_threshold=None`` to skip it.

Fast backward pass (transpose)
------------------------------

The gradient of ``sum``/``mean`` with respect to ``x`` is ``A^T @ grad``.
:func:`~mlx_cluster.coo_to_csr` also builds the transposed CSR
(``csr.transpose``, edges grouped by source node, with its own ``long_rows``).
When it is passed to :func:`~mlx_cluster.spmm_csr`, the backward pass runs on
the CSR kernel too: no atomics, deterministic, and load-balanced for nodes
with many outgoing edges. This holds for every reduction: for ``max``/``min``
each source node loops over its outgoing edges and adds the gradient of the
edges that won (see :ref:`reductions`). Without ``transpose`` (and with
:func:`~mlx_cluster.spmm_coo`), the gradient w.r.t. ``x`` uses atomic adds
instead, so it is correct but its last bits can vary between runs. ``coo_to_csr(..., transpose=False)`` saves the extra
memory (about three index arrays) when only the forward pass is needed.

Using it from mlx-graphs
------------------------

mlx-graphs uses these kernels automatically for layers decorated with
``@fused_message_passing`` (``GCNConv``, ``SAGEConv``, ``GINConv``,
``SimpleConv``). It falls back to gather-scatter whenever the fused path does
not apply. See the mlx-graphs documentation for details.

Limitations
-----------

* Messages must have the form ``values[e] * x[col[e]]`` with a scalar weight
  per edge. Attention layers (GAT) build messages from both endpoints and
  produce per-edge outputs, so they cannot use SpMM.
* ``softmax`` aggregation produces one value per edge, so it is out of scope.
* Indices are processed as int64 and features as float32 inside the kernels.
