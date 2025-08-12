#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <random_walks/RandomWalk.h>
#include <random_walks/BiasedRandomWalk.h>
#include <nanobind/stl/vector.h>
#include <random_walks/NeighborSample.h>
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

NB_MODULE(_ext, m){

    m.def(
      "random_walk",
      [](const mx::array& rowptr,
        const mx::array& col,
        const mx::array& start,
        const mx::array& rand,
        int  walk_length,
        nb::object stream = nb::none()) {

          // call the real C++ implementation
          auto outs = mlx_random_walk::random_walk(
              rowptr, col, start, rand, walk_length,
              stream.is_none() ? mx::StreamOrDevice{}
                              : nb::cast<mx::StreamOrDevice>(stream));

          // vector -> tuple (move avoids a copy)
          return nb::make_tuple(std::move(outs[0]), std::move(outs[1]));
      },
      "rowptr"_a, "col"_a, "start"_a, "rand"_a, "walk_length"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(
          Uniform random walks.

          Args:
            rowptr (mlx.core.array): rowptr of graph in csr format.
            col (mlx.core.array): edges(col) in csr format.
            start_indices (mlx.core.array): starting nodes of graph from which 
                            sampling will be performed.
            random_values (mlx.corearray): random values (between 0 to 1)
            walk_length (int) : walk length of random graph

          Returns:
              (nodes, edges) tuple of arrays
      )",
      nb::rv_policy::move);

      m.def(
        "rejection_sampling",
        [](const mx::array& rowptr,
          const mx::array& col,
          const mx::array& start,
          int walk_length,
          float p,
          float q,
          nb::object stream = nb::none()
        ){
          auto outs = mlx_biased_random_walk::rejection_sampling(
            rowptr, col, start, walk_length, p, q,
            stream.is_none() ? mx::StreamOrDevice{}
                              : nb::cast<mx::StreamOrDevice>(stream));
          return nb::make_tuple(std::move(outs[0]), std::move(outs[1]));
        },
        "rowptr"_a,
        "col"_a,
        "start"_a,
        "walk_length"_a,
        "p"_a,
        "q"_a,
        nb::kw_only(), "stream"_a = nb::none(),
      R"(
        Sample nodes from the graph by sampling neighbors based
        on probablity p and q

        Args:
            rowptr (mlx.core.array): rowptr of graph in csr format.
            col (mlx.core.array): edges in csr format.
            start (mlx.core.array): starting node of graph from which 
                            biased sampling will be performed.
            walk_length (int) : walk length of random graph
            p : Likelihood of immediately revisiting a node in the walk.
            q : Control parameter to interpolate between
                breadth-first strategy and depth-first strategy

        Returns:
            (nodes, edges) tuple of arrays
      )",
      nb::rv_policy::move);

      m.def(
        "neighbor_sample",
        [](const mx::array& colptr,
           const mx::array& row,
           const mx::array& input_node,
           const std::vector<int64_t>& num_neighbors,
           bool replace = false,
           bool directed = true) {
            
            // Call your C++ function
            auto result = neighbor_sample(colptr, row, input_node, num_neighbors, replace, directed);
            
            // Convert std::tuple to nanobind tuple with move semantics
            return nb::make_tuple(
                std::move(std::get<0>(result)),  // samples
                std::move(std::get<1>(result)),  // rows
                std::move(std::get<2>(result)),  // cols
                std::move(std::get<3>(result))   // edges
            );
        },
        "colptr"_a,
        "row"_a, 
        "input_node"_a,
        "num_neighbors"_a,
        "replace"_a = false,
        "directed"_a = true,
        R"(
            Simple neighbor sampling without primitives.
            
            Args:
                colptr: Column pointers (CSC format)
                row: Row indices (CSC format)  
                input_node: Input nodes to sample from
                num_neighbors: Number of neighbors per hop
                replace: Sample with replacement
                directed: Directed graph
                
            Returns:
                tuple: (samples, rows, cols, edges)
        )",
        nb::rv_policy::move  // Add this return value policy
    );
  }