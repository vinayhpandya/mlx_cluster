#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <random_walks/RandomWalk.h>
#include <random_walks/BiasedRandomWalk.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

NB_MODULE(_ext, m){

      m.def(
        "random_walk",
        &random_walk,
        "rowptr"_a,
        "col"_a,
        "start"_a,
        "rand"_a,
        "walk_length"_a,
        nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        uniformly sample a graph


        Args:
            rowptr (array): rowptr of graph in csr format.
            col (array): edges in csr format.
            walk_length (int) : walk length of random graph

        Returns:
            array: consisting of nodes visited on random walk
      )");

      m.def(
        "rejection_sampling",
        &rejection_sampling,
        "rowptr"_a,
        "col"_a,
        "start"_a,
        "walk_length"_a,
        "p"_a,
        "q"_a,
        nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Sample nodes from the graph by sampling neighbors based
        on probablity p and q


        Args:
            rowptr (array): rowptr of graph in csr format.
            col (array): edges in csr format.
            start (array): starting node of graph from which 
                            biased sampling will be performed.
            walk_length (int) : walk length of random graph
            p : Likelihood of immediately revisiting a node in the walk.
            q : Control parameter to interpolate between
                breadth-first strategy and depth-first strategy

        Returns:
            array: consisting of nodes visited on random walk
      )");
}

