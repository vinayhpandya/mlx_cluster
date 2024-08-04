#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <random_walks/AddTwoNumbers.h>
#include <random_walks/RandomWalk.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

NB_MODULE(_ext, m){
    m.def(
      "addTwoNumbers",
      &addTwoNumbers,
      "x"_a,
      "y"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Add two floating point arrays 
        ``z = x + y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.

        Returns:
            array: ``x + y``
      )");

      m.def(
        "random_walk",
        &random_walk,
        "rowptr"_a,
        "col"_a,
        "start"_a,
        "walk_length"_a,
        nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Perform random walk on a graph


        Args:
            rowptr (array): rowptr of graph in csr format.
            col (array): edges in csr format.
            walk_length (int) : walk length of random graph

        Returns:
            array: consisting of nodes visited on random walk
      )");
}

