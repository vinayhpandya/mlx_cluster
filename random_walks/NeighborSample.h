#include <mlx/mlx.h>
#include <vector>
#include <tuple>

namespace mx = mlx::core;

std::tuple<mx::array, mx::array, mx::array, mx::array>
neighbor_sample(const mx::array &colptr, const mx::array &row,
               const mx::array &input_node, const std::vector<int64_t> num_neighbors,
               bool replace = true, bool directed = true);
