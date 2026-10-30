#pragma once

// Heterogeneous multi-hop neighbor sampling (GraphSAGE style).
//
// Port of pyg-lib's `hetero_neighbor_sample` (CSC mode) to MLX arrays.
// Edge types are passed across the C++ boundary as "src__rel__dst" strings
// (same convention as PyG); the Python wrapper in mlx_cluster/hetero.py
// converts to and from (src, rel, dst) tuples.

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace mlx_cluster {

using NodeType = std::string;
using RelType = std::string;  // "src__rel__dst"
using EdgeType = std::tuple<std::string, std::string, std::string>;

RelType to_rel_type(const EdgeType& edge_type);

struct HeteroSampleOutput {
  // Global ids of sampled nodes per node type. For every seeded node type the
  // first len(seed) entries are the seeds, in the order given.
  std::unordered_map<NodeType, mx::array> node_id;
  // Per edge type (src, rel, dst): local src index, local dst index and the
  // offset of the sampled edge in that edge type's CSC `row` array.
  std::unordered_map<RelType, mx::array> row;
  std::unordered_map<RelType, mx::array> col;
  std::unordered_map<RelType, mx::array> edge_id;
  // num_sampled_nodes[t][0] = #seeds of type t, [h] = #new nodes at hop h.
  std::unordered_map<NodeType, std::vector<int64_t>> num_sampled_nodes;
  // num_sampled_edges[r][h] = #edges of type r sampled at hop h.
  std::unordered_map<RelType, std::vector<int64_t>> num_sampled_edges;
};

// All arrays must be 1-D int64. For an edge type (src, rel, dst):
//   colptr_dict[rel_type] has length num_dst_nodes + 1,
//   row_dict[rel_type]    holds the src node of each edge, sorted by dst.
// num_nodes_dict is optional (may be empty); it lets every node type use the
// fast dense mapper.
HeteroSampleOutput hetero_neighbor_sample(
    const std::vector<NodeType>& node_types,
    const std::vector<EdgeType>& edge_types,
    const std::unordered_map<RelType, mx::array>& colptr_dict,
    const std::unordered_map<RelType, mx::array>& row_dict,
    const std::unordered_map<NodeType, mx::array>& seed_dict,
    const std::unordered_map<RelType, std::vector<int64_t>>& num_neighbors_dict,
    const std::unordered_map<NodeType, int64_t>& num_nodes_dict,
    bool replace,
    bool directed,
    uint64_t seed);

}  // namespace mlx_cluster