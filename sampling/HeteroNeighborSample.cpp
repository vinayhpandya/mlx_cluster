#include "sampling/HeteroNeighborSample.h"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "sampling/Mapper.h"

namespace mlx_cluster {

namespace {

const std::string kDelim = "__";

// Raises if `a` is not a 1-D int64 array; returns a contiguous view of it.
// The caller evaluates all prepared arrays in one batch.
mx::array prepare_index_array(const mx::array& a, const std::string& what) {
  if (a.dtype() != mx::int64) {
    throw std::invalid_argument(what + " must have dtype int64 (cast it with "
                                       ".astype(mx.int64) once, up front)");
  }
  if (a.ndim() != 1) {
    throw std::invalid_argument(what + " must be 1-D, got " +
                                std::to_string(a.ndim()) + " dimensions");
  }
  return mx::contiguous(a);
}

mx::array to_mx_array(const std::vector<int64_t>& v) {
  if (v.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::length_error(
        "hetero_neighbor_sample: output has more than 2^31 - 1 entries for a "
        "single node/edge type; reduce the batch size or fan-out");
  }
  return mx::array(v.begin(), mx::Shape{static_cast<int32_t>(v.size())},
                   mx::int64);
}

template <typename Map>
const typename Map::mapped_type& at_or_throw(const Map& m,
                                             const typename Map::key_type& k,
                                             const std::string& what) {
  auto it = m.find(k);
  if (it == m.end()) {
    throw std::invalid_argument(what + " is missing key '" + k + "'");
  }
  return it->second;
}

// Per-edge-type output buffers.
struct EdgeBuffer {
  std::vector<int64_t> rows, cols, edge_ids;
  std::vector<int64_t> per_hop;
};

}  // namespace

RelType to_rel_type(const EdgeType& edge_type) {
  return std::get<0>(edge_type) + kDelim + std::get<1>(edge_type) + kDelim +
         std::get<2>(edge_type);
}

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
    uint64_t seed) {
  if (!directed) {
    throw std::invalid_argument(
        "hetero_neighbor_sample: directed=False is not supported yet for "
        "heterogeneous graphs (pyg-lib has the same restriction)");
  }

  // ---------------------------------------------------------------------
  // 1. Validate structure and prepare (contiguous, evaluated) inputs.
  // ---------------------------------------------------------------------
  std::unordered_set<NodeType> node_type_set(node_types.begin(),
                                             node_types.end());
  if (node_type_set.size() != node_types.size()) {
    throw std::invalid_argument("node_types contains duplicates");
  }

  std::vector<RelType> rel_types;
  rel_types.reserve(edge_types.size());
  std::unordered_map<RelType, mx::array> colptrs, rows;
  std::vector<mx::array> to_eval;

  for (const auto& et : edge_types) {
    const auto& src = std::get<0>(et);
    const auto& dst = std::get<2>(et);
    if (!node_type_set.count(src) || !node_type_set.count(dst)) {
      throw std::invalid_argument("edge type '" + to_rel_type(et) +
                                  "' refers to a node type not in node_types");
    }
    const RelType rt = to_rel_type(et);
    if (colptrs.count(rt)) {
      throw std::invalid_argument("edge type '" + rt + "' given twice");
    }
    rel_types.push_back(rt);
    auto cp = prepare_index_array(at_or_throw(colptr_dict, rt, "colptr_dict"),
                                  "colptr_dict['" + rt + "']");
    auto rw = prepare_index_array(at_or_throw(row_dict, rt, "row_dict"),
                                  "row_dict['" + rt + "']");
    at_or_throw(num_neighbors_dict, rt, "num_neighbors_dict");
    to_eval.push_back(cp);
    to_eval.push_back(rw);
    colptrs.emplace(rt, std::move(cp));
    rows.emplace(rt, std::move(rw));
  }

  std::unordered_map<NodeType, mx::array> seeds;
  for (const auto& kv : seed_dict) {
    if (!node_type_set.count(kv.first)) {
      throw std::invalid_argument("seed_dict key '" + kv.first +
                                  "' is not in node_types");
    }
    auto s = prepare_index_array(kv.second, "seed_dict['" + kv.first + "']");
    to_eval.push_back(s);
    seeds.emplace(kv.first, std::move(s));
  }
  if (seeds.empty()) {
    throw std::invalid_argument("seed_dict must contain at least one node type");
  }

  mx::eval(to_eval);

  // Number of nodes per type: explicit value wins, otherwise infer it from
  // any edge type whose dst is that node type (CSC colptr has N + 1 entries).
  std::unordered_map<NodeType, int64_t> num_nodes;
  for (const auto& kv : num_nodes_dict) {
    if (node_type_set.count(kv.first)) {
      num_nodes[kv.first] = kv.second;
    }
  }
  for (size_t i = 0; i < edge_types.size(); ++i) {
    const auto& rt = rel_types[i];
    const auto& dst = std::get<2>(edge_types[i]);
    const auto& cp = colptrs.at(rt);
    const auto& rw = rows.at(rt);
    if (cp.size() < 1) {
      throw std::invalid_argument("colptr_dict['" + rt + "'] is empty");
    }
    const int64_t n_dst = static_cast<int64_t>(cp.size()) - 1;
    const int64_t last = cp.data<int64_t>()[n_dst];
    if (last != static_cast<int64_t>(rw.size())) {
      throw std::invalid_argument(
          "colptr_dict['" + rt + "'][-1] = " + std::to_string(last) +
          " does not match len(row_dict['" + rt +
          "']) = " + std::to_string(rw.size()));
    }
    auto it = num_nodes.find(dst);
    if (it == num_nodes.end()) {
      num_nodes[dst] = n_dst;
    } else if (it->second != n_dst) {
      throw std::invalid_argument(
          "colptr_dict['" + rt + "'] implies " + std::to_string(n_dst) +
          " '" + dst + "' nodes, but " + std::to_string(it->second) +
          " were given/implied elsewhere");
    }
  }

  // Fan-out length (number of hops).
  size_t num_hops = 0;
  for (const auto& rt : rel_types) {
    num_hops = std::max(num_hops, num_neighbors_dict.at(rt).size());
  }

  // ---------------------------------------------------------------------
  // 2. Initialise per-node-type state with the seeds.
  // ---------------------------------------------------------------------
  std::mt19937_64 gen(seed);

  std::unordered_map<NodeType, std::vector<int64_t>> sampled;
  std::unordered_map<NodeType, Mapper> mappers;
  std::unordered_map<NodeType, std::pair<size_t, size_t>> frontier;
  std::unordered_map<NodeType, std::vector<int64_t>> nodes_per_hop;

  for (const auto& nt : node_types) {
    auto it = num_nodes.find(nt);
    const int64_t n = (it == num_nodes.end()) ? 0 : it->second;
    sampled[nt];
    mappers.emplace(nt, Mapper(n));
    frontier[nt] = {0, 0};
    nodes_per_hop[nt] = {0};
  }

  for (const auto& kv : seeds) {
    const auto& nt = kv.first;
    const auto& s = kv.second;
    const auto* s_data = s.data<int64_t>();
    auto& mapper = mappers.at(nt);
    auto& out = sampled.at(nt);
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
      if (!mapper.insert(s_data[i]).second) {
        throw std::invalid_argument("seed_dict['" + nt +
                                    "'] contains duplicate node " +
                                    std::to_string(s_data[i]));
      }
      out.push_back(s_data[i]);
    }
    frontier[nt] = {0, out.size()};
    nodes_per_hop[nt][0] = static_cast<int64_t>(out.size());
  }

  std::unordered_map<RelType, EdgeBuffer> edges;
  for (const auto& rt : rel_types) {
    edges[rt];
  }

  // ---------------------------------------------------------------------
  // 3. Multi-hop sampling.
  //
  // For every hop, the frontier of each node type is fixed up front, so a
  // node discovered during hop h is only expanded at hop h + 1 (regardless
  // of edge-type iteration order). For edge type (src, rel, dst) we sample
  // incoming neighbours (src nodes) of every frontier dst node.
  // ---------------------------------------------------------------------
  std::unordered_set<int64_t> picked;  // reused for sampling w/o replacement

  for (size_t hop = 0; hop < num_hops; ++hop) {
    for (size_t e = 0; e < edge_types.size(); ++e) {
      const auto& rt = rel_types[e];
      const auto& src = std::get<0>(edge_types[e]);
      const auto& dst = std::get<2>(edge_types[e]);
      const auto& fanout = num_neighbors_dict.at(rt);
      const int64_t count = hop < fanout.size() ? fanout[hop] : 0;

      auto& buf = edges.at(rt);
      const size_t edges_before = buf.rows.size();
      buf.per_hop.push_back(0);
      if (count == 0) {
        continue;
      }

      const auto* colptr = colptrs.at(rt).data<int64_t>();
      const auto* row = rows.at(rt).data<int64_t>();
      const int64_t n_dst = static_cast<int64_t>(colptrs.at(rt).size()) - 1;

      // NB: when src == dst, `src_nodes` and `dst_nodes` alias and grow while
      // we iterate, so always index by position and copy values out.
      auto& dst_nodes = sampled.at(dst);
      auto& src_nodes = sampled.at(src);
      auto& src_mapper = mappers.at(src);
      const auto [begin, end] = frontier.at(dst);

      auto add_edge = [&](int64_t offset, size_t local_dst) {
        const int64_t v = row[offset];
        const auto res = src_mapper.insert(v);
        if (res.second) {
          src_nodes.push_back(v);
        }
        buf.rows.push_back(res.first);
        buf.cols.push_back(static_cast<int64_t>(local_dst));
        buf.edge_ids.push_back(offset);
      };

      for (size_t i = begin; i < end; ++i) {
        const int64_t w = dst_nodes[i];
        if (w < 0 || w >= n_dst) {
          throw std::out_of_range(
              "node " + std::to_string(w) + " of type '" + dst +
              "' is out of range for colptr_dict['" + rt + "'] (" +
              std::to_string(n_dst) + " nodes)");
        }
        const int64_t col_start = colptr[w];
        const int64_t col_end = colptr[w + 1];
        const int64_t deg = col_end - col_start;
        if (deg <= 0) {
          continue;
        }

        if (count < 0 || (!replace && count >= deg)) {
          // Take every neighbour.
          for (int64_t off = col_start; off < col_end; ++off) {
            add_edge(off, i);
          }
        } else if (replace) {
          std::uniform_int_distribution<int64_t> dist(0, deg - 1);
          for (int64_t j = 0; j < count; ++j) {
            add_edge(col_start + dist(gen), i);
          }
        } else {
          // Robert Floyd's algorithm: `count` distinct picks from [0, deg).
          picked.clear();
          for (int64_t j = deg - count; j < deg; ++j) {
            std::uniform_int_distribution<int64_t> dist(0, j);
            int64_t r = dist(gen);
            if (!picked.insert(r).second) {
              r = j;
              picked.insert(j);
            }
            add_edge(col_start + r, i);
          }
        }
      }
      buf.per_hop.back() =
          static_cast<int64_t>(buf.rows.size() - edges_before);
    }

    // Advance every frontier to "nodes discovered during this hop".
    for (const auto& nt : node_types) {
      const size_t prev_end = frontier.at(nt).second;
      const size_t now = sampled.at(nt).size();
      frontier[nt] = {prev_end, now};
      nodes_per_hop.at(nt).push_back(static_cast<int64_t>(now - prev_end));
    }
  }

  // ---------------------------------------------------------------------
  // 4. Pack outputs.
  // ---------------------------------------------------------------------
  HeteroSampleOutput out;
  for (const auto& nt : node_types) {
    out.node_id.emplace(nt, to_mx_array(sampled.at(nt)));
    out.num_sampled_nodes.emplace(nt, std::move(nodes_per_hop.at(nt)));
  }
  for (const auto& rt : rel_types) {
    auto& buf = edges.at(rt);
    out.row.emplace(rt, to_mx_array(buf.rows));
    out.col.emplace(rt, to_mx_array(buf.cols));
    out.edge_id.emplace(rt, to_mx_array(buf.edge_ids));
    out.num_sampled_edges.emplace(rt, std::move(buf.per_hop));
  }
  return out;
}

}  // namespace mlx_cluster