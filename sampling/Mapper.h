#pragma once

// Global -> local node-id mapper used by the samplers.
//
// Ported from pyg-lib (pyg_lib/csrc/sampler/cpu/mapper.h). Every node type
// gets its own Mapper. Local ids are handed out consecutively in insertion
// order, so `local id == position in the sampled-node list` for that type.
//
// For small node types (or when we expect to touch a large fraction of the
// nodes) a dense vector lookup is much faster than a hash map; for very large
// node types we fall back to a hash map to avoid allocating O(num_nodes)
// memory on every call.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "parallel_hashmap/phmap.h"

namespace mlx_cluster {

class Mapper {
 public:
  // num_nodes <= 0 means "unknown", which forces the hash-map path.
  explicit Mapper(int64_t num_nodes, int64_t expected_entries = 0)
      : num_nodes_(num_nodes) {
    use_vec_ = (num_nodes > 0) &&
               ((num_nodes < 1000000) || (expected_entries > num_nodes / 10));
    if (use_vec_) {
      to_local_vec_.assign(static_cast<size_t>(num_nodes), -1);
    }
  }

  // Returns (local_id, inserted). `inserted` is false if `node` was already
  // present, in which case `local_id` is the id assigned the first time.
  std::pair<int64_t, bool> insert(int64_t node) {
    if (use_vec_) {
      check_bounds(node);
      int64_t& slot = to_local_vec_[static_cast<size_t>(node)];
      if (slot >= 0) {
        return {slot, false};
      }
      slot = curr_;
      return {curr_++, true};
    }
    if (node < 0) {
      throw std::out_of_range("node id " + std::to_string(node) +
                              " is negative");
    }
    auto res = to_local_map_.try_emplace(node, curr_);
    if (res.second) {
      ++curr_;
    }
    return {res.first->second, res.second};
  }

  // Returns the local id of `node`, or -1 if it has not been inserted.
  int64_t map(int64_t node) const {
    if (use_vec_) {
      if (node < 0 || node >= num_nodes_) {
        return -1;
      }
      return to_local_vec_[static_cast<size_t>(node)];
    }
    auto it = to_local_map_.find(node);
    return it == to_local_map_.end() ? -1 : it->second;
  }

  bool exists(int64_t node) const { return map(node) >= 0; }

  int64_t size() const { return curr_; }

 private:
  void check_bounds(int64_t node) const {
    if (node < 0 || node >= num_nodes_) {
      throw std::out_of_range("node id " + std::to_string(node) +
                              " is out of range for a node type with " +
                              std::to_string(num_nodes_) + " nodes");
    }
  }

  int64_t num_nodes_;
  int64_t curr_ = 0;
  bool use_vec_ = false;
  std::vector<int64_t> to_local_vec_;
  phmap::flat_hash_map<int64_t, int64_t> to_local_map_;
};

}  // namespace mlx_cluster