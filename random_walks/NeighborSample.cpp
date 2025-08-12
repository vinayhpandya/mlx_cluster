#include <mlx/mlx.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <random>
using namespace std;
namespace mx = mlx::core;

tuple<mx::array, mx::array, mx::array, mx::array>
neighbor_sample(const mx::array &colptr, const mx::array &row,
       const mx::array &input_node, const vector<int64_t> num_neighbors,
       bool replace = true, bool directed = true) {
  // Initialize some data structures for the sampling process:
  vector<int64_t> samples;
  unordered_map<int64_t, int64_t> to_local_node;
  
  // Get raw data pointers from MLX arrays
  auto colptr_data = colptr.data<int64_t>();
  auto row_data = row.data<int64_t>();
  auto input_node_data = input_node.data<int64_t>();
  
  // Initialize with input nodes
  for (int64_t i = 0; i < input_node.size(); i++) {
    const auto &v = input_node_data[i];
    samples.push_back(v);
    to_local_node.insert({v, i});
  }
  
  vector<int64_t> rows_out, cols_out, edges_out;
  int64_t begin = 0, end = samples.size();
  
  // Multi-hop sampling
  for (int64_t ell = 0; ell < (int64_t)num_neighbors.size(); ell++) {
    const auto &num_samples = num_neighbors[ell];
    for (int64_t i = begin; i < end; i++) {
      const auto &w = samples[i];
      const auto &col_start = colptr_data[w];
      const auto &col_end = colptr_data[w + 1];
      const auto col_count = col_end - col_start;
      
      if (col_count == 0)
        continue;
      
      if ((num_samples < 0) || (!replace && (num_samples >= col_count))) {
        // Sample all neighbors
        for (int64_t offset = col_start; offset < col_end; offset++) {
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols_out.push_back(i);
            rows_out.push_back(res.first->second);
            edges_out.push_back(offset);
          }
        }
      } else if (replace) {
        // Sample with replacement
        for (int64_t j = 0; j < num_samples; j++) {
            

            // At the top of the function or as a static var
          const int64_t offset = col_start + (std::rand() % col_count);
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols_out.push_back(i);
            rows_out.push_back(res.first->second);
            edges_out.push_back(offset);
          }
        }
      } else {
        // Sample without replacement using reservoir sampling
        std::unordered_set<int64_t> rnd_indices;
        for (int64_t j = col_count - num_samples; j < col_count; j++) {
          int64_t rnd = rand() % (j + 1);
          if (!rnd_indices.insert(rnd).second) {
            rnd = j;
            rnd_indices.insert(j);
          }
          const int64_t offset = col_start + rnd;
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols_out.push_back(i);
            rows_out.push_back(res.first->second);
            edges_out.push_back(offset);
          }
        }
      }
    }
    begin = end;
    end = samples.size();
  }
  
  // Handle undirected case
  if (!directed) {
    unordered_map<int64_t, int64_t>::iterator iter;
    for (int64_t i = 0; i < (int64_t)samples.size(); i++) {
      const auto &w = samples[i];
      const auto &col_start = colptr_data[w];
      const auto &col_end = colptr_data[w + 1];
      for (int64_t offset = col_start; offset < col_end; offset++) {
        const auto &v = row_data[offset];
        iter = to_local_node.find(v);
        if (iter != to_local_node.end()) {
          rows_out.push_back(iter->second);
          cols_out.push_back(i);
          edges_out.push_back(offset);
        }
      }
    }
  }
  
  // Convert vectors to MLX arrays
  auto samples_array = mx::array(samples.data(), {static_cast<int>(samples.size())}, mx::int64);
  auto rows_array = mx::array(rows_out.data(), {static_cast<int>(rows_out.size())}, mx::int64);
  auto cols_array = mx::array(cols_out.data(), {static_cast<int>(cols_out.size())}, mx::int64);
  auto edges_array = mx::array(edges_out.data(), {static_cast<int>(edges_out.size())}, mx::int64);
  
  return make_tuple(samples_array, rows_array, cols_array, edges_array);
}