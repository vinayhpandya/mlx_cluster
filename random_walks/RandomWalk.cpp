#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"
#include "mlx/random.h"
#include "mlx/ops.h"
#include "mlx/array.h"
#include "random_walks/RandomWalk.h"

namespace mlx::core {
    void RandomWalk::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    auto& rowptr = inputs[0];
    auto& col = inputs[1];
    auto& start = inputs[2];
    int numel = start.size();

    
    // Generate random numbers
    auto rand = random::uniform({numel,walk_length_});

    rand.eval();

    // Initialize outputs
    assert(outputs.size() == 2);
    // Allocate memory for outputs if not already allocated
    outputs[0].set_data(allocator::malloc_or_wait(numel*(walk_length_+1)*sizeof(int64_t)));
    outputs[1].set_data(allocator::malloc_or_wait(numel*walk_length_*sizeof(int64_t)));
    auto& n_out = outputs[0];
    auto& e_out = outputs[1];

    auto* n_out_ptr = n_out.data<int32_t>();
    auto* e_out_ptr = e_out.data<int32_t>();
    auto* start_values = start.data<int32_t>();
    auto* row_ptr = rowptr.data<int32_t>();
    auto* col_values = col.data<int32_t>();
    auto* rand_values = rand.data<float>();
    for (int64_t n = 0; n < numel; n++) {
        int64_t n_cur = start_values[n];
        n_out_ptr[n * (walk_length_ + 1)] = n_cur;
        for (int l = 0; l < walk_length_; l++) {
            int64_t row_start = row_ptr[n_cur];
            int64_t row_end = row_ptr[n_cur+1];
            int64_t e_cur;
            if (row_end - row_start == 0) {
                e_cur = -1;
            } else {
                float r = rand_values[n*walk_length_+l];
                int64_t idx = static_cast<int64_t>(r * (row_end - row_start));
                e_cur = row_start + idx;
                n_cur = col_values[e_cur];
            }

            n_out_ptr[n * (walk_length_ + 1) + (l + 1)] = n_cur;
            e_out_ptr[n * walk_length_ + l] = e_cur;
        }
    }
   
    };

    std::vector<array> RandomWalk::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums)
{
    // Random walk is not differentiable, so we return zero tangents
    auto output_shape = output_shapes(primals)[0];
    return {zeros(output_shape, primals[0].dtype(), stream())};
}

void RandomWalk::eval_gpu(
        const std::vector<array>& input,
        std::vector<array>& output
    ){
       throw std::runtime_error("Random walk has no GPU implementation.");
    }

std::vector<array> RandomWalk::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs)
{
    // Random walk is not differentiable, so we return zero gradients
    throw std::runtime_error("Random walk has no GPU implementation.");
}

std::pair<std::vector<array>, std::vector<int>> RandomWalk::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes)
{
    throw std::runtime_error("vmap not implemented for RandomWalk");
}

bool RandomWalk::is_equivalent(const Primitive& other) const
{
    if (auto* p = dynamic_cast<const RandomWalk*>(&other)) {
        return walk_length_ == p->walk_length_;
    }
    return false;
}

std::vector<std::vector<int>> RandomWalk::output_shapes(const std::vector<array>& inputs)
{
    int num_starts = inputs[2].shape()[0];
    return {
        {num_starts, walk_length_ + 1},  // Shape of node sequence output
        {num_starts, walk_length_}       // Shape of edge sequence output
    };
}

array random_walk(const array& rowptr, const array& col, const array& start, int walk_length, StreamOrDevice s)
{   
    int nodes = start.size();
    auto primitive = std::make_shared<RandomWalk>(walk_length, to_stream(s));
    return array::make_arrays({{nodes,walk_length},{nodes, walk_length+1}},
     {start.dtype(), start.dtype()},
     primitive,
     {rowptr, col, start}
    )[0];
}
}