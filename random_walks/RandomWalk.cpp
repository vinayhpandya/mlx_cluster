#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"
#include "mlx/random.h"
#include "mlx/ops.h"
#include "mlx/array.h"
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif
#include "random_walks/RandomWalk.h"

namespace mlx::core {
    void RandomWalk::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    auto& rowptr = inputs[0];
    auto& col = inputs[1];
    auto& start = inputs[2];
    auto& rand = inputs[3];
    int numel = start.size();

    // Initialize outputs
    assert(outputs.size() == 2);
    // Allocate memory for outputs if not already allocated
    outputs[0].set_data(allocator::malloc_or_wait(numel*(walk_length_+1)*sizeof(int64_t)));
    outputs[1].set_data(allocator::malloc_or_wait(numel*walk_length_*sizeof(int64_t)));
    auto& n_out = outputs[0];
    auto& e_out = outputs[1];

    auto* n_out_ptr = n_out.data<int64_t>();
    auto* e_out_ptr = e_out.data<int64_t>();
    auto* start_values = start.data<int64_t>();
    auto* row_ptr = rowptr.data<int64_t>();
    auto* col_values = col.data<int64_t>();
    auto* rand_values = rand.data<float>();

    std::cout<<"After evaluating outputs"<<std::endl;
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
    throw std::runtime_error("Random walk has no GPU implementation.");
}
#ifdef _METAL_
void RandomWalk::eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs
    ){
       auto& rowptr = inputs[0];
        auto& col = inputs[1];
        auto& start = inputs[2];
        auto& rand = inputs[3];
        int numel = start.size();
        
        assert(outputs.size() == 2);
        outputs[0].set_data(allocator::malloc_or_wait(numel * (walk_length_ + 1) * sizeof(int64_t)));
        outputs[1].set_data(allocator::malloc_or_wait(numel * walk_length_ * sizeof(int64_t)));
        std::cout<<"after setting data"<<std::endl;
        auto& s = stream();
        auto& d = metal::device(s.device);

        d.register_library("mlx_cluster");
        auto kernel = d.get_kernel("random_walk", "mlx_cluster");

        auto& compute_encoder = d.get_command_encoder(s.index);
        compute_encoder->setComputePipelineState(kernel);

        compute_encoder.set_input_array(rowptr, 0);
        compute_encoder.set_input_array(col, 1);
        compute_encoder.set_input_array(start, 2);
        compute_encoder.set_input_array(rand, 3);
        compute_encoder.set_output_array(outputs[0], 4);
        compute_encoder.set_output_array(outputs[1], 5);
        compute_encoder->setBytes(&walk_length_, sizeof(int32), 6);

        MTL::Size grid_size = MTL::Size(numel, 1, 1);
        MTL::Size thread_group_size = MTL::Size(kernel->maxTotalThreadsPerThreadgroup(), 1, 1);

        compute_encoder.dispatchThreads(grid_size, thread_group_size);
    }
#endif

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
    throw std::runtime_error("Random walk has no GPU implementation.");
}

std::vector<std::vector<int>> RandomWalk::output_shapes(const std::vector<array>& inputs)
{
    throw std::runtime_error("Random walk has no GPU implementation.");
}

array random_walk(const array& rowptr, const array& col, const array& start, const array& rand, int walk_length, StreamOrDevice s)
{   
    std::cout<<"Inside random walk"<<std::endl;
    int nodes = start.size();
    auto primitive = std::make_shared<RandomWalk>(walk_length, to_stream(s));
    return array::make_arrays({{nodes,walk_length+1},{nodes, walk_length}},
     {start.dtype(), start.dtype()},
     primitive,
     {rowptr, col, start, rand}
    )[0];
}
}