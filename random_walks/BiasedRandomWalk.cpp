#include <cassert>
#include <iostream>
#include <sstream>
#include <random>

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
#include "random_walks/BiasedRandomWalk.h"


namespace mlx_biased_random_walk {

    bool inline is_neighbor(const int64_t *rowptr, const int64_t *col, int64_t v,
                        int64_t w) {
        int64_t row_start = rowptr[v], row_end = rowptr[v + 1];
        for (auto i = row_start; i < row_end; i++) {
            if (col[i] == w)
            return true;
            }
        return false;
    }

    void BiasedRandomWalk::eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
        auto& rowptr = inputs[0];
        auto& col = inputs[1];
        auto& start = inputs[2];
        auto& rand = inputs[3];
        int numel = start.size();
        std::cout<<"Inside biased random walk"<<std::endl;
        // Initialize outputs
        assert(outputs.size() == 2);
        // Allocate memory for outputs if not already allocated
        outputs[0].set_data(mx::allocator::malloc(numel*(walk_length_+1)*sizeof(int64_t)));
        outputs[1].set_data(mx::allocator::malloc(numel*walk_length_*sizeof(int64_t)));
        auto& n_out = outputs[0];
        auto& e_out = outputs[1];
        auto* n_out_ptr = n_out.data<int64_t>();
        auto* e_out_ptr = e_out.data<int64_t>();
        auto* start_values = start.data<int64_t>();
        auto* row_ptr = rowptr.data<int64_t>();
        auto* col_values = col.data<int64_t>();
        auto* rand_values = rand.data<float>();

        double max_prob = fmax(fmax(1. / p_, 1.), 1. / q_);
        double prob_0 = 1. / p_ / max_prob;
        double prob_1 = 1. / max_prob;
        double prob_2 = 1. / q_ / max_prob;
        
        for (int64_t n = 0; n < numel; n++) {
            int64_t t = start_values[n], v, x, e_cur, row_start, row_end;
            n_out_ptr[n * (walk_length_ + 1)] = t;
            row_start = row_ptr[t], row_end = row_ptr[t + 1];
            if (row_end - row_start == 0) {
                e_cur = -1;
                v = t;
            } else {
                e_cur = row_start + (std::rand() % (row_end - row_start));
                v = col_values[e_cur];
            }
            n_out_ptr[n * (walk_length_ + 1) + 1] = v;
            e_out_ptr[n * walk_length_] = e_cur;
            for (auto l = 1; l < walk_length_; l++) {
                row_start = row_ptr[v], row_end = row_ptr[v + 1];

                if (row_end - row_start == 0) {
                e_cur = -1;
                x = v;
                } else if (row_end - row_start == 1) {
                e_cur = row_start;
                x = col_values[e_cur];
                } else {
                while (true) {
                    e_cur = row_start + (std::rand() % (row_end - row_start));
                    x = col_values[e_cur];

                    auto r = ((double)std::rand() / (RAND_MAX)); // [0, 1)

                    if (x == t && r < prob_0)
                    break;
                    else if (is_neighbor(row_ptr, col_values, x, t) && r < prob_1)
                    break;
                    else if (r < prob_2)
                    break;
                    }
                }
                n_out_ptr[n * (walk_length_ + 1) + (l + 1)] = x;
                e_out_ptr[n * walk_length_ + l] = e_cur;
                t = v;
                v = x;
            }
        }
   
    };

    std::vector<mx::array> BiasedRandomWalk::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums)
{
    // Random walk is not differentiable, so we return zero tangents
    throw std::runtime_error("Biased random walk has no jvp implementation.");
}
// #ifdef _METAL_
// void BiasedRandomWalk::eval_gpu(
//         const std::vector<array>& inputs,
//         std::vector<array>& outputs
//     ){
//        auto& rowptr = inputs[0];
//         auto& col = inputs[1];
//         auto& start = inputs[2];
//         auto& rand = inputs[3];
//         int numel = start.size();
        
//         assert(outputs.size() == 2);
//         outputs[0].set_data(allocator::malloc(numel * (walk_length_ + 1) * sizeof(int64_t)));
//         outputs[1].set_data(allocator::malloc(numel * walk_length_ * sizeof(int64_t)));
//         std::cout<<"after setting data"<<std::endl;
//         auto& s = stream();
//         auto& d = metal::device(s.device);

//         d.register_library("mlx_cluster", metal::get_colocated_mtllib_path);
//         auto kernel = d.get_kernel("random_walk", "mlx_cluster");

//         auto& compute_encoder = d.get_command_encoder(s.index);
//         compute_encoder->setComputePipelineState(kernel);

//         compute_encoder.set_input_array(rowptr, 0);
//         compute_encoder.set_input_array(col, 1);
//         compute_encoder.set_input_array(start, 2);
//         compute_encoder.set_input_array(rand, 3);
//         compute_encoder.set_output_array(outputs[0], 4);
//         compute_encoder.set_output_array(outputs[1], 5);
//         compute_encoder->setBytes(&walk_length_, sizeof(int32), 6);

//         MTL::Size grid_size = MTL::Size(numel, 1, 1);
//         MTL::Size thread_group_size = MTL::Size(kernel->maxTotalThreadsPerThreadgroup(), 1, 1);

//         compute_encoder.dispatchThreads(grid_size, thread_group_size);
//     }
// #endif
void BiasedRandomWalk::eval_gpu(
       const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs
  )
{
    throw std::runtime_error("Random walk has no GPU implementation.");
}
std::vector<mx::array> BiasedRandomWalk::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs)
{
    // Random walk is not differentiable, so we return zero gradients
    throw std::runtime_error("Random walk has no JVP implementation.");
}

std::pair<std::vector<mx::array>, std::vector<int>> BiasedRandomWalk::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes)
{
    throw std::runtime_error("vmap not implemented for biasedRandomWalk");
}

bool BiasedRandomWalk::is_equivalent(const mx::Primitive& other) const
{
    throw std::runtime_error("biased Random walk has no GPU implementation.");
}

std::vector<mx::array> rejection_sampling(const mx::array& rowptr, const mx::array& col, const mx::array& start, int walk_length, const double p, 
       const double q, mx::StreamOrDevice s)
{   
    int nodes = start.size();
    auto primitive = std::make_shared<BiasedRandomWalk>(to_stream(s), walk_length, p, q);
    return mx::array::make_arrays({{nodes,walk_length+1},{nodes, walk_length}},
     {rowptr.dtype(), rowptr.dtype()},
     primitive,
     {rowptr, col, start}
    );
}
}