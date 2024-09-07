#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
using namespace metal;

kernel void random_walk(
    const device int64_t* rowptr [[buffer(0)]],
    const device int64_t* col [[buffer(1)]],
    const device int64_t* start [[buffer(2)]],
    const device float* rand [[buffer(3)]],
    device int64_t* n_out [[buffer(4)]],
    device int64_t* e_out [[buffer(5)]],
    constant int& walk_length [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    int64_t n_cur = start[tid];
    n_out[tid * (walk_length + 1)] = n_cur;
    
    for (int l = 0; l < walk_length; l++) {
        int64_t row_start = rowptr[n_cur];
        int64_t row_end = rowptr[n_cur + 1];
        int64_t e_cur;
        
        if (row_end - row_start == 0) {
            e_cur = -1;
        } else {
            float r = rand[tid * walk_length + l];
            int64_t idx = static_cast<int64_t>(r * (row_end - row_start));
            e_cur = row_start + idx;
            n_cur = col[e_cur];
        }
        
        n_out[tid * (walk_length + 1) + (l + 1)] = n_cur;
        e_out[tid * walk_length + l] = e_cur;
    }
}