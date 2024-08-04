#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"
#include "mlx/random.h"
#include "random_walks/AddTwoNumbers.h"

namespace mlx::core {

    array addTwoNumbers(
    const array& x, // Input array x
    const array& y, // Input array y
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
    ) {
  // Promote dtypes between x and y as needed
    // auto promoted_dtype = promote_types(x.dtype(), y.dtype());

    // // Upcast to float32 for non-floating point inputs x and y
    // auto out_dtype = issubdtype(promoted_dtype, float32)
    //     ? promoted_dtype
    //     : promote_types(promoted_dtype, float32);

    // // Cast x and y up to the determined dtype (on the same stream s)
    // auto x_casted = astype(x, out_dtype, s);
    // auto y_casted = astype(y, out_dtype, s);

    // // Broadcast the shapes of x and y (on the same stream s)
    // auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
    // auto out_shape = broadcasted_inputs[0].shape();

    // Construct the array as the output of the Axpby primitive
    // with the broadcasted and upcasted arrays as inputs
    return array(
        /* const std::vector<int>& shape = */ x.shape(),
        /* Dtype dtype = */ x.dtype(),
        /* std::unique_ptr<Primitive> primitive = */
        std::make_shared<AddTwoNumbers>(to_stream(s)),
        /* const std::vector<array>& inputs = */ {x,y});
    }


    template <typename T>
    void add_impl(
        const array& x,
        const array& y,
        array& out) {
        // We only allocate memory when we are ready to fill the output
        // malloc_or_wait synchronously allocates available memory
        // There may be a wait executed here if the allocation is requested
        // under memory-pressured conditions
        out.set_data(allocator::malloc_or_wait(out.nbytes()));

        auto rand  = mlx::core::random::uniform({2,3});
        rand.eval();
        // Collect input and output data pointers
        const T* x_ptr = x.data<T>();
        const T* y_ptr = y.data<T>();
        T* out_ptr = out.data<T>();

        float* rand_values = rand.data<float>();

        
        // Do the element-wise operation for each output
        for (size_t out_idx = 0; out_idx < out.size(); out_idx++) {
            // Map linear indices to offsets in x and y
            auto x_offset = elem_to_loc(out_idx, x.shape(), x.strides());
            auto y_offset = elem_to_loc(out_idx, y.shape(), y.strides());
            
            std::cout<<rand_values[1]<<std::endl;
            // We allocate the output to be contiguous and regularly strided
            // (defaults to row major) and hence it doesn't need additional mapping
            out_ptr[out_idx] = x_ptr[x_offset] + y_ptr[y_offset];
        }
    }

    void AddTwoNumbers::eval(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) {
        // Check the inputs (registered in the op while constructing the out array)
        assert(inputs.size() == 2);
        auto& x = inputs[0];
        auto& y = inputs[1];
        auto& out = outputs[0];

        // Dispatch to the correct dtype
        if (out.dtype() == float32) {
            return add_impl<float>(x, y, out);
        } else if (out.dtype() == float16) {
            return add_impl<float16_t>(x, y, out);
        } else if (out.dtype() == bfloat16) {
            return add_impl<bfloat16_t>(x, y, out);
        } else if (out.dtype() == complex64) {
            return add_impl<complex64_t>(x, y, out);
        } else {
            throw std::runtime_error(
                "AddTwoNumbers is only supported for floating point types.");
                }
    }

    void AddTwoNumbers::eval_cpu(
        const std::vector<array>& input,
        std::vector<array>& output
    ){
        eval(input, output);
    }

    void AddTwoNumbers::eval_gpu(
        const std::vector<array>& input,
        std::vector<array>& output
    ){
       throw std::runtime_error("AddTwoNumbers has no GPU implementation.");
    }

    std::vector<array> AddTwoNumbers::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
        throw std::runtime_error("AddTwoNumbers has no jvp implementation.");
    }


    std::pair<std::vector<array>, std::vector<int>> AddTwoNumbers::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
        throw std::runtime_error("AddTwoNumbers has no vmap implementation.");
    }

    std::vector<array> AddTwoNumbers::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
        throw std::runtime_error("AddTwoNumbers has no vjp implementation.");
    }

    bool AddTwoNumbers::is_equivalent(const Primitive& other) const {
        throw std::runtime_error("AddTwoNumbers has no is_equivalent implementation.");
    }

};