#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

    class AddTwoNumbers : public Primitive {
        public:
            explicit AddTwoNumbers(Stream stream) : Primitive(stream) {};
            void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
            override;
            void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
            override;

            /** The Jacobian-vector product. */
            std::vector<array> jvp(
                const std::vector<array>& primals,
                const std::vector<array>& tangents,
                const std::vector<int>& argnums) override;

            /** The vector-Jacobian product. */
            std::vector<array> vjp(
                const std::vector<array>& primals,
                const std::vector<array>& cotangents,
                const std::vector<int>& argnums,
                const std::vector<array>& outputs) override;

            /**
             * The primitive must know how to vectorize itself across
             * the given axes. The output is a pair containing the array
             * representing the vectorized computation and the axis which
             * corresponds to the output vectorized dimension.
             */
            std::pair<std::vector<array>, std::vector<int>> vmap(
                const std::vector<array>& inputs,
                const std::vector<int>& axes) override;

            /** Print the primitive. */
            void print(std::ostream& os) override {
                os << "Axpby";
            }

            /** Equivalence check **/
            bool is_equivalent(const Primitive& other) const override;

            void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
            
    };

    array addTwoNumbers(
    const array& x, // Input array x
    const array& y, // Input array y
    StreamOrDevice s = {} // Stream on which to schedule the operation
    );
};