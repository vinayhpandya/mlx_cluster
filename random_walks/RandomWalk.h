#pragma once

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace mlx::core{

    class RandomWalk : public Primitive {
        public:
            explicit RandomWalk(int walk_length, Stream stream):
                Primitive(stream), walk_length_(walk_length) {};
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
                os << "Random walk implementation";
            }

            /** Equivalence check **/
            bool is_equivalent(const Primitive& other) const override;

            std::vector<std::vector<int>> output_shapes(const std::vector<array>& inputs) override;
        
        private:
            int walk_length_;

    };

    array random_walk(const array& rowptr,
     const array& col,
    const array& start,
    const array& rand,
       int walk_length,
        StreamOrDevice s = {});

};