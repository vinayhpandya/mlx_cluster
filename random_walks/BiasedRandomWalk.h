#pragma once

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace mx = mlx::core;
namespace mlx_biased_random_walk{

    class BiasedRandomWalk : public mx::Primitive {
        public:
            BiasedRandomWalk(mx::Stream stream, int walk_length, double p, double q)
            : mx::Primitive(stream), walk_length_(walk_length), p_(p), q_(q) {}
            void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
            override;
            void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
            override;

            /** The Jacobian-vector product. */
            std::vector<mx::array> jvp(
                const std::vector<mx::array>& primals,
                const std::vector<mx::array>& tangents,
                const std::vector<int>& argnums) override;
          
            /** The vector-Jacobian product. */
            std::vector<mx::array> vjp(
                const std::vector<mx::array>& primals,
                const std::vector<mx::array>& cotangents,
                const std::vector<int>& argnums,
                const std::vector<mx::array>& outputs) override;
          
            /**
             * The primitive must know how to vectorize itself across
             * the given axes. The output is a pair containing the array
             * representing the vectorized computation and the axis which
             * corresponds to the output vectorized dimension.
             */
            std::pair<std::vector<mx::array>, std::vector<int>> vmap(
                const std::vector<mx::array>& inputs,
                const std::vector<int>& axes) override;

            /** Print the primitive. */
            virtual const char* name() const override {
                return "biased random walk implementation";
            }

            /** Equivalence check **/
            bool is_equivalent(const mx::Primitive& other) const override;
        
        private:
            int walk_length_;
            double p_;
            double q_;

    };

    std::vector<mx::array> rejection_sampling(const mx::array& rowptr,
     const mx::array& col,
    const mx::array& start,
       int walk_length,
       const double p, 
       const double q,
        mx::StreamOrDevice s = {});

};