/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;

template <typename T>
class SparseCountNonzeroOp : public OpKernel {
 public:
  explicit SparseCountNonzeroOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices_t = ctx->input(0);
    // make shape for output
    const Tensor& shape_t = ctx->input(1);

    auto shape_vec = shape_t.flat<int64>();
    int64 ndims = shape_vec.size();
    if (axis_ < 0) {
      axis_ = ndims + axis_;
    }
    OP_REQUIRES(ctx, (axis_ > 0 && axis_ < ndims),
                errors::InvalidArgument("axis must be among (0, ndims)"));
    std::vector<int64> dims;
    for (int d = 0; d < ndims; ++d) {
      if (d < axis_) {
        dims.push_back(shape_vec(d));
      }
    }
    TensorShape output_shape = TensorShape(dims);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
// LOG(INFO) << "==================> SparseCountNonzeroOp: " << name()
//<< ", indices_t: " << indices_t.DebugString()
//<< ", shape_t: " << shape_t.DebugString()
//<< ", axis_: " << axis_
//<< ", output: " << output->DebugString();

  
    // Hack here
    auto output_flat = output->flat<int32>();
    for (size_t i = 0; i < output_flat.size(); ++i) {
      output_flat(i) = 1;
    }
 
    // invoke the functor
    //functor::SparseCountNonzeroFunctor<T> count_functor_;
    //count_functor_(ctx, &indices_t, &shape_t, out_values, axis_, ndims);
  }

 private:
  int axis_;
};

#define REGISTER_SPARSE_COUNT_NONZERO_KERNEL(T)          \
  REGISTER_KERNEL_BUILDER(Name("SparseCountNonzero")     \
                              .Device(DEVICE_CPU)        \
                              .HostMemory("input_shape") \
                              .TypeConstraint<T>("T"),   \
                          SparseCountNonzeroOp<T>)

TF_CALL_int32(REGISTER_SPARSE_COUNT_NONZERO_KERNEL);
TF_CALL_int64(REGISTER_SPARSE_COUNT_NONZERO_KERNEL);

#undef REGISTER_SPARSE_COUNT_NONZERO_KERNEL
}  // namespace tensorflow

