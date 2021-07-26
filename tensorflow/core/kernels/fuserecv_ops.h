/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_FUSERECV_OPS_H_
#define TENSORFLOW_KERNELS_FUSERECV_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class FuseRecvOp : public AsyncOpKernel {
 public:
  explicit FuseRecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::vector<string> key_prefixs_;
  std::vector<Rendezvous::ParsedKey> parsed_keys_;
  bool hostmem_sendrecv_;
  int fuse_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(FuseRecvOp);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FUSERECV_OPS_H_