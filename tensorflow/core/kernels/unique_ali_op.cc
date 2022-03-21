/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <functional>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/task_runner.h"
#include "tensorflow/core/kernels/unique_ali_op_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {
const char* kUniqueOpSerialEnv = "DEEPREC_UNIQUE_OP_SERIAL";
const char* kUniqueOpHashMapEnv = "DEEPREC_UNIQUE_OP_HASH_MAP";
const char* kUniqueOpUniqRatioHint = "DEEPREC_UNIQUE_OP_UNIQ_RATIO_HINT";
const char* kUniqueOpPartitionSizeEnv = "DEEPREC_UNIQUE_OP_PARTITION_SIZE";
const char* kMultiMapString = "MULTIMAP";
const char* kStlHashMapString = "STL";
const char* kAbslHashMapString = "ABSL";
const char* kGoogleHashMapString = "GOOGLE";
const int64 kDefaultUniqueRatioHint = 4;
}

template <typename T, typename TIndex>
class UniqueAliOp : public OpKernel {
 public:
  explicit UniqueAliOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, ReadInt64FromEnvVar(kUniqueOpPartitionSizeEnv,
                                             kPartitionSize, &partition_size_));
    OP_REQUIRES(context, partition_size_ > 0,
                errors::InvalidArgument("Invaild PARTITION_SIZE=",
                                        partition_size_));

    OP_REQUIRES_OK(context, ReadBoolFromEnvVar(kUniqueOpSerialEnv,
                                                false, &serial_));

    // NOTE(zycao>: Hash map insertion and lookup performance is dominating in
    // Unique Op. Based on benchmark results, 'google::dense_hash_map' will be
    // used as default for most key types except string.
    //
    // By setting "DEEPREC_UNIQUE_OP_HASH_MAP" environment variable, a particular
    // hash map could be seleteed to use. Possible choices are listed below:
    //     "MULTIMAP" for multimap parrallel process,
    //     "STL" for std::unordred_map,
    //     "ABSL" for absl::flat_hash_map,
    //     "GOOGLE" for google::dense_hash_map.
    std::string hash_map_str;
    OP_REQUIRES_OK(context, ReadStringFromEnvVar(kUniqueOpHashMapEnv,
                                                 kGoogleHashMapString,
                                                 &hash_map_str));
    std::transform(hash_map_str.begin(), hash_map_str.end(),
                   hash_map_str.begin(), ::toupper);

    OP_REQUIRES_OK(context, ReadInt64FromEnvVar(kUniqueOpUniqRatioHint,
        kDefaultUniqueRatioHint, &unique_ratio_hint_));
    OP_REQUIRES(context, unique_ratio_hint_ > 0,
                errors::InvalidArgument("Invaild ", kUniqueOpUniqRatioHint, "=",
                                        unique_ratio_hint_));

    if (!hash_map_str.compare(kMultiMapString)) {
      map_flag_ = MULTIMAP;
      static char print_once = [] {
        LOG(INFO) << "MultiMapCompute preserved "
          "dense hash map key: " << kPreseverdEmptyKey;
        return '\0';
      }();
    } else if (!hash_map_str.compare(kStlHashMapString)) {
      map_flag_ = STL;
    } else if (!hash_map_str.compare(kAbslHashMapString)) {
      map_flag_ = ABSL;
    } else if (!hash_map_str.compare(kGoogleHashMapString)) {
      map_flag_ = GOOGLE;
    } else {
      map_flag_ = GOOGLE;
    }

  }

  void Compute(OpKernelContext* context) override {
    VLOG(2) << "Unique V2 executed";
    ComputeInternal(context);
  }

 private:
  void ComputeInternal(OpKernelContext* context) {
    const Tensor& input = context->input(0);
    Tensor idx;
    Tensor output;
    Tensor output_counter;
    if (context->num_inputs() == 1) {
      UniqueWithoutAxis<T, TIndex>(context, input,
          &idx, &output, &output_counter, num_outputs(),
          partition_size_, serial_, unique_ratio_hint_,
          map_flag_);
	  //LOG(ERROR) << "===============> Unique: " << name() << ", input: " << input.DebugString();
    } else {
      const Tensor& axis_tensor = context->input(1);
      UniqueWithAxis<T, TIndex>(context, input,
          axis_tensor, &idx, &output, &output_counter,
          num_outputs(), partition_size_, serial_,
          unique_ratio_hint_, map_flag_);
	  //LOG(ERROR) << "===============> Unique: " << name() << ", axis_tensor: " << axis_tensor.DebugString();
    }
    context->set_output(0, output);
    context->set_output(1, idx);
    if (num_outputs() > 2) {
      context->set_output(2, output_counter);
//	  LOG(ERROR) << "===============> Unique: " << name() 
//		  << ", output[0]: " << output.DebugString() << ", output[1]: " << idx.DebugString()
//		  << ", output[2]: " << output_counter.DebugString();
    } else {
  //    LOG(ERROR) << "===============> Unique: " << name()
	//            << ", output[0]: " << output.DebugString() << ", output[1]: " << idx.DebugString();	  
	}
  }

  bool serial_ = false;
  int64 partition_size_ = 0;
  int64 unique_ratio_hint_;
  UniqueMaps map_flag_ = GOOGLE;  // "GOOGLE" dense hash map is default
};

#define REGISTER_UNIQUE(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueAliOp<type, int32>);             \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueAliOp<type, int64>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueV2")                       \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueAliOp<type, int32>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueV2")                       \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueAliOp<type, int64>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueAliOp<type, int32>)              \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueAliOp<type, int64>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV2")             \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueAliOp<type, int32>)              \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCountsV2")             \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueAliOp<type, int64>)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_UNIQUE);
REGISTER_UNIQUE(string)
#undef REGISTER_UNIQUE

// Fake integer GPU kernels so that the use of Unique in optimizers (to
// de-duplicate sparse gradient indices) does not conflict with gradients being
// located on a GPU. These kernels run on the CPU, their inputs and outputs
// residing in host (not GPU) memory.
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int32, int32>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int32, int64>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int32>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int64, int32>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int64>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int64, int64>);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int32, int32>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int32>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int64, int32>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int32, int64>);
REGISTER_KERNEL_BUILDER(Name("Unique")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int64>("T")
                            .TypeConstraint<int64>("out_idx")
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("idx"),
                        UniqueAliOp<int64, int64>);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
