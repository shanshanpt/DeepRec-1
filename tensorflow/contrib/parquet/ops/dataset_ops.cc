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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("ParquetTabularDataset")
    .Output("handle: variant")
    .Input("filename: string")
    .Input("batch_size: int64")
    .Attr("field_names: list(string) >= 1")
    .Attr("field_dtypes: list(type) >= 1")
    .Attr("field_ragged_ranks: list(int) >= 1")
    .Attr("partition_count: int = 1")
    .Attr("partition_index: int = 0")
    .Attr("drop_remainder: bool = false")
    .SetIsStateful()  // NOTE: Source dataset ops must be marked stateful to
                      // inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
A dataset that outputs batches from a parquet file.

handle: The handle to reference the dataset.
filename: Path of file to read.
batch_size: Maxium number of samples in an output batch.
field_names: List of field names to read.
field_dtypes: List of data types for each field.
field_ragged_ranks: List of ragged rank for each field.
partition_count: Count of row group partitions.
partition_index: Index of row group partitions.
drop_remainder: If True, only keep batches with exactly `batch_size` samples.
)doc");

REGISTER_OP("RebatchTabularDataset")
    .Output("handle: variant")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("min_batch_size: int64")
    .Attr("field_ids: list(int) >= 1")
    .Attr("field_ragged_indices: list(int) >= 1")
    .Attr("drop_remainder: bool")
    .Attr("num_parallel_scans: int = 1")
    .SetIsStateful()  // NOTE: Source dataset ops must be marked stateful to
                      // inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // min_batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
A dataset that resizes batches from another tabular dataset.

handle: The handle to reference the dataset.
input_dataset: Input batch dataset.
batch_size: Maxium number of samples in an output batch.
min_batch_size: Minimum number of samples in an non-final batch.
field_ids: A list of tensor indices to indicate the type of a tensor is
  values (0), batch splits (1) or other splits (>1).
field_ragged_indices: A list of indices to indicate the type of a tensor is
  values (0), batch splits (1) or other splits (>1).
drop_remainder: If True, only keep batches with exactly `batch_size` samples.
num_parallel_scans: Number of concurrent scans against fields of input dataset.
)doc");

REGISTER_OP("DetectEndDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

}  // namespace tensorflow

