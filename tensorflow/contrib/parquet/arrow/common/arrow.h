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

#ifndef TENSORFLOW_CONTRIB_PARQUET_ARROW_COMMON_ARROW_
#define TENSORFLOW_CONTRIB_PARQUET_ARROW_COMMON_ARROW_

#include <deque>
#include <string>

#include <arrow/dataset/api.h>
#include <arrow/record_batch.h>
#include <parquet/arrow/reader.h>
#include <parquet/properties.h>

#include <arrow/filesystem/hdfs.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/filesystem/localfs.h>

namespace tensorflow {
namespace arrow {

int UpdateArrowCpuThreadPoolCapacityFromEnv();

int GetArrowFileBufferSizeFromEnv();

::arrow::Status OpenArrowFile(
    std::shared_ptr<::arrow::io::RandomAccessFile>* file,
    const std::string& filename);

::arrow::Status OpenParquetReader(
    std::unique_ptr<::parquet::arrow::FileReader>* reader,
    const std::shared_ptr<::arrow::io::RandomAccessFile>& file);

::arrow::Status GetParquetDataFrameFields(
    std::vector<std::string>* field_names,
    std::vector<std::string>* field_dtypes,
    std::vector<int>* field_ragged_ranks, const std::string& filename);

} // namespace arrow
} // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_PARQUET_ARROW_COMMON_ARROW_

