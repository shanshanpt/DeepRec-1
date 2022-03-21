# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

r'''Input pipelines.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.parquet.python.ops.dataframe import DataFrame
from tensorflow.contrib.parquet.python.ops.dataframe import to_sparse
from tensorflow.contrib.parquet.python.ops.dataframe import unbatch_and_to_sparse
from tensorflow.contrib.parquet.python.ops.detect_end_dataset import DetectEndDataset
from tensorflow.contrib.parquet.python.ops.detect_end_dataset import join
from tensorflow.contrib.parquet.python.ops.parquet_dataset import ParquetDataset
from tensorflow.contrib.parquet.python.ops.parquet_dataset import read_parquet
from tensorflow.contrib.parquet.python.ops.rebatch_dataset import RebatchDataset
from tensorflow.contrib.parquet.python.ops.rebatch_dataset import rebatch

