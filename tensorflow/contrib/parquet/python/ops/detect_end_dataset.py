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

r'''DetectEndDataset that reports the existence of next element.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.parquet.python.ops import gen_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import variable_scope

from tensorflow.python.data.ops.dataset_ops import DatasetV2 as _dataset


class _DetectEndDataset(dataset_ops.DatasetSource):
  r'''Wrapping a dataset to notify whether it still has next input.
  '''
  def __init__(self, input_dataset):
    r'''Create a `_DetectEndDataset`.

    Args:
      input_dataset: A `dataset` to be wrapped to verify its last element.
    '''
    self._input_dataset = input_dataset
    self._marker_spec = tensor_spec.TensorSpec(shape=[], dtype=dtypes.bool)
    variant_tensor = gen_dataset_ops.detect_end_dataset(
        self._input_dataset._variant_tensor) #pylint: disable=protected-access
    super(_DetectEndDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    return self._marker_spec, self._input_dataset.element_spec  # pylint: disable=protected-access

class DetectEndDataset(dataset_ops.Dataset):
  r'''Wrapping a dataset to verify whether it still has next input.
  '''
  def __init__(self, input_dataset):
    r'''Create a `DetectEndDataset`.

    Args:
      input_dataset: A `dataset` to be wrapped to verify its last element.
    '''
    self._input_dataset = input_dataset
    with ops.name_scope(None):
      self._detect_end_marker = variable_scope.get_variable(
          'detect_end_marker',
          shape=[],
          dtype=dtypes.bool,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          use_resource=True)
    self._impl = _DetectEndDataset(self._input_dataset).map(self._mark_end)
    super(DetectEndDataset, self).__init__(self._impl._variant_tensor)  # pylint: disable=protected-access

  def _mark_end(self, marker, args):
    r'''Mark end of dataset.
    '''
    self._detect_end_marker.assign(marker)
    return args

  def _inputs(self):
    return [self._input_dataset]

  @property
  def element_spec(self):
    return self._input_dataset.element_spec  # pylint: disable=protected-access

def join():
  r'''Allow user to synchronize the ending of training data on
    distributed workers when meeting uneven input samples.
    It prevents collective communication operations from hanging due to an
    exit of workers.
  '''
  def _apply_fn(dataset):
    return DetectEndDataset(dataset)
  return _apply_fn

