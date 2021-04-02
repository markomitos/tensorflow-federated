# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for building broadcast and aggregate functions with encoding.

This file contains utilities for building `tff.templates.MeasuredProcess`
objects using `Encoder` class from `tensor_encoding` project, to realize
encoding (compression) of values being communicated between `SERVER`
and `CLIENTS`.
"""

from typing import Callable

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.utils import encoding_utils
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_model_optimization.python.core.internal import tensor_encoding

# Type aliases.
_ModelConstructor = Callable[[], model_lib.Model]
_EncoderConstructor = Callable[[tf.TensorSpec],
                               tensor_encoding.core.SimpleEncoder]


# TODO(b/138081552): Move to tff.learning when ready.
def build_encoded_broadcast_process_from_model(
    model_fn: _ModelConstructor,
    encoder_fn: _EncoderConstructor) -> measured_process.MeasuredProcess:
  """Builds `MeasuredProcess` for weights of model returned by `model_fn`.

  This method creates a `SimpleEncoder` for every weight of model created by
  `model_fn`, as returned by `encoder_fn`.

  Args:
    model_fn: A Python callable with no arguments function that returns a
      `tff.learning.Model`.
    encoder_fn: A Python callable with a single argument, which is expected to
      be a `tf.Tensor` of shape and dtype to be encoded. The function must
      return a `tensor_encoding.core.SimpleEncoder`, which expects a `tf.Tensor`
      with compatible type as the input to its `encode` method.

  Returns:
    A `MeasuredProcess` for encoding and broadcasting the weights of model
    created by `model_fn`.

  Raises:
    TypeError: If `model_fn` or `encoder_fn` are not callable objects.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_callable(encoder_fn)
  weight_type = model_utils.weights_type_from_model(model_fn)
  weight_tensor_specs = type_conversions.type_to_tf_tensor_specs(weight_type)
  encoders = tf.nest.map_structure(encoder_fn, weight_tensor_specs)
  return encoding_utils.build_encoded_broadcast_process(weight_type, encoders)
