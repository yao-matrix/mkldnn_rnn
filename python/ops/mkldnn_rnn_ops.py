# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Mkldnn RNN operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools

from tensorflow.contrib.mkldnn_rnn.ops import gen_mkldnn_rnn_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.training import saver

_mkldnn_rnn_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_mkldnn_rnn_ops.so"))


_mkldnn_rnn_common_doc_string = """
  Mkldnn RNN has an opaque parameter buffer that can be used for inference and
  training. But it is possible that the layout of the parameter buffers
  changes between generations. So it is highly recommended to use
  RNNParamsSaveable to save and restore weights and biases in a canonical
  format.

  This is a typical use case:
    * The user creates a MkldnnRNN model.
    * The user query that parameter buffer size.
    * The user creates a variable of that size that serves as the parameter
        buffers.
    * The user either initialize the parameter buffer, or load the canonical
        weights into the parameter buffer.
    * The user calls the model with the parameter buffer for inference, or
        training.
    * If training, the user creates a Saver object.
    * If training, the user creates a RNNParamsSaveable object from the
        parameter buffer for it to be later saved in the canonical format. When
        creating a RNNParamsSaveable object, a name could be provided, which is
        useful in distinguishing the names of multiple RNNParamsSaveable
        objects (e.g. for an encoder-decoder model).
    * Once a while, the user saves the parameter buffer into model checkpoints
        with Saver.save().
    * When restoring, the user creates a RNNParamsSaveable object and uses
      Saver.restore() to restore the paramter buffer from the canonical format
      to a user-defined format, as well as to restore other savable objects
      in the checkpoint file.
"""


class _MkldnnRNN(object):
  """Creates an RNN model using the underlying Mkldnn implementation.

  Note that self._NUM_PARAMS_PER_LAYER is the number of parameter sets of
  weight and bias per layer. It needs to be defined in subclasses.
  """
  __doc__ += _mkldnn_rnn_common_doc_string

  def __init__(self,
               rnn_mode,
               num_layers,
               num_units,
               input_size,
               input_mode="linear_input",
               direction="unidirectional",
               dropout=0.,
               seed=0):
    """Creates a MkldnnRNN model from model spec.

    Args:
      rnn_mode: a string specifies the mode, under which this RNN model runs.
          Could be either 'lstm', 'gru', 'rnn_tanh' or 'rnn_relu'.
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      input_size: the size of the input, it could be different from the
          num_units.
      input_mode: indicate whether there is a linear projection between the
          input and the actual computation before the first layer. It could be
          'linear_input', 'skip_input' or 'auto_select'.
          'linear_input' (default) always applies a linear projection of input
          onto RNN hidden state. (standard RNN behavior).
          'skip_input' is only allowed when input_size == num_units;
          'auto_select' implies 'skip_input' when input_size == num_units;
          otherwise, it implies 'linear_input'.
      direction: the direction model that the model operates. Could be either
          'unidirectional' or 'bidirectional'
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the op seed used for initializing dropout. See @{tf.set_random_seed}
          for behavior.
    """
    self._num_layers = num_layers
    self._num_units = num_units
    self._input_size = input_size
    self._rnn_mode = rnn_mode
    self._input_mode = input_mode
    self._direction = direction
    self._dropout = dropout
    # get graph and op seed.
    self._seed, self._seed2 = random_seed.get_seed(seed)
    if self._seed is None and self._seed2 is None:
      self._seed, self._seed2 = 0, 0

  def params_size(self):
    """Calculates the size of the opaque parameter buffer needed for this model.

    Returns:
      The calculated parameter buffer size.
    """
    return gen_mkldnn_rnn_ops.mkldnn_rnn_params_size(
        num_layers=self._num_layers,
        num_units=self._num_units,
        input_size=self._input_size,
        T=dtypes.float32,
        S=dtypes.int32,
        dropout=self._dropout,
        seed=self._seed,
        seed2=self._seed2,
        rnn_mode=self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction)[0]

  def __call__(self, input_data, input_h, input_c, params, is_training=True):
    """Runs the forward step for the RNN model.

    Args:
      input_data: the input sequence to the RNN model.
      input_h: the initial hidden state for h.
      input_c: the initial hidden state for c. This is only relevant for LSTM.
      params: the parameter buffer created for this model.
      is_training: whether this operation will be used in training or inference.

    Returns:
      output: the output sequuence.
      output_h: the final state for h.
      output_c: the final state for c. This is only relevant for LSTM.
    """
    if self._rnn_mode != "lstm":
      # For model that doesn't take input_c, replace with a dummy tensor.
      input_c = array_ops.constant([], dtype=dtypes.float32)
    output, output_h, output_c, _ = gen_mkldnn_rnn_ops.mkldnn_rnn(
        input=input_data,
        input_h=input_h,
        input_c=input_c,
        params=params,
        rnn_mode=self._rnn_mode,
        input_mode=self._input_mode,
        direction=self._direction,
        dropout=self._dropout,
        seed=self._seed,
        seed2=self._seed2,
        is_training=is_training)
    return (output, output_h, output_c)

class MkldnnLSTM(_MkldnnRNN):
  """Mkldnn implementation of the LSTM model."""
  __doc__ += _mkldnn_rnn_common_doc_string
  # 4 sets of weight and bias parameters for the recurrent input, and 4 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = 8

  def __init__(self,
               num_layers,
               num_units,
               input_size,
               input_mode="auto_select",
               direction="unidirectional",
               dropout=0.,
               seed=0):
    """Creates a Mkldnn LSTM model from model spec.

    Args:
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      input_size: the size of the input, it could be different from the
          num_units.
      input_mode: indicate whether there is a linear projection between the
          input and The actual computation before the first layer. It could be
          'skip_input', 'linear_input' or 'auto_select'.
          'skip_input' is only allowed when input_size == num_units;
          'auto_select' implies 'skip_input' when input_size == num_units;
          otherwise, it implies 'linear_input'.
      direction: the direction model that the model operates. Could be either
          'unidirectional' or 'bidirectional'
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the seed used for initializing dropout.
    """
    super(MkldnnLSTM, self).__init__(
        "lstm",
        num_layers,
        num_units,
        input_size,
        input_mode=input_mode,
        direction=direction,
        dropout=dropout,
        seed=seed)

  def __call__(self, input_data, input_h, input_c, params, is_training=True):
    """Runs the forward step for the Mkldnn LSTM model.

    Args:
      input_data: the input sequence to the LSTM model.
      input_h: the initial hidden state for h.
      input_c: the initial hidden state for c.
      params: the parameter buffer created for this model.
      is_training: whether this operation will be used in training or inference.

    Returns:
      output: the output sequuence.
      output_h: the final state for h.
      output_c: the final state for c.
    """
    output, output_h, output_c = super(MkldnnLSTM, self).__call__(
        input_data, input_h, input_c, params, is_training=is_training)
    return (output, output_h, output_c)


class _MkldnnRNNNoInputC(_MKldnnRNN):
  """Simple MkldnnRNN models without input_c."""
  __doc__ += _mkldnn_rnn_common_doc_string

  def __init__(self,
               num_layers,
               num_units,
               input_size,
               input_mode="auto_select",
               direction="unidirectional",
               dropout=0.,
               seed=0):
    """Creates a Mkldnn RNN model from model without hidden-state C.

    Args:
      num_layers: the number of layers for the RNN model.
      num_units: the number of units within the RNN model.
      input_size: the size of the input, it could be different from the
          num_units.
      input_mode: indicate whether there is a linear projection between the
          input and The actual computation before the first layer. It could be
          'skip_input', 'linear_input' or 'auto_select'.
          'skip_input' is only allowed when input_size == num_units;
          'auto_select' implies 'skip_input' when input_size == num_units;
          otherwise, it implies 'linear_input'.
      direction: the direction model that the model operates. Could be either
          'unidirectional' or 'bidirectional'
      dropout: whether to enable dropout. With it is 0, dropout is disabled.
      seed: the seed used for initializing dropout.
    """
    super(_MkldnnRNNNoInputC, self).__init__(
        self._rnn_mode,
        num_layers,
        num_units,
        input_size,
        input_mode=input_mode,
        direction=direction,
        dropout=dropout,
        seed=seed)

  def __call__(self, input_data, input_h, params, is_training=True):
    """Runs the forward step for the MKldnn LSTM model.

    Args:
      input_data: the input sequence to the LSTM model.
      input_h: the initial hidden state for h.
      params: the parameter buffer created for this model.
      is_training: whether this operation will be used in training or inference.

    Returns:
      output: the output sequuence.
      output_h: the final state for h.
    """
    output, output_h, _ = super(_MkldnnRNNNoInputC, self).__call__(
        input_data, input_h, None, params, is_training=is_training)
    return (output, output_h)


class MkldnnGRU(_MkldnnRNNNoInputC):
  """Mkldnn implementation of the GRU model."""
  __doc__ += _mkldnn_rnn_common_doc_string
  _rnn_mode = "gru"
  # 3 sets of weight and bias parameters for the recurrent input, and 3 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = 6


class MkldnnRNNTanh(_MkldnnRNNNoInputC):
  """Mkldnn implementation of the RNN-tanh model."""
  __doc__ += _mkldnn_rnn_common_doc_string
  _rnn_mode = "rnn_tanh"
  # 1 set of weight and bias parameters for the recurrent input, and 1 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = 2


class MkldnnRNNRelu(_MkldnnRNNNoInputC):
  """Mkldnn implementation of the RNN-relu model."""
  __doc__ += _mkldnn_rnn_common_doc_string
  _rnn_mode = "rnn_relu"
  # 1 set of weight and bias parameters for the recurrent input, and 1 for the
  # previous layer input.
  _NUM_PARAMS_PER_LAYER = 2


@ops.RegisterGradient("MKldnnRNN")
def _mkldnn_rnn_backward(op, *grad):
  if not op.get_attr("is_training"):
    raise ValueError(
        "MkldnnRNN must set is_training to True to be used in gradients")
  return gen_mkldnn_rnn_ops.mkldnn_rnn_backprop(
      input=op.inputs[0],
      input_h=op.inputs[1],
      input_c=op.inputs[2],
      params=op.inputs[3],
      output=op.outputs[0],
      output_h=op.outputs[1],
      output_c=op.outputs[2],
      output_backprop=grad[0],
      output_h_backprop=grad[1],
      output_c_backprop=grad[2],
      reserve_space=op.outputs[3],
      dropout=op.get_attr("dropout"),
      seed=op.get_attr("seed"),
      seed2=op.get_attr("seed2"),
      rnn_mode=op.get_attr("rnn_mode"),
      input_mode=op.get_attr("input_mode"),
      direction=op.get_attr("direction"))


ops.RegisterShape("MkldnnRNNParamsSize")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("MkldnnRNN")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("MkldnnRNNBackprop")(common_shapes.call_cpp_shape_fn)
