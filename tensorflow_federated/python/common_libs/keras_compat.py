import keras
import tf_keras
import tensorflow as tf
from typing import Union
from keras.src.backend.common import KerasVariable

Model = Union[tf_keras.Model, keras.Model]

def is_compiled(model: Model):
    if isinstance(model, tf_keras.Model):
        return model._is_compiled
    else:
        return model.compiled

def keras_dtype_to_tf(dtype_str):
    if not isinstance(dtype_str, str):
        return dtype_str
    return {
        'float32': tf.float32,
        'float64': tf.float64,
        'int32': tf.int32,
        'int64': tf.int64
    }.get(dtype_str, tf.float32)


#TODO Add additional checks
def is_keras3(obj: object):
    return isinstance(obj, (keras.Model, keras.Variable, keras.Metric, keras.Layer, keras.Loss, keras.Optimizer, KerasVariable))


def get_optimizer_variables(optimizer: Union[tf_keras.optimizers.Optimizer, keras.optimizers.Optimizer]):
    if callable(optimizer.variables):
        return optimizer.variables()
    return optimizer.variables