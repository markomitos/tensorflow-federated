import keras
import tf_keras
import tensorflow as tf
from typing import Union


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