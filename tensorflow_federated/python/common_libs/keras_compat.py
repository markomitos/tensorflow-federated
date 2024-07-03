import keras
import tf_keras
from typing import Union


Model = Union[tf_keras.Model, keras.Model]

def is_compiled(model: Model):
    if isinstance(model, tf_keras.Model):
        return model._is_compiled
    else:
        return model.compiled