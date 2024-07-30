
import keras
import tensorflow.keras as tf_keras
import tensorflow as tf
from absl.testing import absltest
from tensorflow_federated.python.common_libs import keras_compat
import keras.src.backend.common as keras_common


class KerasCompatTest(absltest.TestCase):

  def test_is_compiled_with_keras_model(self):
    model = keras.models.Sequential()
    model.compile()
    self.assertTrue(keras_compat.is_compiled(model))

  def test_is_compiled_with_tf_keras_model(self):
    model = tf_keras.models.Sequential()
    model.compile()
    self.assertTrue(keras_compat.is_compiled(model))

  def test_is_not_compiled_with_keras_model(self):
    model = keras.models.Sequential()
    self.assertFalse(keras_compat.is_compiled(model))

  def test_is_not_compiled_with_tf_keras_model(self):
    model = tf_keras.models.Sequential()
    self.assertFalse(keras_compat.is_compiled(model))

  def test_is_compiled_with_unsupported_type(self):
    with self.assertRaises(TypeError):
      keras_compat.is_compiled(10)

  def test_keras_dtype_to_tf_known(self):
    self.assertEqual(keras_compat.keras_dtype_to_tf('float16'), tf.float16)

  def test_keras_dtype_to_tf_unknown(self):
    self.assertEqual(keras_compat.keras_dtype_to_tf('unknown'), tf.float32)

  def test_is_keras3_with_keras_layer(self):
    layer = keras.layers.Dense(10)
    self.assertTrue(keras_compat.is_keras3(layer))

  def test_is_keras3_with_unsupported_type(self):
    self.assertFalse(keras_compat.is_keras3('not_keras_object'))

  def test_get_optimizer_variables_with_keras_optimizer(self):
    optimizer = keras.optimizers.Adam()
    self.assertIsInstance(keras_compat.get_optimizer_variables(optimizer), list)

  def test_get_optimizer_variables_with_tf_keras_optimizer(self):
    optimizer = tf_keras.optimizers.Adam()
    self.assertIsInstance(keras_compat.get_optimizer_variables(optimizer), list)

  def test_get_optimizer_variables_with_unsupported_type(self):
    with self.assertRaises(TypeError):
      keras_compat.get_optimizer_variables(10)

  def test_ref_with_tf_variable(self):
    variable = tf.Variable(0)
    self.assertEqual(keras_compat.ref(variable), variable.ref())

  def test_ref_with_keras_variable(self):
    variable = keras.Variable(0)
    self.assertEqual(keras_compat.ref(variable), id(variable))

  def test_ref_with_unsupported_type(self):
    with self.assertRaises(TypeError):
      keras_compat.ref(10)


if __name__ == '__main__':
  absltest.main()
