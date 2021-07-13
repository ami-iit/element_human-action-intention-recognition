import tensorflow as tf

class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs
