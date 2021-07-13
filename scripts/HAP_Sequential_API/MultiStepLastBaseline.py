import tensorflow as tf

class MultiStepLastBaseline(tf.keras.Model):
  def __init__(self, OUT_STEPS):
    super().__init__()
    self.OUT_STEPS = OUT_STEPS
    print('MultiStepLastBaseline constructor.')

  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, self.OUT_STEPS, 1])

  def __repr__(self):
    return '\n'.join([
        f'MultiStepLastBaseline',
        f'IOUT_STEPS: {self.OUT_STEPS}'])

