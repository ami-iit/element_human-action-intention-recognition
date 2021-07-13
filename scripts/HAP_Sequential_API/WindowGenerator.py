import numpy as np
import tensorflow as tf
import random

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_input_df, val_input_df, test_input_df,
               train_target_df, val_target_df, test_target_df,
               label_columns=None):
    # Store the raw data.
    self.train_input_df = train_input_df
    self.val_input_df = val_input_df
    self.test_input_df = test_input_df

    self.train_target_df = train_target_df
    self.val_target_df = val_target_df
    self.test_target_df = test_target_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_input_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.label_columns is not None:
          labels = tf.stack(
              [labels[:, :, self.column_indices[name]] for name in self.label_columns],
              axis=-1)

      # Slicing doesn't preserve static shape information, so set the shapes
      # manually. This way the `tf.data.Datasets` are easier to inspect.
      inputs.set_shape([None, self.input_width, None])
      labels.set_shape([None, self.label_width, None])

      # input         shape: (batch_size, Tx, nx)
      # label(output) shape: (batch_size, Ty, ny)

      self._example = inputs, labels
      return inputs, labels

  def plot(self, model=None, plot_col='', max_subplots=1):
      print('plot')
      inputs, labels = self.example
      plt.figure(figsize=(12, 8))
      if not plot_col:
          print('plot_col is empty, using the first feature.')
          plot_col=list(self.column_indices.keys())[0]

      plt.title(" state: {}".format(plot_col))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))

      r = lambda: random.randint(0, 255)/255.0
      # add for each class of labels a new color for the plot
      labels_array=np.array(labels)
      labels_array = np.reshape(labels_array, len(labels_array))
      number_classes= set(labels_array)
      classes_color={}
      for i in number_classes:
          classes_color.update({i: (r(), r(), r())})

      for n in range(max_n):
          print(max_n, n+1)
          plt.subplot(max_n, 1, n + 1)
          plt.ylabel(f'{plot_col} [normed]')
          plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                   label='Inputs', marker='.', zorder=-10, color=classes_color[labels_array[n]])

          # if self.label_columns:
          #     label_col_index = self.label_columns_indices.get(plot_col, None)
          # else:
          #     label_col_index = plot_col_index
          #
          # if label_col_index is None:
          #     continue
          #
          # plt.scatter(self.label_indices, labels[n, :, label_col_index],
          #             edgecolors='k', label='Labels', c='#2ca02c', s=64)
          if model is not None:
              predictions = model(inputs)
              predictions_array = np.array(predictions)
              predictions_array = np.reshape(predictions_array, len(predictions_array))
              plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                       label='Inputs', marker='.', zorder=-10, color=classes_color[predictions_array[n] > 0])
          if n == 0:
              plt.legend()

      plt.xlabel('Time [samples]')

  def make_dataset(self, input_data, target_data):
      input_data = np.array(input_data, dtype=np.float32)
      target_data = np.array(target_data, dtype=np.float32)
      input_data = input_data[:-self.total_window_size]
      target_data = target_data[self.total_window_size-1:-1]

      ds = tf.keras.preprocessing.timeseries_dataset_from_array(
          data=input_data,
          targets=target_data,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=True,
          batch_size=32)

      # ds = ds.map(self.split_window)
      for batch in ds:
          inputs, labels = batch
          break
      self._example = inputs, labels

      return ds

  @property
  def train(self):
      return self.make_dataset(self.train_input_df, self.train_target_df)

  @property
  def val(self):
      return self.make_dataset(self.val_input_df, self.val_target_df)

  @property
  def test(self):
      return self.make_dataset(self.test_input_df, self.test_target_df)

  @property
  def example(self):
      """Get and cache an example batch of `inputs, labels` for plotting."""
      result = getattr(self, '_example', None)
      if result is None:
          print('example is empty, feeding with test set')
          # No example batch was found, so get one from the `.train` dataset
          result = next(iter(self.test))
          # And cache it for next time
          self._example = result
      return result


