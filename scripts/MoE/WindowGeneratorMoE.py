import numpy as np
import tensorflow as tf
import random

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class WindowGeneratorMoE:
    def __init__(self, input_width, label_width, shift,
                 train_input_df, val_input_df, test_input_df,
                 train_target_df, val_target_df, test_target_df,
                 label_columns=None, output_labels=None):
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
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_input_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = self.input_width + self.shift

        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self._example = None

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
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        # input         shape: (batch_size, Tx, nx)
        # label(output) shape: (batch_size, Ty, ny)

        self._example = inputs, labels
        return inputs, labels

    def plot(self, model=None, plot_col='', max_subplots=1, output_labels=[]):
        print('plot')
        print('output_labels: ', output_labels)
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        max_n = min(max_subplots, len(inputs))
        ax=0
        for n in range(max_n):
            ax=ax+1
            if model is not None:
                plt.subplot(max_n, 2, ax)
            else:
                plt.subplot(max_n, 1, ax)

            if not plot_col:
                print('plot_col is empty, using the first feature.')
                plot_col = list(self.column_indices.keys())[0]
            plot_col_index = self.column_indices[plot_col]

            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(inputs[n, :, plot_col_index])

            if model is not None:
                predictions = model(inputs)
                predicted_label = np.argmax(predictions[n, :])
                label = np.argmax(labels[n, :])

                if predicted_label == label:
                    color = 'blue'
                else:
                    color = 'red'

                plt.xlabel("{} {:2.0f}% ({})".format(output_labels[predicted_label],
                                                     100 * np.max(predictions[n, :]),
                                                     output_labels[label]), color=color)

                ax = ax + 1
                plt.subplot(max_n, 2, ax)
                plt.xticks(range(output_labels.size))
                plt.yticks([])
                thisplot = plt.bar(range(output_labels.size), predictions[n, :], color="#777777")
                plt.ylim([0, 1])
                thisplot[predicted_label].set_color('red')
                thisplot[label].set_color('blue')

            else:
                label = np.argmax(labels[n, :])
                plt.xlabel("({})".format(output_labels[label]), color='blue')

    def make_gate_dataset(self, input_data, target_data):
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

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_gate_dataset(self.train_input_df, self.train_target_df)

    @property
    def val(self):
        return self.make_gate_dataset(self.val_input_df, self.val_target_df)

    @property
    def test(self):
        return self.make_gate_dataset(self.test_input_df, self.test_target_df)

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
