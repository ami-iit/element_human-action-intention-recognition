import numpy as np
import tensorflow as tf
from copy import deepcopy
import matplotlib.pyplot as plt
import random


class WindowGenerator:
    def __init__(self,
                 input_width,
                 label_width,
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 label_columns=None,
                 output_labels=None):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.output_labels = output_labels

        # define datasets
        self._example = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the "time" window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.label_start = self.total_window_size - self.label_width

        self.input_slice = slice(0, input_width)
        self.labels_slice = slice(self.label_start, None)

        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # work around for input and output label indices
        self.input_label_slice = slice(None, None)
        self.gate_output_label_slice = slice(None, None)
        self.experts_output_label_slice = slice(None, None)

        if self.output_labels is not None:
            df_keys = np.array(train_df.keys())
            first_output_key = output_labels[0]
            if first_output_key in df_keys:
                first_idx_array = np.where(df_keys == first_output_key)  # ! type is array
                first_idx = int(first_idx_array[0])

                self.input_label_slice = slice(0, first_idx)
                self.gate_output_label_slice = slice(first_idx, None)
                # exp_output_idx= first_idx
                exp_output_idx = 66
                self.experts_output_label_slice = slice(0, exp_output_idx)  # ! currently all features : to update later
                self.experts_output_label_slices = [slice(0, exp_output_idx), slice(132, 144)]  # ! currently all features : to update later
                for i in self.experts_output_label_slices:
                    print('target features for the {}\'th slice are: {}'.format(i, df_keys[i]))

        print("init is done.")

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def plot(self, model=None, max_subplots=3, plot_col='jLeftKnee_roty_val'):
        print('plot')
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plt.title(" state: {}".format(plot_col))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [samples]')

    def split_window(self, input_data):
        print('type(input_data): {}, shape: {}'.format(type(input_data), tf.shape(input_data)))

        # input_data.set_shape([None, 30, 148]): (None, time, features)
        inputs = input_data[:, self.input_slice, self.input_label_slice]
        gate_labels = input_data[:, self.labels_slice, self.gate_output_label_slice]
        # experts_labels = input_data[:, self.labels_slice, self.experts_output_label_slice]

        experts_labels_check = [input_data[:, self.labels_slice, data_slice] for data_slice in
                                self.experts_output_label_slices]

        experts_labels = tf.concat(experts_labels_check, -1) # to check

        print('type(experts_labels): {}, shape: {}'.format(type(experts_labels), tf.shape(experts_labels)))
        print('type(experts_labels): {}, shape: {}'.format(type(experts_labels),
                                                                   tf.shape(experts_labels)))


        if self.label_columns is not None:
            gate_labels = tf.stack([gate_labels[:, :, self.column_indices[name]] for name in self.label_columns],
                                   axis=-1)
            experts_labels = tf.stack([experts_labels[:, :, self.column_indices[name]] for name in self.label_columns],
                                      axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])  # ! shape: (batch_size, Tx, nx)
        gate_labels.set_shape([None, self.label_width, None])  # ! shape: (batch_size, Ty, ny)
        experts_labels.set_shape([None, self.label_width, None])  # ! shape: (batch_size, Ty, ny)

        self._example = inputs, gate_labels

        return inputs, {"gate_output": gate_labels, "moe_output": experts_labels}

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )
        print('ds: {}'.format(ds))
        print('type(ds): {}'.format(type(ds)))

        # ds = ds.shuffle(1000000, reshuffle_each_iteration=False)
        ds = ds.map(self.split_window)
        # ds.shuffle(32, reshuffle_each_iteration=False)
        # ds.batch(32)
        #
        # validation_split = 5
        # test_split = 10
        #
        # test_ds = ds.enumerate().filter(lambda x, y: x % test_split == 0).map(lambda x, y: y)
        # train_ds_tmp = ds.enumerate().filter(lambda x, y: x % test_split != 0).map(lambda x, y: y)
        #
        # val_ds = train_ds_tmp.enumerate().filter(lambda x, y: x % validation_split == 0).map(lambda x, y: y)
        # train_ds = train_ds_tmp.enumerate().filter(lambda x, y: x % validation_split != 0).map(lambda x, y: y)

        #
        # def is_test(data_):
        #     return random.random() < test_split
        #
        # is_test = lambda x: random.random() < test_split
        # is_validation = lambda x: (not is_test(x)) and (random.random() < validation_split)
        # is_train = lambda x: (not is_validation(x)) and (not is_test(x))
        #
        # test_ds = ds.filter(is_test)
        # val_ds = ds.filter(is_validation)
        # train_ds = ds.filter(is_train)
        #
        # val_ds = val_ds.map(self.split_window)
        # train_ds = train_ds.map(self.split_window)
        #
        # val_ds = val_ds.shuffle(100000)
        # test_ds = test_ds.shuffle(100000)

        # self.train_ds = train_ds
        # self.val_ds = val_ds
        # self.test_ds = test_ds

        return ds

    @property
    def train(self):
        result = getattr(self, 'train_ds', None)
        if result is None:
            print('no training dataset, generating')
            self.train_ds = self.make_dataset(self.train_df)
        return self.train_ds

    @property
    def val(self):
        result = getattr(self, 'val_ds', None)
        if result is None:
            print('no val dataset, generating')
            self.val_ds = self.make_dataset(self.val_df)
        return self.val_ds

    @property
    def test(self):
        result = getattr(self, 'test_ds', None)
        if result is None:
            print('no test dataset, generating')
            self.test_ds = self.make_dataset(self.test_df)
        return self.test_ds

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
