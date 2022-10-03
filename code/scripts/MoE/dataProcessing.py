########################### 
##### WindowGenerator #####
###########################

#######################
##### DataUtility #####
#######################

import numpy as np
import pandas as pd
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import dataConfig as cfg
import seaborn as sns

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

        self.output_labels = output_labels # ['rising', 'squatting', 'standing']

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
        self.input_width = input_width # 10
        self.label_width = label_width # 25
        self.shift = shift # 25
        self.total_window_size = input_width + shift # 35
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
            df_keys = np.array(train_df.keys()) # from first joint position name to 'rising', 'squatting', 'standing'
            first_output_key = output_labels[0] # 'rising'
            if first_output_key in df_keys:
                first_idx_array = np.where(df_keys == first_output_key)  # ! type is array
                first_idx = int(first_idx_array[0])

                self.input_label_slice = slice(0, first_idx) # 31 positions, 31 velocities, 12 wrenches
                self.gate_output_label_slice = slice(first_idx, None) # from first_idx to the end
                # exp_output_idx= first_idx
                #exp_output_idx = 31
                self.experts_output_label_slice = slice(0, cfg.exp_output_idx)  # only joint positions
                # currently use the following one as expert outputs
                self.experts_output_label_slices = [slice(0, cfg.exp_output_idx), slice(62, 74)]  # joint positions + foot wrenches
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
        # use last 10 data frames, 74 input features (31 positions, 31 velocities, 12 wrenches)
        inputs = input_data[:, self.input_slice, self.input_label_slice]
        # predict future 25 data frames, 
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

        ds = ds.map(self.split_window)
   
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

def dataset_utility(data_path='dataset.txt',
                    output_steps=1,
                    input_width=1,
                    features_list=[],
                    pop_list=['time'],
                    plot_figures=False,
                    max_subplots=3,
                    user_weight=100):
    df_row = pd.read_csv(data_path, sep=' ')
    # slice [start:stop:step], starting from index 5 take every 6th record.
    if features_list:
        df = df_row[features_list]
    else:
        df = df_row

    if pop_list is not None:
        for pop_name in pop_list:
            if pop_name in df:
                df.pop(pop_name)

    if 'time' in df_row:
        df_time = df_row['time'] - df_row['time'][0]
    df.head()

    if plot_figures:
        plot_cols = features_list
        plot_features = df[plot_cols]
        plot_features.index = df_time
        _ = plot_features.plot(subplots=True)

        plot_features = df[plot_cols][:480]
        plot_features.index = df_time[:480]
        _ = plot_features.plot(subplots=True)

    df.describe().transpose()

    if plot_figures:
        plt.figure(figsize=(12, 8))
        plt.hist2d(df['jLeftKnee_roty_val'], df['jLeftKnee_roty_vel'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('jLeftKnee_roty_val')
        plt.ylabel('jLeftKnee_roty_vel')

        plt.figure(figsize=(12, 8))
        plt.hist2d(df['jRightKnee_roty_val'], df['jRightKnee_roty_vel'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('jRightKnee_roty_val')
        plt.ylabel('jRightKnee_roty_vel')

        plt.figure(figsize=(12, 8))
        plt.hist2d(df['jRightKnee_roty_val'], df['jLeftKnee_roty_val'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('jRightKnee_roty_val')
        plt.ylabel('jLeftKnee_roty_val')

    fft = tf.signal.rfft(df['jRightKnee_roty_val'])
    f_per_dataset = np.arange(0, len(fft))

    if plot_figures:
        plt.figure(figsize=(12, 8))
        plt.step(f_per_dataset, np.abs(fft))
        plt.xscale('log')
        # plt.ylim(0, 400000)
        plt.xlim([0.1, max(plt.xlim())])
        _ = plt.xlabel('Frequency (log scale)')

    column_indices = {name: i for i, name in enumerate(df.columns)}

    # ! normalize the force/torque values with the user weight
    wrench_keys = [key for key in df.keys() if 'shoe' in key.lower()]
    wrench_indices = [int(np.where(df.keys() == key)[0]) for key in df.keys() if 'shoe' in key.lower()]
    df_input_weight_normalized = df
    for key in wrench_keys:
        df_input_weight_normalized[key] = df[key] / user_weight

    n = len(df_input_weight_normalized)
    train_df = df_input_weight_normalized[0:int(n * 0.7)]
    val_df = df_input_weight_normalized[int(n * 0.7):int(n * 0.9)]
    test_df = df_input_weight_normalized[int(n * 0.9):]

    num_features = df_input_weight_normalized.shape[1]

    # normalize the data:
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')

    if plot_figures:
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(df.keys(), rotation=90)

        plt.figure(figsize=(12, 6))
        df2 = df.melt(var_name='Column', value_name='Normalized')
        ax2 = sns.violinplot(x='Column', y='Normalized', data=df2)
        _ = ax2.set_xticklabels(df.keys(), rotation=90)

    multi_window = WindowGenerator(input_width=input_width,
                                   label_width=output_steps,
                                   shift=output_steps,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df)
    if plot_figures:
        multi_window.plot(max_subplots=max_subplots)

    return multi_window, train_mean, train_std, wrench_indices, df


def current_milli_time():
    return round(time.time() * 1000)


class PlotInferenceResults:
    def __init__(self):
        font = {'size': 15}
        plt.rc('font', **font)
        self.plot_action = plt.figure(1, figsize=(8, 6))
        self.axs_action = self.plot_action.subplots(1)

        self.plot_motion = plt.figure(2, figsize=(8, 8))
        self.axs_motion = self.plot_motion.subplots(2)

        self.input_time = []
        self.output_time = []
        self.input_states = []
        self.output_prediction = []
        return

    def action(self, prediction, labels, prediction_time_idx):

        plt.figure(1, figsize=(8, 6))
        self.plot_action.clf()
        self.axs_action = self.plot_action.subplots(1)

        bars = []
        colors = []
        for i in prediction_time_idx:
            bars.append(prediction[i, :])
            colors.append((0, 0, 1, (1 / (i + 1)) ** (1 / 4)))
     
        width = 0.25
        sampling_time = 0.04
        x0 = np.arange(0, 2 * len(labels), 2)
        x = x0 - (len(prediction_time_idx) // 2) * width

        for i in range(len(prediction_time_idx)):
            self.axs_action.bar(x, bars[i], width=width, color=colors[i],
                                   label='t + {} [sec]'.format(prediction_time_idx[i] * sampling_time))
            x = x + width

        self.axs_action.set_title('human action prediction', fontsize=16)
        self.axs_action.set_xlabel('human actions')
        self.axs_action.set_ylabel('Probability')
        self.axs_action.legend(loc='upper left')
        plt.xticks(x0, labels)
        plt.ylim([0, 1])
        plt.pause(0.001)

        plt.show()
        plt.pause(0.001)
        plt.tight_layout()
        return

    def motion(self, time, inputs, prediction, plot_indices, plot_columns):
        prediction = np.float64(np.array(prediction))
        t_y = np.shape(prediction)[0]

        self.input_time.append(time)
        self.input_states.append(inputs[int(plot_indices[0])])

        self.output_time.append([i+time for i in range(1, t_y+1)])

        self.output_prediction.append(np.reshape(prediction[:, int(plot_indices[0])], t_y))
    
        if len(self.input_time)>50:
            self.input_time.pop(0)
            self.output_time.pop(0)
            self.input_states.pop(0)
            self.output_prediction.pop(0)
        max_n = len(plot_indices)

        # plot
        plt.figure(2, figsize=(8, 8))
        self.plot_motion.clf()
        self.axs_motion = self.plot_motion.subplots(max_n)

        for n in range(max_n):
            n_index = int(plot_indices[n])

            total_data = np.shape(self.output_time)[0]
            for i in range(np.shape(self.output_time)[0]):
                if True:  # self.output_time[i][-1] > time:
                    alpha = i / total_data
            
                    self.axs_motion[n].plot(self.output_time[i], self.output_prediction[i], color=(alpha, 0.2, 0.2, alpha))
                                               # marker='X', edgecolors='k', label='Predictions',
                                               # c='#ff7f0e', s=64, zorder=-10, alpha=0.7)
                    self.axs_motion[n].scatter(self.output_time[i][0], self.output_prediction[i][0],
                                               marker='o',
                                               color=(alpha, 0.2, 0.2, alpha),
                                               zorder=2)
                    self.axs_motion[n].scatter(self.output_time[i][-1], self.output_prediction[i][-1],
                                               marker='o',
                                               color=(alpha, 0.2, 0.2, alpha),
                                               zorder=2)

            self.axs_motion[n].plot(self.input_time, self.input_states, 'b', linewidth=4, markersize=12)


        for ax in self.axs_motion.flat:
            ax.label_outer()

        plt.show(block=False)
        plt.pause(0.0001)
        plt.tight_layout()

        return
