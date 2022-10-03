import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import time
from WindowGenerator import WindowGenerator
import copy


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

    # print('wrench_keys: {}'.format(wrench_keys))
    # print('wrench_indices: {}'.format(wrench_indices))

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


def plot_motion_prediction_data(plt_, time, inputs, prediction, plot_indices, plot_columns):
    plt.title('state prediction')
    prediction = np.float64(np.array(prediction))
    Tx= np.shape(inputs)[1]
    print('input shape: {}'.format(np.shape(inputs)))
    Ty= np.shape(prediction)[1]
    max_n = len(plot_indices)
    axs_loss = plt_.subplots(max_n)

    for n in range(max_n):
        # plt_.subplot(max_n, 1, n + 1)
        axs_loss[n].set(ylabel=f'{plot_columns[n]}')
        n_index = int(plot_indices[n])
        input_time = np.array(range(-Tx, 0)) + time + 1  # ! time: (-Tx + time, time)
        input_time = input_time.reshape(Tx, 1)
        print(input_time)
        output_time = np.array(range(1, Ty+1)) + time
        output_time = output_time.reshape(Ty, 1)
        axs_loss[n].plot(input_time, inputs[:, :, n_index].transpose(), 'b',
                 linewidth=5, label='Inputs', marker='.', zorder=10, alpha=0.7)
        axs_loss[n].plot(output_time, prediction[:, :, n_index].transpose(), 'r',
                 linewidth=5, label='prediction', marker='.', zorder=10, alpha=0.6)
        # plt.scatter(output_time, prediction[:, :, n_index],
        #             marker='X', edgecolors='k', label='Predictions',
        #             c='#ff7f0e', s=64, zorder=-10, alpha=0.7)

        # plt.scatter(output_time, labels[:, n_index],
        #             edgecolors='k', label='Labels', c='#2ca02c', s=64, zorder=5, alpha=0.7)
    plt.legend()
    plt.xlabel('Time [samples]')
    plt.pause(0.001)


def plot_action_recognition_prediction(plt_, prediction, labels, prediction_time_idx):
    print('plot_prediction')
    bars = []
    colors = []
    for i in prediction_time_idx:
        bars.append(prediction[i, :])
        colors.append((0, 0, 1, (1/(i+1))**(1/4)))
    print('prediction bars: {}'.format(bars))
    width = 0.25
    sampling_time = 0.04
    x0 = np.arange(0, 2*len(labels), 2)
    x = x0 - (len(prediction_time_idx)//2) * width

    for i in range(len(prediction_time_idx)):
        plt.bar(x, bars[i], width=width, color=colors[i],
                label='t + {} [sec]'.format(prediction_time_idx[i]*sampling_time))
        x = x + width

    plt.xticks(x0, labels)
    plt.ylim([0, 1])
    plt.title("human action prediction")
    plt.xlabel('human actions')
    plt.ylabel('Probability')
    plt.legend()
    plt.pause(0.001)
    plt.clf()


def current_milli_time():
    return round(time.time() * 1000)


class PlotInferenceResults:
    def __init__(self):
        # variables
        self.input_time = []
        self.output_time = []
        self.input_states = []
        self.output_prediction = []

        # human action plot
        self.plot_action = plt.figure(1, figsize=(8, 6))
        self.fig_action, self.axs_action = self.plot_action.subplots(1)





        # human motion plot
        self.plot_motion = plt.figure(2, figsize=(8, 8))
        self.fig_motion, self.axs_motion = self.plot_motion.subplots(2)


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
        # print('prediction bars: {}'.format(bars))
        width = 0.25
        sampling_time = 0.04
        x0 = np.arange(0, 2 * len(labels), 2)
        x = x0 - (len(prediction_time_idx) // 2) * width

        for i in range(len(prediction_time_idx)):
            self.axs_action.bar(x, bars[i], width=width, color=colors[i],
                                   label='t + {} [sec]'.format(prediction_time_idx[i] * sampling_time))
            x = x + width

        self.axs_action.set_title('human action prediction')
        self.axs_action.set(Xlabel='human actions')
        self.axs_action.set(ylabel='Probability')
        self.axs_action.legend(loc='upper right')
        plt.xticks(x0, labels)
        plt.ylim([0, 1])
        plt.pause(0.001)

        # for ax in self.axs_metrics.flat:
        #     ax.label_outer()
        plt.show()
        plt.pause(0.001)
        plt.tight_layout()
        return

    def motion(self, time, inputs, prediction, plot_indices, plot_columns):
        # data processing
        # inputs = inputs.tolist()
        # print('inputs type: {} , shape : {}'.format(type(inputs), np.shape(inputs)))
        # print('prediction type: {} , shape : {}'.format(type(prediction), np.shape(prediction)))

        prediction = np.float64(np.array(prediction))
        t_x = np.shape(inputs)[1]
        t_y = np.shape(prediction)[1]

        if len(self.input_time):
            self.input_time.append(time)
            self.input_states.append(inputs[:, -1, int(plot_indices[0])])
        else:
            self.input_time = [i+time+1 for i in range(-t_x, 0)]   # ! time: (-Tx + time, time)
            self.input_states = (inputs[:, :, int(plot_indices[0])].transpose()).tolist()

        # print('output times before appending; current time: {}, future horizon: {}'.format(time, self.output_time))
        self.output_time.append([i+time for i in range(1, t_y+1)])
        # print('output times after appending; current time: {}, future horizon: {}'.format(time, self.output_time))

        self.output_prediction.append(np.reshape(prediction[:, :, int(plot_indices[0])], t_y))
        # if not self.input_time:
        #     np.array(range(-t_x, 0)) + time + 1  # ! time: (-Tx + time, time)
        # else:
        #     self.input_time.append(time)
        #
        max_n = len(plot_indices)

        # plot
        plt.figure(2, figsize=(8, 8))
        self.plot_motion.clf()
        self.axs_motion = self.plot_motion.subplots(2)

        for n in range(max_n):
            n_index = int(plot_indices[n])
            # input_time = np.array(range(-t_x, 0)) + time + 1  # ! time: (-Tx + time, time)
            # input_time = input_time.reshape(t_x, 1)
            # # output_time = np.array(range(1, t_y + 1)) + time
            # output_time = output_time.reshape(t_y, 1)
            self.axs_motion[n].plot(self.input_time, self.input_states, 'b', linewidth=4, markersize=12)

            for i in range(np.shape(self.output_time)[0]):
                if True:  # self.output_time[i][-1] > time:
                    # self.axs_motion[n].plot(self.output_time[i], self.output_prediction[i], 'r')
                    # print('output timings: current time: {}, future horizon: {}'.format(time, self.output_time[i]))
                    self.axs_motion[n].scatter(self.output_time[i], self.output_prediction[i],
                                               marker='X', edgecolors='k', label='Predictions',
                                               c='#ff7f0e', s=64, zorder=-10, alpha=0.7)

            # plt.scatter(output_time, prediction[:, :, n_index],
            #             marker='X', edgecolors='k', label='Predictions',
            #             c='#ff7f0e', s=64, zorder=-10, alpha=0.7)

            # self.axs_motion[0].set_title('motion prediction for {} time horizon'.format(t_y))
            # self.axs_motion[n].legend(['input', 'prediction'], loc='upper right')
            #
            # self.axs_motion[n].set(ylabel=f'{plot_columns[n]}')
            # self.axs_motion[n].set(xlabel='time steps', ylabel='{}'.format(plot_columns[n]))

        for ax in self.axs_motion.flat:
            ax.label_outer()

        plt.show(block=False)
        plt.pause(0.0001)
        plt.tight_layout()

        return

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
        # print('prediction bars: {}'.format(bars))
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

        # for ax in self.axs_metrics.flat:
        #     ax.label_outer()
        plt.show()
        plt.pause(0.001)
        plt.tight_layout()
        return

    def motion(self, time, inputs, prediction, plot_indices, plot_columns):
        # data processing
        # inputs = inputs.tolist()
        # print('inputs type: {} , shape : {}'.format(type(inputs), np.shape(inputs)))
        # print('prediction type: {} , shape : {}'.format(type(prediction), np.shape(prediction)))

        prediction = np.float64(np.array(prediction))
        # t_x = np.shape(inputs)[1]
        t_y = np.shape(prediction)[0]

        # if len(self.input_time):
        self.input_time.append(time)
        self.input_states.append(inputs[int(plot_indices[0])])
        # else:
            # self.input_time = [i+time+1 for i in range(-t_x, 0)]   # ! time: (-Tx + time, time)
            # self.input_states = (inputs[:, :, int(plot_indices[0])].transpose()).tolist()

        # print('output times before appending; current time: {}, future horizon: {}'.format(time, self.output_time))
        self.output_time.append([i+time for i in range(1, t_y+1)])
        # print('output times after appending; current time: {}, future horizon: {}'.format(time, self.output_time))

        self.output_prediction.append(np.reshape(prediction[:, int(plot_indices[0])], t_y))
        # print('prediction[:, int(plot_indices[0])]: ',  self.output_prediction)
        # print('output time [:, int(plot_indices[0])]: ',  self.output_time)

        # if not self.input_time:
        #     np.array(range(-t_x, 0)) + time + 1  # ! time: (-Tx + time, time)
        # else:
        #     self.input_time.append(time)
        #
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
            # input_time = np.array(range(-t_x, 0)) + time + 1  # ! time: (-Tx + time, time)
            # input_time = input_time.reshape(t_x, 1)
            # # output_time = np.array(range(1, t_y + 1)) + time
            # output_time = output_time.reshape(t_y, 1)

            total_data = np.shape(self.output_time)[0]
            for i in range(np.shape(self.output_time)[0]):
                if True:  # self.output_time[i][-1] > time:
                    # self.axs_motion[n].plot(self.output_time[i], self.output_prediction[i], 'r')
                    # print('output timings: current time: {}, future horizon: {}'.format(time, self.output_time[i]))
                    alpha = i / total_data
                    # print('alpha: {}'.format(alpha))
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

            # plt.scatter(output_time, prediction[:, :, n_index],
            #             marker='X', edgecolors='k', label='Predictions',
            #             c='#ff7f0e', s=64, zorder=-10, alpha=0.7)

            # self.axs_motion[0].set_title('motion prediction for {} time horizon'.format(t_y))
            # self.axs_motion[n].legend(['input', 'prediction'], loc='upper right')
            #
            # self.axs_motion[n].set(ylabel=f'{plot_columns[n]}')
            # self.axs_motion[n].set(xlabel='time steps', ylabel='{}'.format(plot_columns[n]))

        for ax in self.axs_motion.flat:
            ax.label_outer()

        plt.show(block=False)
        plt.pause(0.0001)
        plt.tight_layout()

        return



