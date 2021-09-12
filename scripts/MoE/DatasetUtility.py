import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import time
from WindowGenerator import WindowGenerator


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


def plot_motion_prediction_data(time_, inputs, labels, prediction, plot_index, plot_columns):
    print('plot_prediction')
    #
    plt.title(" state: {}".format(plot_columns))
    max_n = len(plot_index)
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_columns[n]} [normed]')
        n_index = int(plot_index[n])
        input_time = np.array(range(0, inputs.shape[1])) + time_
        input_time = input_time.reshape(1, input_time.shape[0])
        output_time = np.array(range(inputs.shape[1], inputs.shape[1] + labels.shape[0])) + time_
        output_time = output_time.reshape(1, output_time.shape[0])
        plt.plot(input_time.transpose(), (inputs[:, :, n_index]).transpose(), 'b',
                 linewidth=5, label='Inputs', marker='.', zorder=10, alpha=0.7)
        plt.scatter(output_time, prediction[:, :, n_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64, zorder=-10, alpha=0.7)
        plt.scatter(output_time, labels[:, n_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64, zorder=5, alpha=0.7)

    # plt.legend()
    plt.xlabel('Time [samples]')
    plt.pause(0.02)


def plot_action_recognition_prediction(prediction, labels, prediction_time_idx):
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
