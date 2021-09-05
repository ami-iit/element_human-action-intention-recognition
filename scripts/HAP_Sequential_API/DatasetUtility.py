import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import time
from WindowGenerator import WindowGenerator


def DatasetUtility(data_path='dataset.txt', OUT_STEPS=1, INPUT_WIDTH=1, features_list=[],
                   pop_list=['time'], plot_figures=False, MAX_SUBPLOTS=3):

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
        df_time = df_row['time']- df_row['time'][0]
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

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

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

    # multi_window = WindowGenerator(input_width=INPUT_WIDTH,
    #                                label_width=OUT_STEPS,
    #                                shift=OUT_STEPS,
    #                                train_input_df=train_df, val_input_df=val_df, test_input_df=test_df)
    #
    # multi_window.plot(max_subplots=MAX_SUBPLOTS)

    return train_mean, train_std, df


def plot_prediction(prediction, labels):
    print('plot_prediction')
    # plt.xticks(range(np.size(labels)))
    # plt.yticks([])
    plt.bar(labels, prediction, width=0.5)
    plt.ylim([0, 1])
    plt.title("human action prediction")
    plt.xlabel('human actions')
    plt.ylabel('Probability')
    plt.pause(0.001)
    plt.clf()

def current_milli_time():
    return round(time.time() * 1000)