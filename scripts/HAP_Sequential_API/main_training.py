import os
import datetime
import sys

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import copy

from WindowGenerator import WindowGenerator
from Baseline import Baseline
from ResidualWrapper import ResidualWrapper
from MultiStepLastBaseline import MultiStepLastBaseline
from RepeatBaseline import RepeatBaseline
from FeedBack import FeedBack
from Utilities import get_dense_model, get_cnn_model, get_lstm_model, compile_model, fit_model, plot_losses, plot_accuracy
from Utilities import save_model
from Utilities import visualize_model

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
DO_DATA_PREPROCESSING = False
LEARN_DENSE_MODEL = True
LEARN_CNN_MODEL = True
LEARN_LSTM_MODEL = True
DO_PERFORMANCE_ANALYSIS = True
NORMALIZE_INPUT = True
OUTPUT_CATEGORIAL = True
NUMBER_CATEGORIES = 4
SAVE_MODEL = True
MORE_PLOTS = False
model_name = 'model'
models_path = 'models'

MAX_EPOCHS = 100  # Default: 20
OUT_STEPS = 1  # only `1` is possible
SHIFT = 1  # offset
INPUT_WIDTH = 15  # Default: 10
HIDDEN_LAYER_SIZE = 64  # Default: 256
PATIENCE = 10  # Default: 4
PLOT_COL = 'l_shoe_fz'
MAX_SUBPLOTS = 5
CONV_WIDTH = 10  # Default: 10
TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 1.0 - (TRAIN_PERCENTAGE + VAL_PERCENTAGE)

regularization_l2 = 1e-3
dropout_rate = 0.3


# def main():
if __name__ == "__main__":
    # Download the dataset

    # plot_losses = PlotLosses(file_path=models_path, file_name=model_name)

    # features_list = ['jLeftKnee_roty_val', 'jRightKnee_roty_val', 'jLeftKnee_roty_vel', 'jRightKnee_roty_vel']
    # input_feature_list = ['temperature']
    input_feature_list = []

    output_feature_list = ['label']
    pop_list = ['time', 'label']  # pop list for only input features list
    data_path = '/home/kourosh/icub_ws/external/DataSet/' \
                'HumanDataForActionMotionPrediction/ActionRecognition/' \
                'carefulAnnotation/2/Dataset_2021_08_19_20_06_39.txt'

    # df: data frame
    df_row = pd.read_csv(data_path, sep=' ')

    #     data_path = '/home/kourosh/icub_ws/external/element_human-action' \
    #                 '-intention-recognition/dataset/HumanActionIntentionPrediction/' \
    #                 'RawData/Dataset01/Dataset_2021_03_23_13_45_06.txt'
    #
    #     df_row = pd.read_csv(data_path, sep=' ')

    # slice [start:stop:step], starting from index 5 take every 6th record.
    if input_feature_list:
        df_input = df_row[input_feature_list]
    else:
        df_input = df_row

    if output_feature_list:
        df_output = df_row[output_feature_list]
    else:
        df_output = df_row

    if OUTPUT_CATEGORIAL:
        df_output = pd.get_dummies(df_output)
        output_labels = df_output.keys()

    print('01: output_labels', output_labels)

    # start the time from the zero, depends on the application
    if 'time' in df_row:
        df_time = df_row['time'] - df_row['time'][0]
    df_input.head()

    if pop_list is not None:
        for pop_name in pop_list:
            if pop_name in df_input:
                df_input.pop(pop_name)
    df_input.describe().transpose()

    if DO_DATA_PREPROCESSING:
        plot_cols = input_feature_list
        if not plot_cols:
            plot_cols = ['jRightKnee_roty_val', 'jRightKnee_roty_vel', 'r_shoe_fz']
        plot_features = df_input[plot_cols]
        plot_features.index = df_time
        _ = plot_features.plot(subplots=True)

        plot_features = df_input[plot_cols][:1480]
        plot_features.index = df_time[:1480]
        _ = plot_features.plot(subplots=True)

        # histogram
        plt.figure(figsize=(12, 8))
        plt.hist2d(df_input['jLeftKnee_roty_val'], df_input['jLeftKnee_roty_vel'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('jLeftKnee_roty_val')
        plt.ylabel('jLeftKnee_roty_vel')

        plt.figure(figsize=(12, 8))
        plt.hist2d(df_input['jRightKnee_roty_val'], df_input['r_shoe_fz'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('jRightKnee_roty_val')
        plt.ylabel('r_shoe_fz')

        plt.figure(figsize=(12, 8))
        plt.hist2d(df_input['jRightKnee_roty_val'], df_input['jLeftKnee_roty_val'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('jRightKnee_roty_val')
        plt.ylabel('jLeftKnee_roty_val')

        ## fft
        fft = tf.signal.rfft(df_input['r_shoe_fz'])
        f_per_dataset = np.arange(0, len(fft))

        plt.figure(figsize=(12, 8))
        plt.step(f_per_dataset, np.abs(fft))
        plt.xscale('log')
        # plt.ylim(0, 400000)
        plt.xlim([0.1, max(plt.xlim())])
        _ = plt.xlabel('Frequency (log scale)')

#    column_indices = {name: i for i, name in enumerate(df_input.columns)}

    n = len(df_input)
    train_input_df = df_input[0:int(n * TRAIN_PERCENTAGE)]
    val_input_df = df_input[int(n * TRAIN_PERCENTAGE):int(n * (TRAIN_PERCENTAGE + VAL_PERCENTAGE))]
    test_input_df = df_input[int(n * (TRAIN_PERCENTAGE + VAL_PERCENTAGE)):]

    train_target_df = df_output[0:int(n * TRAIN_PERCENTAGE)]
    val_target_df = df_output[int(n * TRAIN_PERCENTAGE):int(n * (TRAIN_PERCENTAGE + VAL_PERCENTAGE))]
    test_target_df = df_output[int(n * (TRAIN_PERCENTAGE + VAL_PERCENTAGE)):]

    num_features = df_input.shape[1]

    # normalize the data:
    if NORMALIZE_INPUT:
        train_input_mean = train_input_df.mean()
        train_input_std = train_input_df.std()

        train_input_df = (train_input_df - train_input_mean) / train_input_std
        val_input_df = (val_input_df - train_input_mean) / train_input_std
        test_input_df = (test_input_df - train_input_mean) / train_input_std

        df_std = (df_input - train_input_mean) / train_input_std
        df_std = df_std.melt(var_name='Column', value_name='Normalized')

    if DO_DATA_PREPROCESSING:
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(df_input.keys(), rotation=90)

        plt.figure(figsize=(12, 6))
        df2 = df_input.melt(var_name='Column', value_name='Normalized')
        ax2 = sns.violinplot(x='Column', y='Normalized', data=df2)
        _ = ax2.set_xticklabels(df_input.keys(), rotation=90)

        w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                             train_df=train_input_df, val_df=val_input_df, test_df=test_input_df,
                             label_columns=['jLeftKnee_roty_val'])
        print(w1)

        w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                             train_df=train_input_df, val_df=val_input_df, test_df=test_input_df,
                             label_columns=['jLeftKnee_roty_val'])
        print(w2)

        example_window = tf.stack([np.array(train_input_df[:w2.total_window_size]),
                                   np.array(train_input_df[100:100 + w2.total_window_size]),
                                   np.array(train_input_df[200:200 + w2.total_window_size])])

        example_inputs, example_labels = w2.split_window(example_window)

        print('All shapes are: (batch, time, features)')
        print(f'Window shape: {example_window.shape}')
        print(f'Inputs shape: {example_inputs.shape}')
        print(f'labels shape: {example_labels.shape}')

        w2.plot()

        # Each element is an (inputs, label) pair
        w2.train.element_spec

        for example_inputs, example_labels in w2.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')

        single_step_window = WindowGenerator(
            input_width=1, label_width=1, shift=1,
            train_df=train_input_df, val_df=val_input_df, test_df=test_input_df,
            label_columns=['jLeftKnee_roty_val'])

        single_step_window
        for example_inputs, example_labels in single_step_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')

    multi_window = WindowGenerator(input_width=INPUT_WIDTH,
                                   label_width=OUT_STEPS,
                                   shift=SHIFT,
                                   train_input_df=train_input_df, val_input_df=val_input_df, test_input_df=test_input_df,
                                   train_target_df=train_target_df, val_target_df=val_target_df, test_target_df=test_target_df)

    multi_window.train
    print('02: output_labels', output_labels)
    if MORE_PLOTS:
        multi_window.plot(max_subplots=3,  output_labels=output_labels)

    multi_window_cpy = copy.deepcopy(multi_window)

    input_data_example, __ = multi_window_cpy.example
    input_shape = (input_data_example.shape[1], input_data_example.shape[2])
    multi_val_performance = {}
    multi_performance = {}

    # DENSE
    if LEARN_DENSE_MODEL:

        model_dense = get_dense_model(NUMBER_CATEGORIES, input_shape, regularization_l2, dropout_rate)
        model_dense = compile_model(model_dense)
        history_dense = fit_model(model_dense,
                                  multi_window_cpy,
                                  PATIENCE,
                                  MAX_EPOCHS,
                                  model_path=models_path,
                                  model_name=model_name + '_Dense_Best')
        plot_losses(history_dense)
        plot_accuracy(history_dense)

        # history = compile_and_fit(multi_dense_model, multi_window_cpy, plot_losses=plot_losses,
        #                       patience=PATIENCE, MAX_EPOCHS=MAX_EPOCHS)

        IPython.display.clear_output()
        multi_val_performance['Dense'] = model_dense.evaluate(multi_window_cpy.val)
        multi_performance['Dense'] = model_dense.evaluate(multi_window_cpy.test, verbose=0)
        if MORE_PLOTS:
            multi_window.plot(model_dense, max_subplots=3, output_labels=output_labels)

    # ## CONV
    if LEARN_CNN_MODEL:
        # multi_conv_model = tf.keras.Sequential([
        #     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        #     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        #     # Shape => [batch, 1, conv_units]
        #     tf.keras.layers.Conv1D(HIDDEN_LAYER_SIZE, activation='relu', kernel_size=(CONV_WIDTH), input_shape=()),
        #     # Shape => [batch, 1,  out_steps*features]
        #     tf.keras.layers.Dense(OUT_STEPS * NUMBER_CATEGORIES,
        #                           kernel_initializer=tf.initializers.zeros(), activation='softmax'),
        #     # Shape => [batch, out_steps, features]
        #     tf.keras.layers.Reshape([OUT_STEPS*NUMBER_CATEGORIES])
        # ])
        # multi_conv_model._name = 'multi_conv_model'
        #
        # history = compile_and_fit(multi_conv_model, multi_window_cpy, plot_losses=plot_losses,
        #                           patience=PATIENCE, MAX_EPOCHS=MAX_EPOCHS)
        model_cnn = get_cnn_model(NUMBER_CATEGORIES, input_shape, regularization_l2, dropout_rate)
        model_cnn.summary()
        model_cnn = compile_model(model_cnn)
        history_cnn = fit_model(model_cnn,
                                multi_window_cpy,
                                PATIENCE,
                                MAX_EPOCHS,
                                model_path=models_path,
                                model_name=model_name + '_CNN_Best')
        plot_losses(history_cnn)
        plot_accuracy(history_cnn)

        IPython.display.clear_output()

        multi_val_performance['Conv'] = model_cnn.evaluate(multi_window_cpy.val)
        multi_performance['Conv'] = model_cnn.evaluate(multi_window_cpy.test, verbose=0)
        if MORE_PLOTS:
            multi_window.plot(model_cnn, max_subplots=MAX_SUBPLOTS,  output_labels=output_labels)

    # RNN
    if LEARN_LSTM_MODEL:
        model_lstm = get_lstm_model(NUMBER_CATEGORIES, input_shape, regularization_l2, dropout_rate)
        model_lstm.summary()
        model_lstm = compile_model(model_lstm)
        history_lstm = fit_model(model_lstm,
                                 multi_window_cpy,
                                 PATIENCE,
                                 MAX_EPOCHS,
                                 model_path=models_path,
                                 model_name=model_name + '_LSTM_Best')
        plot_losses(history_lstm)
        plot_accuracy(history_lstm)

        IPython.display.clear_output()
        multi_val_performance['LSTM'] = model_lstm.evaluate(multi_window_cpy.val)
        multi_performance['LSTM'] = model_lstm.evaluate(multi_window_cpy.test, verbose=0)

        if MORE_PLOTS:
            multi_window.plot(model_lstm, max_subplots=MAX_SUBPLOTS, output_labels=output_labels)

    # performances
    if DO_PERFORMANCE_ANALYSIS:
        x = np.arange(len(multi_performance))
        width = 0.3

        metric_name = 'accuracy'
        if LEARN_LSTM_MODEL:
            metric_index = model_lstm.metrics_names.index('accuracy')
        elif LEARN_DENSE_MODEL:
            metric_index = model_dense.metrics_names.index('accuracy')
        elif LEARN_CNN_MODEL:
            metric_index = model_cnn.metrics_names.index('accuracy')

        val_mae = [v[metric_index] for v in multi_val_performance.values()]
        test_mae = [v[metric_index] for v in multi_performance.values()]
        plt.figure(figsize=(12, 8))
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=multi_performance.keys(),
                   rotation=45)
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()
        for name, value in multi_performance.items():
            print(f'{name:8s}: {value[1]:0.4f}')

    if SAVE_MODEL:
        if LEARN_DENSE_MODEL:
            save_model(model_dense, file_path=models_path, file_name=model_name + '_Dense')
        if LEARN_CNN_MODEL:
            save_model(model_cnn, file_path=models_path, file_name=model_name + '_CNN')
        if LEARN_LSTM_MODEL:
            save_model(model_lstm, file_path=models_path, file_name=model_name + '_LSTM')

# if __name__ == "__main__":
#     main()
