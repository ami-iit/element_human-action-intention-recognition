import os
import datetime

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
from Utilities import compile_and_fit
from Utilities import save_model
from Utilities import visualize_model
from Utilities import PlotLosses


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plot_figures= False

# def main():
if __name__ == "__main__":
    # Download the dataset

    model_name = 'model'
    models_path = 'models/models'
    MAX_EPOCHS = 20
    OUT_STEPS = 150
    INPUT_WIDTH = 150
    HIDDEN_LAYER_SIZE = 256
    PATIENCE = 5
    PLOT_COL = 'r_shoe_ty'
    MAX_SUBPLOTS = 5

    plot_losses = PlotLosses(file_path=models_path, file_name=model_name)

    # features_list = ['jLeftKnee_roty_val', 'jRightKnee_roty_val', 'jLeftKnee_roty_vel', 'jRightKnee_roty_vel']
    features_list = []
    pop_list = ['time']
    data_path = '/home/kourosh/icub_ws/external/element_human-action-intention-recognition/dataset/HumanActionIntentionPrediction/RawData/Dataset01/Dataset_2021_03_23_13_45_06.txt'

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

    # w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
    #                      train_df=train_df, val_df=val_df, test_df=test_df,
    #                      label_columns=['jLeftKnee_roty_val'])
    # print(w1)
    #
    # w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
    #                      train_df=train_df, val_df=val_df, test_df=test_df,
    #                      label_columns=['jLeftKnee_roty_val'])
    # print(w2)
    #
    # example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
    #                            np.array(train_df[100:100 + w2.total_window_size]),
    #                            np.array(train_df[200:200 + w2.total_window_size])])
    #
    # example_inputs, example_labels = w2.split_window(example_window)
    #
    # print('All shapes are: (batch, time, features)')
    # print(f'Window shape: {example_window.shape}')
    # print(f'Inputs shape: {example_inputs.shape}')
    # print(f'labels shape: {example_labels.shape}')
    #
    # if plot_fingures:
    #     w2.plot()

    # # Each element is an (inputs, label) pair
    # w2.train.element_spec

    # for example_inputs, example_labels in w2.train.take(1):
    #     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    #     print(f'Labels shape (batch, time, features): {example_labels.shape}')
    #
    # single_step_window = WindowGenerator(
    #     input_width=1, label_width=1, shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     label_columns=['jLeftKnee_roty_val'])
    #
    # single_step_window
    # for example_inputs, example_labels in single_step_window.train.take(1):
    #     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    #     print(f'Labels shape (batch, time, features): {example_labels.shape}')
    #
    #
    # ## BASELINE
    #
    # baseline = Baseline(label_index=column_indices['jLeftKnee_roty_val'])
    #
    # baseline.compile(loss=tf.losses.MeanSquaredError(),
    #                  metrics=[tf.metrics.MeanAbsoluteError()])
    #
    # val_performance = {}
    # performance = {}
    # val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    # performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
    #
    # wide_window = WindowGenerator(
    #     input_width=24, label_width=24, shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     label_columns=['jLeftKnee_roty_val'])
    #
    # wide_window
    # print('Input shape:', wide_window.example[0].shape)
    # print('Output shape:', baseline(wide_window.example[0]).shape)
    #
    # wide_window.plot(baseline)
    #
    # ## LINEAR
    # linear = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=1)
    # ])
    #
    # print('Input shape:', single_step_window.example[0].shape)
    # # print('Output shape:', linear(single_step_window.example[0]).shape)
    #
    #
    #
    # history = compile_and_fit(linear, single_step_window, MAX_EPOCHS=MAX_EPOCHS)
    #
    # val_performance['Linear'] = linear.evaluate(single_step_window.val)
    # performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
    #
    # print('Input shape:', wide_window.example[0].shape)
    # print('Output shape:', baseline(wide_window.example[0]).shape)
    #
    # wide_window.plot(linear)
    #
    # plt.figure(figsize=(12, 6))
    # plt.bar(x=range(len(train_df.columns)),
    #         height=linear.layers[0].kernel[:, 0].numpy())
    # axis = plt.gca()
    # axis.set_xticks(range(len(train_df.columns)))
    # _ = axis.set_xticklabels(train_df.columns, rotation=90)
    #
    #
    # ## DENSE
    # dense = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=1)
    # ])
    #
    # history = compile_and_fit(dense, single_step_window, MAX_EPOCHS=MAX_EPOCHS)
    #
    # val_performance['Dense'] = dense.evaluate(single_step_window.val)
    # performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
    #
    #
    # # MULTI-STEP DENSE
    # CONV_WIDTH = 3
    # conv_window = WindowGenerator(
    #     input_width=CONV_WIDTH,
    #     label_width=1,
    #     shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     label_columns=['jLeftKnee_roty_val'])
    #
    # conv_window
    #
    # conv_window_cpy = copy.deepcopy(conv_window)
    #
    # conv_window.plot()
    # plt.title("Given 3h as input, predict 1h into the future.")
    #
    # multi_step_dense = tf.keras.Sequential([
    #     # Shape: (time, features) => (time*features)
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(units=32, activation='relu'),
    #     tf.keras.layers.Dense(units=32, activation='relu'),
    #     tf.keras.layers.Dense(units=1),
    #     # Add back the time dimension.
    #     # Shape: (outputs) => (1, outputs)
    #     tf.keras.layers.Reshape([1, -1]),
    # ])
    #
    # print('Input shape:', conv_window.example[0].shape)
    # print('Output shape:', multi_step_dense(conv_window.example[0]).shape)
    #
    # history = compile_and_fit(multi_step_dense, conv_window_cpy, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window_cpy.val)
    # performance['Multi step dense'] = multi_step_dense.evaluate(conv_window_cpy.test, verbose=0)
    #
    # conv_window.plot(multi_step_dense)
    #
    # ## CNN
    #
    # conv_model = tf.keras.Sequential([
    #     tf.keras.layers.Conv1D(filters=32,
    #                            kernel_size=(CONV_WIDTH,),
    #                            activation='relu'),
    #     tf.keras.layers.Dense(units=32, activation='relu'),
    #     tf.keras.layers.Dense(units=1),
    # ])
    #
    # print("Conv model on `conv_window`")
    # print('Input shape:', conv_window.example[0].shape)
    # print('Output shape:', conv_model(conv_window.example[0]).shape)
    # history = compile_and_fit(conv_model, conv_window, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # val_performance['Conv'] = conv_model.evaluate(conv_window.val)
    # performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
    #
    # print("Wide window")
    # print('Input shape:', wide_window.example[0].shape)
    # print('Labels shape:', wide_window.example[1].shape)
    # print('Output shape:', conv_model(wide_window.example[0]).shape)
    #
    # LABEL_WIDTH = 24
    # INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    # wide_conv_window = WindowGenerator(
    #     input_width=INPUT_WIDTH,
    #     label_width=LABEL_WIDTH,
    #     shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     label_columns=['jLeftKnee_roty_val'])
    #
    # wide_conv_window
    #
    # wide_conv_window.plot(conv_model)
    #
    #
    #
    # ## LSTM
    # lstm_model = tf.keras.models.Sequential([
    #     # Shape [batch, time, features] => [batch, time, lstm_units]
    #     tf.keras.layers.LSTM(32, return_sequences=True),
    #     # Shape => [batch, time, features]
    #     tf.keras.layers.Dense(units=1)
    # ])
    #
    # print('Input shape:', wide_window.example[0].shape)
    # print('Output shape:', lstm_model(wide_window.example[0]).shape)
    #
    # wide_window_cpy = copy.deepcopy(wide_window)
    # history = compile_and_fit(lstm_model, wide_window_cpy, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # val_performance['LSTM'] = lstm_model.evaluate(wide_window_cpy.val)
    # performance['LSTM'] = lstm_model.evaluate(wide_window_cpy.test, verbose=0)
    # wide_window.plot(lstm_model)
    #
    #
    # ## Performance
    #
    # x = np.arange(len(performance))
    # width = 0.3
    # metric_name = 'mean_absolute_error'
    # metric_index = lstm_model.metrics_names.index('mean_absolute_error')
    # val_mae = [v[metric_index] for v in val_performance.values()]
    # test_mae = [v[metric_index] for v in performance.values()]
    #
    # plt.ylabel('mean_absolute_error [T (degC), normalized]')
    # plt.bar(x - 0.17, val_mae, width, label='Validation')
    # plt.bar(x + 0.17, test_mae, width, label='Test')
    # plt.xticks(ticks=x, labels=performance.keys(),
    #            rotation=45)
    # _ = plt.legend()
    #
    # for name, value in performance.items():
    #     print(f'{name:12s}: {value[1]:0.4f}')

    ## Multi-output models
    # single_step_window = WindowGenerator(
    #     # `WindowGenerator` returns all features as labels if you
    #     # don't set the `label_columns` argument.
    #     input_width=1, label_width=1, shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df)
    #
    # wide_window = WindowGenerator(
    #     input_width=24, label_width=24, shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df)
    #
    # for example_inputs, example_labels in wide_window.train.take(1):
    #     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    #     print(f'Labels shape (batch, time, features): {example_labels.shape}')
    #
    # baseline = Baseline()
    # baseline.compile(loss=tf.losses.MeanSquaredError(),
    #                  metrics=[tf.metrics.MeanAbsoluteError()])
    # val_performance = {}
    # performance = {}
    # val_performance['Baseline'] = baseline.evaluate(wide_window.val)
    # performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)
    #
    # dense = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=num_features)
    # ])
    #
    # history = compile_and_fit(dense, single_step_window, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # val_performance['Dense'] = dense.evaluate(single_step_window.val)
    # performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
    #
    # # %%time
    # wide_window = WindowGenerator(
    #     input_width=24, label_width=24, shift=1,    train_df=train_df, val_df=val_df, test_df=test_df)
    #
    # lstm_model = tf.keras.models.Sequential([
    #     # Shape [batch, time, features] => [batch, time, lstm_units]
    #     tf.keras.layers.LSTM(32, return_sequences=True),
    #     # Shape => [batch, time, features]
    #     tf.keras.layers.Dense(units=num_features)
    # ])
    #
    # history = compile_and_fit(lstm_model, wide_window, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
    # performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
    #
    # print()
    #
    # ##
    # # %%time
    # residual_lstm = ResidualWrapper(
    #     tf.keras.Sequential([
    #         tf.keras.layers.LSTM(32, return_sequences=True),
    #         tf.keras.layers.Dense(
    #             num_features,
    #             # The predicted deltas should start small
    #             # So initialize the output layer with zeros
    #             kernel_initializer=tf.initializers.zeros())
    #     ]))
    #
    # history = compile_and_fit(residual_lstm, wide_window, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
    # performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)
    # print()
    #
    # # performance
    # x = np.arange(len(performance))
    # width = 0.3
    #
    # metric_name = 'mean_absolute_error'
    # metric_index = lstm_model.metrics_names.index('mean_absolute_error')
    # val_mae = [v[metric_index] for v in val_performance.values()]
    # test_mae = [v[metric_index] for v in performance.values()]
    #
    # plt.bar(x - 0.17, val_mae, width, label='Validation')
    # plt.bar(x + 0.17, test_mae, width, label='Test')
    # plt.xticks(ticks=x, labels=performance.keys(),
    #            rotation=45)
    # plt.ylabel('MAE (average over all outputs)')
    # _ = plt.legend()
    #
    # for name, value in performance.items():
    #     print(f'{name:15s}: {value[1]:0.4f}')
    #

    ####################################
    ####################################
    ## Multi-step models

    multi_window = WindowGenerator(input_width=INPUT_WIDTH,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS,
                                   train_df=train_df, val_df=val_df, test_df=test_df)

    multi_window.plot(max_subplots=MAX_SUBPLOTS)
    multi_window
    multi_window_cpy = copy.deepcopy(multi_window)

    multi_val_performance = {}
    multi_performance = {}


    # last_baseline = MultiStepLastBaseline(OUT_STEPS)
    # last_baseline.compile(loss=tf.losses.MeanSquaredError(),
    #                       metrics=[tf.metrics.MeanAbsoluteError()])


    # multi_val_performance['Last'] = last_baseline.evaluate(multi_window_cpy.val)
    # multi_performance['Last'] = last_baseline.evaluate(multi_window_cpy.test, verbose=0)
    # multi_window.plot(last_baseline)

    # # similar multi step prediction
    # repeat_baseline = RepeatBaseline()
    # repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
    #                         metrics=[tf.metrics.MeanAbsoluteError()])
    #
    # multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window_cpy.val)
    # multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window_cpy.test, verbose=0)
    # multi_window.plot(repeat_baseline)
    #
    # # Single Shot
    # # Linear
    # multi_linear_model = tf.keras.Sequential([
    #     # Take the last time-step.
    #     # Shape [batch, time, features] => [batch, 1, features]
    #     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    #     # Shape => [batch, 1, out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features,
    #                           kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])
    #
    # history = compile_and_fit(multi_linear_model, multi_window_cpy, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window_cpy.val)
    # multi_performance['Linear'] = multi_linear_model.evaluate(multi_window_cpy.test, verbose=0)
    # multi_window.plot(multi_linear_model)
    #
    # # DENSE
    # multi_dense_model = tf.keras.Sequential([
    #     # Take the last time step.
    #     # Shape [batch, time, features] => [batch, 1, features]
    #     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    #     # Shape => [batch, 1, dense_units]
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     # Shape => [batch, out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features,
    #                           kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])
    #
    # history = compile_and_fit(multi_dense_model, multi_window_cpy, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window_cpy.val)
    # multi_performance['Dense'] = multi_dense_model.evaluate(multi_window_cpy.test, verbose=0)
    # multi_window.plot(multi_dense_model)
    #
    # ## CONV
    # CONV_WIDTH = 3
    # multi_conv_model = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    #     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    #     # Shape => [batch, 1, conv_units]
    #     tf.keras.layers.Conv1D(HIDDEN_LAYER_SIZE, activation='relu', kernel_size=(CONV_WIDTH)),
    #     # Shape => [batch, 1,  out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features,
    #                           kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])
    # multi_conv_model._name = 'multi_conv_model'
    #
    # history = compile_and_fit(multi_conv_model, multi_window_cpy, plot_losses=plot_losses,
    #                           patience=PATIENCE, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    #
    # multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window_cpy.val)
    # multi_performance['Conv'] = multi_conv_model.evaluate(multi_window_cpy.test, verbose=0)
    # multi_window.plot(multi_conv_model, max_subplots=MAX_SUBPLOTS, plot_col=PLOT_COL)

    # RNN
    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(HIDDEN_LAYER_SIZE, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    multi_lstm_model._name = 'multi_LSTM_model'

    history = compile_and_fit(multi_lstm_model, multi_window_cpy, plot_losses=plot_losses, patience=5, MAX_EPOCHS=MAX_EPOCHS)

    IPython.display.clear_output()
    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window_cpy.val)
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window_cpy.test, verbose=0)

    multi_window.plot(multi_lstm_model, max_subplots=MAX_SUBPLOTS, plot_col=PLOT_COL)

    # Residual Net
    # multi_residual_lstm = ResidualWrapper(
    #     tf.keras.Sequential([
    #         tf.keras.layers.LSTM(HIDDEN_LAYER_SIZE, return_sequences=True),
    #         tf.keras.layers.Dense(
    #             num_features,
    #             # The predicted deltas should start small
    #             # So initialize the output layer with zeros
    #             kernel_initializer=tf.initializers.zeros()),
    #         tf.keras.layers.Reshape([OUT_STEPS, num_features])
    #
    #     ]))
    # multi_residual_lstm._name = 'multi_residual_lstm'
    #
    # history = compile_and_fit(multi_residual_lstm, multi_window_cpy, plot_losses=plot_losses, patience=5, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    # multi_val_performance['Residual LSTM'] = multi_residual_lstm.evaluate(multi_window_cpy.val)
    # multi_performance['Residual LSTM'] = multi_residual_lstm.evaluate(multi_window_cpy.test, verbose=0)
    # print()
    # multi_window.plot(multi_residual_lstm, max_subplots=MAX_SUBPLOTS*3, plot_col=PLOT_COL)



    # Autoregressive RNN
    # feedback_model = FeedBack(units=HIDDEN_LAYER_SIZE, out_steps=OUT_STEPS, num_features=num_features)
    # feedback_model._name = 'feedback_model'
    #
    # # prediction, state = feedback_model.warmup(multi_window_cpy.example[0])
    # # prediction.shape
    # # print('Output shape (batch, time, features): ', feedback_model(multi_window_cpy.example[0]).shape)
    #
    # history = compile_and_fit(feedback_model, multi_window_cpy, plot_losses=plot_losses, patience=5, MAX_EPOCHS=MAX_EPOCHS)
    #
    # IPython.display.clear_output()
    #
    # multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window_cpy.val)
    # multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window_cpy.test, verbose=0)
    # multi_window.plot(feedback_model, max_subplots=MAX_SUBPLOTS, plot_col=PLOT_COL)
    #
    # # performances
    # x = np.arange(len(multi_performance))
    # width = 0.3
    #
    # metric_name = 'mean_absolute_error'
    # metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
    # val_mae = [v[metric_index] for v in multi_val_performance.values()]
    # test_mae = [v[metric_index] for v in multi_performance.values()]
    # plt.figure(figsize=(12, 8))
    # plt.bar(x - 0.17, val_mae, width, label='Validation')
    # plt.bar(x + 0.17, test_mae, width, label='Test')
    # plt.xticks(ticks=x, labels=multi_performance.keys(),
    #            rotation=45)
    # plt.ylabel(f'MAE (average over all times and outputs)')
    # _ = plt.legend()
    # for name, value in multi_performance.items():
    #     print(f'{name:8s}: {value[1]:0.4f}')


# if __name__ == "__main__":
#     main()
