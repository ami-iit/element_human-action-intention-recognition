import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from sklearn import preprocessing
from numpy import savetxt

# functions and classes to import
from WindowGeneratorMoE import WindowGeneratorMoE
from WindowGenerator import WindowGenerator
# from Utilities import get_dense_model_classification,\
#     get_cnn_model,\
#     get_lstm_regression_model_sequential,\
#     compile_model,\
#     fit_model, \
#     plot_losses,\
#     plot_accuracy,\
#     get_moe_model_four_experts,\
#     visualize_model,\
#     compile_and_fit_regression,\
#     get_moe_model_one_expert,\
#     get_refined_moe_four_expert

# from Utilities import save_nn_model
from Utilities import *

def get_time_now():
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    return date_time

# def main():
if __name__ == "__main__":

    # =====================
    # == CONFIGURATIONS ==
    # =====================

    # high level flags for training
    learn_moe_model = True
    learn_cnn_model = False
    learn_lstm_model = False
    do_performance_analysis = True
    normalize_input = True
    output_categorical = True
    save_model = True
    verbose = False

    # configurations for the datasets
    model_name = 'model'
    # models_path = '__untrack/models/RAL/'+get_time_now()
    models_path = 'NN_models/' +get_time_now() # save the trained NN model
    # data set for first 5 mins video
    data_path = '~/element_human-action-intention-recognition/dataset/lifting_test/Dataset_2022_04_04_16_17_18.txt' # read the dataset
    
    # data set division
    train_percentage = 0.7  # ! the percentage of of the time series data from starting  for training
    val_percentage = 0.2  # ! the percentage of the time series data after training data for validation
    test_percentage = 1.0 - (train_percentage + val_percentage)  # ! the amount of data at end for testing

    # L1 and L2 regularization 
    regularization_l2_gate = 1.0e-3
    regularization_l1_gate = 1.0e-3
    regularization_l2_experts = 1.0e-3
    regularization_l1_experts = 1.0e-3
    
    # dropout rate
    dropout_rate = 0.4

    user_mass = 79.0
    gravity = 9.81

    output_steps = 25  # ! at the moment only the value `1` is possible
    shift = output_steps  # ! offset, e.g., 10
    input_width = 10  # ! default: 10 
    max_subplots = 5
    max_epochs = 40  # Default: 20
    patience = 5  # ! default: 4
    number_experts_outputs = 78  # to fix later (66 joint angles, 12 wrenches)

    # =====================
    # ====== DATASET ======
    # =====================

    input_feature_list = []  # ! if empty all the columns of the dataset will be added to the dataset data frame
    # ! Examples:
    # !         features_list = ['jLeftKnee_roty_val', 'jRightKnee_roty_val']
    # !         input_feature_list = ['temperature']
    output_feature_list = ['label']  # ! the title of the column for the output dataset
    pop_list = ['time', 'label']  # ! pop list used for only input features list
    df_raw = pd.read_csv(data_path, sep=' ')  # ! raw data frame

    if input_feature_list:  # ! define the input feature list
        df_input = df_raw[input_feature_list].copy()
    else:
        df_input = df_raw.copy()

    if output_feature_list:  # ! define the output feature list
        df_output = df_raw[output_feature_list].copy()
    else:
        df_output = df_raw.copy()

    if output_categorical:  # ! get the output label in case of categorical ouputs
        df_output = pd.get_dummies(df_output)
        output_labels = df_output.keys()
        # number of categories should be equal to the size of annotation list
        number_categories = len(output_labels)  # ! the number of categories
        print('output labels : {}'.format(output_labels))

    # ! start the time from the zero, depends on the application
    if 'time' in df_raw:
        df_time = df_raw['time'] - df_raw['time'][0]
    df_input.head()

    if pop_list is not None:
        for pop_name in pop_list:
            if pop_name in df_input:
                df_input.pop(pop_name)

    if verbose:
        df_input.describe().transpose()

    # ! normalize the force/torque values with the user weight
    wrench_keys = [key for key in df_input.keys() if 'shoe' in key.lower()]
    df_input_weight_normalized = df_input
    for key in wrench_keys:
        df_input_weight_normalized[key] = df_input[key] / (user_mass * gravity)

    # ! divide the data to training, validation, test sets
    n = len(df_input_weight_normalized)
    train_input_df = df_input_weight_normalized[0:int(n * train_percentage)].copy()
    val_input_df = df_input_weight_normalized[int(n * train_percentage):int(n * (train_percentage + val_percentage))].copy()
    test_input_df = df_input_weight_normalized[int(n * (train_percentage + val_percentage)):].copy()

    train_target_df = df_output[0:int(n * train_percentage)]
    val_target_df = df_output[int(n * train_percentage):int(n * (train_percentage + val_percentage))]
    test_target_df = df_output[int(n * (train_percentage + val_percentage)):]

    # ! normalize all the data based on the training data
    if normalize_input:
        #scaler = RobustScaler() 
        #train_input_df = preprocessing.normalize(train_input_df)
        #val_input_df = preprocessing.normalize(val_input_df)
        #test_input_df = preprocessing.normalize(test_input_df)

        train_input_mean = train_input_df.mean()
        train_input_std = train_input_df.std() + 0.000001
        #savetxt('mean_data.csv', train_input_mean, delimiter=' ')
        #savetxt('std_data.csv', train_input_std, delimiter=' ')

        #print('NaN value in mean - :', np.any(np.all(train_input_mean)))
        #print('NaN value in std - :', np.any(np.all(train_input_std)))

        train_input_df = (train_input_df - train_input_mean) / train_input_std
        val_input_df = (val_input_df - train_input_mean) / train_input_std
        test_input_df = (test_input_df - train_input_mean) / train_input_std
        #savetxt('train_data.csv', train_input_df, delimiter=' ')
        #savetxt('val_data.csv', val_input_df, delimiter=' ')
        #savetxt('test_data.csv', test_input_df, delimiter=' ')

        # ! check: maybe I can delete it.
        #df_std = (df_input_weight_normalized - train_input_mean) / train_input_std
        #df_std = df_std.melt(var_name='Column', value_name='Normalized')

    # ! concatenate the two datasets to have one df
    gate_train_df = pd.concat([train_input_df, train_target_df].copy(), axis=1)
    gate_val_df = pd.concat([val_input_df, val_target_df].copy(), axis=1)
    gate_test_df = pd.concat([test_input_df, test_target_df].copy(), axis=1)
    
    expert_train_df = train_input_df.copy()
    expert_val_df = val_input_df.copy()
    expert_test_df = test_input_df.copy()

    # multi_window = WindowGeneratorMoE(input_width=input_width,
    #                                   label_width=output_steps,
    #                                   shift=shift,
    #                                   train_input_df=train_input_df,
    #                                   val_input_df=val_input_df,
    #                                   test_input_df=test_input_df,
    #                                   train_target_df=train_target_df,
    #                                   val_target_df=val_target_df,
    #                                   test_target_df=test_target_df)
    
    multi_window = WindowGenerator(input_width=input_width,
                                   label_width=output_steps,
                                   shift=shift,
                                   train_df=gate_train_df,
                                   val_df=gate_val_df,
                                   test_df=gate_test_df,
                                   output_labels=output_labels)

    multi_window.train

    if verbose:
        multi_window.plot(max_subplots=3, output_labels=output_labels)

    input_data_example, __ = multi_window.example
    input_shape = (input_data_example.shape[1], input_data_example.shape[2])
    multi_val_performance = {}
    multi_test_performance = {}

    # MoE
    if learn_moe_model:
        model_moe = get_refined_moe_four_expert(number_categories=number_categories,
                                                number_experts_outputs=number_experts_outputs,
                                                output_steps=output_steps,
                                                input_shape=input_shape,
                                                reg_l1_gate=regularization_l1_gate,
                                                reg_l2_gate=regularization_l2_gate,
                                                reg_l1_experts=regularization_l1_experts,
                                                reg_l2_experts=regularization_l2_experts,
                                                dp_rate=dropout_rate)
        model_moe.summary()
        # print(asghar)

        model_moe = compile_model(model_moe)

        history_moe = fit_model(model=model_moe,
                                window=multi_window,
                                patience=patience,
                                max_epochs=max_epochs,
                                model_path=models_path,
                                model_name=model_name + '_MoE_Best')

        if verbose:
            plot_losses(history_moe)
            plot_accuracy(history_moe)

        # history = compile_and_fit(multi_dense_model, multi_window_cpy, plot_losses=plot_losses,
        #                       patience=PATIENCE, MAX_EPOCHS=MAX_EPOCHS)

        multi_val_performance['MoE'] = model_moe.evaluate(multi_window.val)
        multi_test_performance['MoE'] = model_moe.evaluate(multi_window.test, verbose=0)
        if verbose:
            multi_window.plot(model_moe, max_subplots=2, plot_col='jRightHip_roty_val')

    # ## CONV
    if learn_cnn_model:
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
        model_cnn = get_cnn_model(number_categories, input_shape, regularization_l2_gate, dropout_rate)
        model_cnn.summary()
        model_cnn = compile_model(model_cnn)
        history_cnn = fit_model(model_cnn,
                                multi_window,
                                patience,
                                max_epochs,
                                model_path=models_path,
                                model_name=model_name + '_CNN_Best')

        if verbose:
            plot_losses(history_cnn)
            plot_accuracy(history_cnn)

        multi_val_performance['Conv'] = model_cnn.evaluate(multi_window.val)
        multi_test_performance['Conv'] = model_cnn.evaluate(multi_window.test, verbose=0)
        if verbose:
            multi_window.plot(model_cnn, max_subplots=max_subplots, output_labels=output_labels)

    # RNN
    if learn_lstm_model:
        model_lstm = get_lstm_regression_classification_model_ablation(number_categories=number_categories,
                                                                       number_experts_outputs=number_experts_outputs,
                                                                       output_steps=output_steps,
                                                                       input_shape=input_shape,
                                                                       reg_l1_gate=regularization_l1_gate,
                                                                       reg_l2_gate=regularization_l2_gate,
                                                                       reg_l1_experts=regularization_l1_experts,
                                                                       reg_l2_experts=regularization_l2_experts,
                                                                       dp_rate=dropout_rate)

        model_lstm.summary()
        model_lstm = compile_model(model_lstm)

        history_lstm = fit_model(model=model_lstm,
                                 window=multi_window,
                                 patience=patience,
                                 max_epochs=max_epochs,
                                 model_path=models_path,
                                 model_name=model_name + '_LSTM_Best')
        if verbose:
            plot_accuracy(history_lstm)

        plot_losses(history_lstm)

        multi_val_performance['LSTM'] = model_lstm.evaluate(multi_window.val)
        multi_test_performance['LSTM'] = model_lstm.evaluate(multi_window.test, verbose=0)

        if verbose:
            multi_window.plot(model_lstm, max_subplots=max_subplots, output_labels=output_labels)

    # performances
    if do_performance_analysis:
        x = np.arange(len(multi_test_performance))
        width = 0.3

        metrics_list = ['gate_output_accuracy', 'moe_output_mae']
        metrics_names = []
        for metrics_name in metrics_list:
            if learn_lstm_model:
                metric_index = model_lstm.metrics_names.index(metrics_name)
                metrics_names = model_lstm.metrics_names
            elif learn_moe_model:
                metric_index = model_moe.metrics_names.index(metrics_name)
                metrics_names = model_moe.metrics_names
            elif learn_cnn_model:
                metric_index = model_cnn.metrics_names.index(metrics_name)
                metrics_names = model_cnn.metrics_names

            val_mae = [v[metric_index] for v in multi_val_performance.values()]
            test_mae = [v[metric_index] for v in multi_test_performance.values()]
            plt.figure(figsize=(12, 8))
            plt.bar(x - 0.17, val_mae, width, label='Validation')
            plt.bar(x + 0.17, test_mae, width, label='Test')
            plt.xticks(ticks=x, labels=multi_test_performance.keys(),
                       rotation=45)
            plt.ylabel('{}'.format(metrics_name))
            _ = plt.legend()

        print('==============================')
        print('======== Test Metrics ========')
        print('==============================')
        for name, value in multi_test_performance.items():
            print('=========== {} ==========='.format(name))
            for i in range(len(metrics_names)):
                print(f'{metrics_names[i]:20s}: {value[i]:0.4f}')
        print('==============================')

    if save_model:
        if learn_moe_model:
            save_nn_model(model_moe, file_path=models_path, file_name=model_name + '_MoE')
            visualize_model(model_moe, file_path=models_path, file_name=model_name + '_MoE')
        if learn_cnn_model:
            save_nn_model(model_cnn, file_path=models_path, file_name=model_name + '_CNN')
            visualize_model(model_cnn, file_path=models_path, file_name=model_name + '_CNN')
        if learn_lstm_model:
            save_nn_model(model_lstm, file_path=models_path, file_name=model_name + '_LSTM')
            visualize_model(model_lstm, file_path=models_path, file_name=model_name + '_LSTM')


# if __name__ == "__main__":
#     main()
