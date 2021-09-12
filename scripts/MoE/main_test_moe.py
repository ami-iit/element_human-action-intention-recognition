import sys
import os
import datetime
import numpy as np
import tensorflow as tf
import copy

from WindowGeneratorMoE import WindowGeneratorMoE
from Utilities import visualize_model
from Utilities import load_model_from_file
from DatasetUtility import dataset_utility
from DatasetUtility import plot_motion_prediction_data, plot_action_recognition_prediction
from DatasetUtility import current_milli_time
import yarp


def get_normalized_features(data, wrench_indices, user_weight, data_mean, data_std):
    data_length = len(data_mean)
    normalized_data = []

    for i in range(data_length):
        normalized_data.append(data.get(i).asFloat64())

    for i in wrench_indices:
        normalized_data[i] = normalized_data[i] / user_weight

    for i in range(data_length):
        normalized_data[i] = (normalized_data[i] - data_mean[i]) / data_std[i]

    return normalized_data


def get_denormalized_features(normalized_data, wrench_indices, user_weight, data_mean, data_std):
    denormalized_data = None
    dat_length = len(data_mean)

    for i in range(dat_length):
        denormalized_data[i] = normalized_data[i] * data_std[i] + data_mean[i]

    for i in wrench_indices:
        denormalized_data[i] = denormalized_data[i] * user_weight

    return denormalized_data


# def main():
if __name__ == "__main__":
    # parameters
    model_name = 'gate_model_trial_MoE'
    model_path = 'models'
    data_path = '/home/kourosh/icub_ws/external/DataSet/' \
                'HumanDataForActionMotionPrediction/ActionRecognition/' \
                'carefulAnnotation/2/Dataset_2021_08_19_20_06_39.txt'

    pop_list = ['time', 'label']
    features_list = []

    plotting_features = ['l_shoe_fz', 'r_shoe_fz', 'jLeftKnee_roty_val', 'jRightKnee_roty_val']
    labels = ['None', 'Rotating', 'Standing', 'Walking']
    plot_prediction = True
    prediction_time_idx = [0, 10, 20]  # ! indexes that have been used for the timings

    user_mass = 79.0
    gravity = 9.81
    user_weight_ = user_mass * gravity

    output_steps = 25  # ! at the moment only the value `1` is possible
    shift = output_steps  # ! offset, e.g., 10
    input_width = 5  # ! default: 10
    max_subplots = 5
    train_percentage = 0.7  # ! the percentage of of the time series data from starting  for training
    val_percentage = 0.2  # ! the percentage of the time series data after training data for validation
    test_percentage = 1.0 - (train_percentage + val_percentage)  # ! the amount of data at end for testing

    ## yarp
    if not yarp.Network.checkNetwork():
        print("[main] Unable to find YARP network")

    yarp.Network.init()
    rf = yarp.ResourceFinder()
    rf.setDefaultContext("myContext")
    rf.setDefaultConfigFile("default.ini")

    human_kin_dyn_port = yarp.BufferedPortBottle()
    human_kin_dyn_port.open("/humanKinDyn:i")
    is_connected = yarp.Network.connect("/humanDataAcquisition/humanKinDyn:o", "/humanKinDyn:i")
    # is_connected = yarp.Network.isConnected("/humanDataAcquisition/humanKinDyn:o", "/humanKinDyn:i")
    print("port is connected: {}".format(is_connected))
    yarp.delay(0.5)

    prediction_port = yarp.BufferedPortVector()
    prediction_port.open("/ActionRecognition:o")

    ## model, data
    model = load_model_from_file(file_path=model_path, file_name=model_name)

    # features_list = ['jLeftKnee_roty_val', 'jRightKnee_roty_val', 'jLeftKnee_roty_vel', 'jRightKnee_roty_vel']
    _, train_mean, train_std, wrench_indices, df = dataset_utility(data_path=data_path,
                                                                   output_steps=output_steps,
                                                                   input_width=input_width,
                                                                   features_list=features_list,
                                                                   pop_list=pop_list,
                                                                   plot_figures=False,
                                                                   max_subplots=max_subplots,
                                                                   user_weight=user_weight_)

    input_feature_length = len(train_mean)

    # window_dataset.plot(model, max_subplots=MAX_SUBPLOTS, plot_col=PLOT_COL)
    # df_std = (df - train_mean) / train_std
    # data = np.array(df_std)

    # plot_index = np.array([])
    # for feature in plotting_features:
    #     plot_index = np.append(plot_index, df.columns.get_loc(feature))

    data_Tx = []
    # human_data.resize(144)
    count = 0
    while True:
        human_kin_dyn = human_kin_dyn_port.read(False)
        # bot = yarp.Bottle(human_kin_dyn)
        # print(human_kin_dyn)
        if not human_kin_dyn:  # print("no data")
            continue
        # if count == 0:
        # human_data.resize(human_kin_dyn.size())
        # print("human_kin_dyn reading ...", human_kin_dyn.size())

        human_data = get_normalized_features(human_kin_dyn, wrench_indices, user_weight_, train_mean, train_std)

        # human_data_std = (human_data - train_mean) / train_std
        data_Tx.append(human_data)
        if np.shape(data_Tx)[0] == input_width:
            # data_Tx= np.delete(data_Tx, 0, axis=0)
            # print(np.shape(data_Tx))
            human_data_Tx = np.array(data_Tx)
            # input array size: (batch_size, Tx, nx) // batch_size=1
            human_data_Tx = np.reshape(human_data_Tx, (1, input_width, input_feature_length))
            # output array size: (batch_size, Ty, nx) // batch_size=1
            tik = current_milli_time()
            # prediction = model.predict(human_data_Tx, workers=1, use_multiprocessing=True)
            predictions = model(human_data_Tx, training=False)

            tok = current_milli_time()

            # print('prediction time: {} ms', tok - tik)
            # print('prediction: {}', prediction)

            ## STREAM DATA
            bottle = prediction_port.prepare()
            bottle.clear()
            predicted_actions = np.float64(np.array(predictions[0]))
            predicted_states = np.float64(np.array(predictions[0]))

            if np.size(predicted_actions.shape) > 2:
                predicted_actions = np.reshape(predicted_actions,
                                               (predicted_actions.shape[1], predicted_actions.shape[2]))
            else:
                predicted_actions = np.reshape(predicted_actions,
                                               (predicted_actions.shape[0], predicted_actions.shape[1]))

            # print("pred[0]: ", pred[0])
            for i in range(predicted_actions.shape[0]):
                for j in range(predicted_actions.shape[1]):
                    bottle.push_back(predicted_actions[i, j])
            # print("Sending ...")
            prediction_port.write()

            data_Tx.pop(0)
            # argMax = np.argmax(pred)
            print('inference time[ms]: {}'.
                  format(tok - tik))

            if plot_prediction:
                plot_action_recognition_prediction(predicted_actions, labels=labels, prediction_time_idx= prediction_time_idx)
        # print("----------")

        # print("human_data shape: ", human_data.shape)
        count += 1

        yarp.delay(0.001)

    # n = 0
    # mean_computation_time = 0.0
    # for t in range(35000, 35300, 10):
    #     n += 1
    #     data_tmp = data[t:t + INPUT_WIDTH]
    #     # (batch_size, Tx, nx)
    #     data_tmp = np.reshape(data_tmp, (1, data_tmp.shape[0], data_tmp.shape[1]))
    #     tik = current_milli_time()
    #     prediction = model.predict(data_tmp)
    #     tok = current_milli_time()
    #     mean_computation_time += tok - tik
    #     labels = data[t + INPUT_WIDTH:t + INPUT_WIDTH + OUT_STEPS]
    #     plot_prediction(time=t, inputs=data_tmp, labels=labels, prediction=prediction,
    #                     plot_index=plot_index, PLOT_COL=PLOT_COL)
    #
    #     bottle = prediction_port.prepare()
    #     bottle.clear()
    #     pred = np.float64(np.array(prediction))
    #     pred = np.reshape(pred, (pred.shape[1] * pred.shape[2]))
    #     vec = yarp.Vector(list(pred))
    #     bottle.addList().read(vec)
    #     print("Sending ...")
    #     prediction_port.write()
    # prediction_port.close()
    # mean_computation_time = mean_computation_time / n
    # print('==> average time for computing for prediction is: {} ms'.format(mean_computation_time))

    # visualize_human()

# if __name__ == "__main__":
#     main()
