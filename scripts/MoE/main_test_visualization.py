import sys
import os
import datetime
import numpy as np
import tensorflow as tf
import copy
import matplotlib.pyplot as plt

from WindowGeneratorMoE import WindowGeneratorMoE
from Utilities import visualize_model
from Utilities import load_model_from_file
from DatasetUtility import dataset_utility
from DatasetUtility import plot_motion_prediction_data, plot_action_recognition_prediction
from DatasetUtility import PlotInferenceResults
from DatasetUtility import current_milli_time
import yarp

# def main():
if __name__ == "__main__":
    # parameters
    data_path = '/home/kourosh/icub_ws/external/DataSet/' \
                'HumanDataForActionMotionPrediction/ActionRecognition/' \
                'carefulAnnotation/2/Dataset_2021_08_19_20_06_39.txt'

    pop_list = ['time', 'label']
    features_list = []

    plotting_features = ['l_shoe_fz', 'r_shoe_fz', 'jLeftKnee_roty_val', 'jRightKnee_roty_val']
    labels = ['None', 'Rotating', 'Standing', 'Walking']
    number_categories = len(labels)

    user_mass = 79.0
    gravity = 9.81
    user_weight_ = user_mass * gravity

    number_motion_output = 66
    output_steps = 60  # ! at the moment only the value `1` is possible
    shift = output_steps  # ! offset, e.g., 10
    input_width = 5  # ! default: 10
    max_subplots = 5
    train_percentage = 0.7  # ! the percentage of of the time series data from starting  for training
    val_percentage = 0.2  # ! the percentage of the time series data after training data for validation
    test_percentage = 1.0 - (train_percentage + val_percentage)  # ! the amount of data at end for testing

    # visualization information
    action_prediction_time_idx = [0, 13, 24]  # ! indexes that have been used for plotting the prediction timings
    motion_prediction_time_idx = 10  # ! indexes that have been used for the prediciotn timings
    plot_keys = ['jRightKnee_roty_val', 'jLeftKnee_roty_val']
    plot_indices = np.array([])

    plot_prediction_result = PlotInferenceResults()

    # ! YARP
    if not yarp.Network.checkNetwork():
        print("[main] Unable to find YARP network")

    yarp.Network.init()
    rf = yarp.ResourceFinder()
    rf.setDefaultContext("myContext")
    rf.setDefaultConfigFile("default.ini")

    human_kin_dyn_port = yarp.BufferedPortBottle()
    human_kin_dyn_port.open("/test_visualization/humanKinDyn:i")
    action_prediction_port = yarp.BufferedPortVector()
    action_prediction_port.open("/test_visualization/actionRecognition:i")
    motion_prediction_port = yarp.BufferedPortVector()
    motion_prediction_port.open("/test_visualization/motionPredictionAll:i")

    is_connected_human_kindyn = yarp.Network.connect("/humanDataAcquisition/humanKinDyn:o",
                                                     "/test_visualization/humanKinDyn:i")
    is_connected_action_recognition = yarp.Network.connect("/test_moe/actionRecognition:o",
                                                           "/test_visualization/actionRecognition:i")
    is_connected_motion_prediction = yarp.Network.connect("/test_moe/motionPredictionAll:o",
                                                          "/test_visualization/motionPredictionAll:i")

    print("human kindyn port is connected: {}".format(is_connected_human_kindyn))
    print("action recognition port is connected: {}".format(is_connected_action_recognition))
    print("motion prediction port is connected: {}".format(is_connected_motion_prediction))
    yarp.delay(0.5)

    # features_list = ['jLeftKnee_roty_val', 'jRightKnee_roty_val', 'jLeftKnee_roty_vel', 'jRightKnee_roty_vel']
    _, train_mean, train_std, wrench_indices, df = dataset_utility(data_path=data_path,
                                                                   output_steps=output_steps,
                                                                   input_width=input_width,
                                                                   features_list=features_list,
                                                                   pop_list=pop_list,
                                                                   plot_figures=False,
                                                                   max_subplots=max_subplots,
                                                                   user_weight=user_weight_)

    for feature in plot_keys:
        plot_indices = np.append(plot_indices, df.columns.get_loc(feature))

    data_Tx = []
    count = 0

    last_human_kin_dyn_data = None
    last_human_kin_dyn_prediction_data = None

    while True:
        tik_total = current_milli_time()
        # ! plot human actions
        human_actions_data = []

        human_actions = action_prediction_port.read(False)
        if human_actions is not None:
            tik_action = current_milli_time()
            for i in range(human_actions.size()):
                human_actions_data.append(human_actions.get(i))

            predicted_actions = np.reshape(human_actions_data, (-1, number_categories))

            plot_prediction_result.action(prediction=predicted_actions,
                                          labels=labels,
                                          prediction_time_idx=action_prediction_time_idx)

            max_idx = np.argmax(predicted_actions[0, :])
            action_ = labels[max_idx]
            action_prob_ = predicted_actions[0, max_idx]
            tok_action = current_milli_time()

            print('action plot time[ms]: {} ,  action: {},  probability: {}'.format(
                (tok_action - tik_action),
                action_,
                action_prob_))

        # ! human motion prediction plot
        # if not human_actions.size():
        #     continue

        human_kin_dyn = human_kin_dyn_port.read(False)
        if human_kin_dyn is not None:
            human_kin_dyn_data = []
            for i in range(human_kin_dyn.size()):
                human_kin_dyn_data.append(human_kin_dyn.get(i).asFloat64())
            last_human_kin_dyn_data = human_kin_dyn_data.copy()

        human_kin_dyn_prediction = motion_prediction_port.read(False)
        if human_kin_dyn_prediction is not None:
            human_kin_dyn_prediction_data = []
            for i in range(human_kin_dyn_prediction.size()):
                human_kin_dyn_prediction_data.append(human_kin_dyn_prediction.get(i))
            human_kin_dyn_prediction_data = np.reshape(human_kin_dyn_prediction_data, (-1, number_motion_output))
            last_human_kin_dyn_prediction_data = human_kin_dyn_prediction_data.copy()

        # if last_human_kin_dyn_prediction_data is not None and (last_human_kin_dyn_data is not None):
        #     plot_prediction_result.motion(time=count,
        #                                   inputs=last_human_kin_dyn_data,
        #                                   prediction=last_human_kin_dyn_prediction_data,
        #                                   plot_indices=plot_indices,
        #                                   plot_columns=plot_keys)

            # print('to plot human actions ...', last_human_kin_dyn_prediction_data, last_human_kin_dyn_data)
            # print(asghar)
            count += 1
            last_human_kin_dyn_data = None
            last_human_kin_dyn_prediction_data = None

        # print('human_actions_data:', human_actions_data)

        #
        # predicted_actions = np.float64(np.array(predictions[0]))
        # predicted_motion = get_denormalized_features(
        #     np.float64(np.array(predictions[1]))[:, motion_prediction_time_idx, :],
        #     wrench_indices=wrench_indices,
        #     user_weight=user_mass * gravity,
        #     data_mean=train_mean,
        #     data_std=train_std)
        #
        # predicted_motion = predicted_motion[0:66]
        #
        # if np.size(predicted_actions.shape) > 2:
        #     predicted_actions = np.reshape(predicted_actions,
        #                                    (predicted_actions.shape[1], predicted_actions.shape[2]))
        # else:
        #     predicted_actions = np.reshape(predicted_actions,
        #                                    (predicted_actions.shape[0], predicted_actions.shape[1]))
        #
        # # print("pred[0]: ", pred[0])
        # for i in range(predicted_actions.shape[0]):
        #     for j in range(predicted_actions.shape[1]):
        #         action_recognition_bottle.push_back(predicted_actions[i, j])
        #
        # for i in range(np.size(predicted_motion)):
        #     motion_prediction_bottle.push_back(predicted_motion[i])
        #
        # action_prediction_port.write()
        # motion_prediction_port.write()
        #
        # data_Tx.pop(0)
        # # argMax = np.argmax(pred)

        # plot_prediction_result.motion(time=count,
        #                               inputs=human_data_Tx,
        #                               prediction=predictions[1],
        #                               plot_indices=plot_indices,
        #                               plot_columns=plot_keys)
        # if count > 50 :
        #     print(asghar)
        # pass

        # plot_motion_prediction_data(plt_=plot_motions,
        #                             time=count,
        #                             inputs=human_data_Tx,
        #                             prediction=predictions[1],
        #                             plot_indices=plot_indices,
        #                             plot_columns=plot_keys)

        tok_total = current_milli_time()
        print('total time[ms]: {} '.format(tok_total - tik_total))

        # print("human_data shape: ", human_data.shape)
        # count += 1
        yarp.delay(0.001)

# if __name__ == "__main__":
#     main()
