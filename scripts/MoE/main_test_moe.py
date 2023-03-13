import numpy as np
from Utilities import load_model_from_file
from DatasetUtility import dataset_utility
from DatasetUtility import PlotInferenceResults
from DatasetUtility import current_milli_time
import yarp
import dataConfig as cfg

def get_normalized_features(data, wrench_indices, user_weight, data_mean, data_std):
    data_length = len(data_mean)
    normalized_data = []

    for i in range(data_length):
        normalized_data.append(data.get(i).asFloat64())
        #normalized_data.append(data[i])

    for i in wrench_indices:
        normalized_data[i] = normalized_data[i] / user_weight

    #for i in range(data_length):
    #    normalized_data[i] = (normalized_data[i] - data_mean[i]) / data_std[i]

    return normalized_data


def get_denormalized_features(normalized_data, wrench_indices, user_weight, data_mean, data_std):
    denormalized_data = []
    data_length = normalized_data.shape[-1]

    # for i in range(data_length):
    #     denormalized_data.append(normalized_data[0, i])

    denormalized_data = np.reshape(normalized_data, (-1))

    # data_tmp = denormalized_data.copy()
    # for i in range(data_length):
    #     denormalized_data[i] = denormalized_data[i] * data_std[i] + data_mean[i]

    #denormalized_data = denormalized_data * np.array(data_std[:66]) + np.array(data_mean[:66])
    # data_tmp = data_tmp*np.array(data_std[:66]) + np.array(data_mean[:66])

    # print('denormalized_data: {}'.format(denormalized_data))
    # print('data_tmp: {}'.format(data_tmp))

    # print(asghar)
    # for i in wrench_indices:
    #     denormalized_data[i] = denormalized_data[i] * user_weight

    return denormalized_data


def get_denormalized_features_all_predictions(normalized_data, denormalizing_mean, denormalizing_std, denormalizing_weight):
    shape_ = np.shape(normalized_data)
    denormalized_data = np.reshape(normalized_data, (shape_[1], shape_[2]))
    #denormalized_data = denormalized_data * denormalizing_std + denormalizing_mean
    denormalized_data = denormalized_data * denormalizing_weight
    return denormalized_data


# def main():
if __name__ == "__main__":
    # parameters
    model_name = 'model_MoE_Best'
    model_path = 'NN_models/2023-02-23 14:30:03'
    # used for data normalization
    data_path = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/01_cheng_labeled.txt' 
                
    pop_list = cfg.pop_list
    features_list = []

    #plotting_features = ['l_shoe_fz', 'r_shoe_fz', 'jLeftKnee_roty_val', 'jRightKnee_roty_val']
    
    labels = cfg.full_labels
    input_feature_list = []
    # index of each joint feature of input_feature_list
    # in the full joint featurens ('time'+66 joint_val+66 joint_vel+12 wrenches+'label')
    reduced_input_indices = cfg.reduced_input_indices
    reduce_joints = False

    user_mass = 75.0
    gravity = 9.81
    user_weight_ = user_mass * gravity

    output_steps = 50  # ! at the moment only the value `1` is possible
                       # 50*0.03=1.5s (every 30ms take a df)
    shift = output_steps  # ! offset, e.g., 10
    input_width = 10  # ! default: 10
    max_subplots = 5
    train_percentage = 0.7  # ! the percentage of of the time series data from starting  for training
    val_percentage = 0.2  # ! the percentage of the time series data after training data for validation
    test_percentage = 1.0 - (train_percentage + val_percentage)  # ! the amount of data at end for testing

    # visualization information
    plot_prediction = False
    action_prediction_time_idx = [0, 10, 20]  # ! indexes that have been used for plotting the prediction timings
    motion_prediction_time_idx = 19  # ! indexes that have been used for the prediction timings
                                    # (index+1) * 0.04 in the future
                                    # (index+1) * 0.1 in the future
                                    # (index+1) * 0.01 in the future (1s, 2s)
                                    # (index+1) * 0.03, now: 0.6s
    plot_keys = ['jRightKnee_roty_val', 'jLeftKnee_roty_val']
    plot_indices = np.array([])

    if plot_prediction:
        plot_prediction_result = PlotInferenceResults()

    # ! YARP
    if not yarp.Network.checkNetwork():
        print("[main] Unable to find YARP network")

    yarp.Network.init()
    rf = yarp.ResourceFinder()
    rf.setDefaultContext("myContext")
    rf.setDefaultConfigFile("default.ini")

    human_kin_dyn_port = yarp.BufferedPortBottle()
    human_kin_dyn_port.open("/test_moe/humanKinDyn:i")
    is_connected = yarp.Network.connect("/humanDataAcquisition/humanKinDyn:o", "/test_moe/humanKinDyn:i")
    print("port is connected: {}".format(is_connected))
    yarp.delay(0.001)

    action_prediction_port = yarp.BufferedPortVector()
    action_prediction_port.open("/test_moe/actionRecognition:o")

    motion_prediction_port = yarp.BufferedPortVector()
    motion_prediction_port.open("/test_moe/motionPrediction:o")
    motion_prediction_all_port = yarp.BufferedPortVector()
    motion_prediction_all_port.open("/test_moe/motionPredictionAll:o")

    dynamic_prediction_port = yarp.BufferedPortVector()
    dynamic_prediction_port.open("/test_moe/dynamicPrediction:o")
    dynamic_prediction_all_port = yarp.BufferedPortVector()
    dynamic_prediction_all_port.open("/test_moe/dynamicPredictionAll:o")

    # model, data
    model = load_model_from_file(file_path=model_path, file_name=model_name)

    # features_list = ['jLeftKnee_roty_val', 'jRightKnee_roty_val', 'jLeftKnee_roty_vel', 'jRightKnee_roty_vel']
    _, train_mean, train_std, wrench_indices, df = dataset_utility(data_path=data_path,
                                                                   output_steps=output_steps,
                                                                   input_width=input_width,
                                                                   features_list=input_feature_list,
                                                                   pop_list=pop_list,
                                                                   plot_figures=False,
                                                                   max_subplots=max_subplots,
                                                                   user_weight=user_weight_)

    input_feature_length = len(train_mean)
    for feature in plot_keys:
        plot_indices = np.append(plot_indices, df.columns.get_loc(feature))

    # for denormalization:
    slices = [slice(0, 31), slice(31, 43)]
    wrench_slice = slice(31, 43)
    denormalize_mean = [train_mean[i] for i in slices]
    denormalize_std = [train_std[i] for i in slices]
    denormalize_mean = np.array(np.concatenate(denormalize_mean, axis=-1))
    denormalize_std = np.array(np.concatenate(denormalize_std, axis=-1))

    denormalize_weight = np.ones(denormalize_mean.shape)
    denormalize_weight[-12:] = np.ones(12)*user_weight_
    denormalize_weight = np.array(denormalize_weight)

    motion_prediction_slice = slice(0, 31)
    dynamic_prediction_slice = slice(31, 43)


    data_Tx = []
    #count = 0
    computation_time = []
    #reduced_human_kinDyn = []
    while True:
        tik_total = current_milli_time()
        human_kin_dyn = human_kin_dyn_port.read()
        # get only desired joint features
        if reduce_joints:
            reduced_human_kinDyn = []
            for i in cfg.reduced_input_indices:
                # data from humanKinDyn:o has no 'time' column
                reduced_human_kinDyn.append(human_kin_dyn.get(i-1).asFloat64())
        
        if not human_kin_dyn:  # print("no data")
            continue

        human_data = get_normalized_features(human_kin_dyn, wrench_indices, user_weight_, train_mean, train_std)

        # human_data_std = (human_data - train_mean) / train_std
        data_Tx.append(human_data)
        # read until there are at least 10 data frames
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
            t1 = current_milli_time()

            # ! stream prediction data
            action_recognition_bottle = action_prediction_port.prepare()
            motion_prediction_bottle = motion_prediction_port.prepare()
            motion_prediction_all_bottle = motion_prediction_all_port.prepare()
            dynamic_prediction_bottle = dynamic_prediction_port.prepare()
            dynamic_prediction_all_bottle = dynamic_prediction_all_port.prepare()

            action_recognition_bottle.clear()
            motion_prediction_bottle.clear()
            motion_prediction_all_bottle.clear()
            dynamic_prediction_bottle.clear()
            dynamic_prediction_all_bottle.clear()

            t2 = current_milli_time()

            predicted_actions = np.float64(np.array(predictions[0]))
            #print("[INFO] size is: ", np.size(predicted_actions.shape))

            t3 = current_milli_time()

            predicted_motion_all = get_denormalized_features_all_predictions(
                np.float64(np.array(predictions[1])),
                denormalizing_mean=denormalize_mean,
                denormalizing_std=denormalize_std,
                denormalizing_weight=denormalize_weight)
            #print("[INFO] size is: ", predicted_motion_all.shape)
            t4 = current_milli_time()

            # all future 100 dfs (1s/0.01)
            motion_prediction_all = predicted_motion_all[:, motion_prediction_slice] # 31 poses
            dynamic_prediction_all = predicted_motion_all[:, dynamic_prediction_slice] # 12 wrenches

            # the number of future data frame
            predicted_motion = motion_prediction_all[motion_prediction_time_idx]
            predicted_dynamics = dynamic_prediction_all[motion_prediction_time_idx]

            if np.size(predicted_actions.shape) > 2:
                predicted_actions = np.reshape(predicted_actions,
                                               (predicted_actions.shape[1], predicted_actions.shape[2]))
                #print("[INFO] Now the shape is: ", predicted_actions.shape)
            else:
                predicted_actions = np.reshape(predicted_actions,
                                               (predicted_actions.shape[0], predicted_actions.shape[1]))
            t5 = current_milli_time()

            for i in range(predicted_actions.shape[0]):
                for j in range(predicted_actions.shape[1]):
                    action_recognition_bottle.push_back(predicted_actions[i, j])

            for i in range(np.size(predicted_motion)):
                motion_prediction_bottle.push_back(predicted_motion[i])

            for i in range(np.size(predicted_dynamics)):
                dynamic_prediction_bottle.push_back(predicted_dynamics[i])

            for i in range(np.shape(motion_prediction_all)[0]):
                for j in range(np.shape(motion_prediction_all)[1]):
                    motion_prediction_all_bottle.push_back(motion_prediction_all[i][j])

            for i in range(np.shape(dynamic_prediction_all)[0]):
                for j in range(np.shape(dynamic_prediction_all)[1]):
                    dynamic_prediction_all_bottle.push_back(dynamic_prediction_all[i][j])

            action_prediction_port.write()
            motion_prediction_port.write()
            motion_prediction_all_port.write()
            dynamic_prediction_port.write()
            dynamic_prediction_all_port.write()

            t6 = current_milli_time()

            data_Tx.pop(0)
            # argMax = np.argmax(pred)

            if plot_prediction:
                plot_prediction_result.action(prediction=predicted_actions,
                                              labels=labels,
                                              prediction_time_idx=action_prediction_time_idx)
    
                pass

            tok_total = current_milli_time()
            max_idx = np.argmax(predicted_actions[0, :])
            action_ = labels[max_idx]
            action_prob_ = predicted_actions[0, max_idx]
            print('inference time[ms]: {} , total time[ms]: {},     == action: {},  probability: {}'.format(
                (tok - tik),
                (tok_total - tik_total),
                action_,
                action_prob_))
            print('t2-t1: {} , t3-t2: {} t4-t3: {} t5-t4: {} t6-t5: {} '.format((t2 - t1), (t3 - t2), (t4 - t3),
                                                                                (t5 - t4), (t6 - t5)))

            # computation_time.append((tok_total-tik_total))

        # print("human_data shape: ", human_data.shape)
        #count += 1

        yarp.delay(0.001)

# if __name__ == "__main__":
#     main()
