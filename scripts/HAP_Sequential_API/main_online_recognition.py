import sys
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
from Utilities import visualize_model
from Utilities import load_model_from_file
from Utilities import PlotLosses
from DatasetUtility import DatasetUtility
from DatasetUtility import plot_prediction
from DatasetUtility import current_milli_time
# from VisualizeHuman import visualize_human
import threading
import yarp
import time

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plot_figures = False
threadLock = threading.Lock()
threads = []
datatoprint = [0]


def print_time(threadName, threadID, delay, counter, data):
    while counter:
        time.sleep(delay)
        threadLock.acquire()
        data.append(threadID * counter)
        print(threadName, "data: ", data)
        threadLock.release()
        counter -= 1


class Inference(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.threadID = 3
        self.threadName = "inference"

    def run(self):
        print("thread starting {}".format(self.threadName))
        # threadLock.acquire()
        print_time(self.threadName, threadID=self.threadID, delay=0.1, counter=3, data=datatoprint)
        # threadLock.release()


class GetHumanData(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.threadID = 2
        self.threadName = "getData"

    def run(self):
        print("thread starting {}".format(self.threadName))
        # threadLock.acquire()
        print_time(self.threadName, threadID=self.threadID, delay=0.2, counter=2, data=datatoprint)
        # threadLock.release()


# def main():
if __name__ == "__main__":
    # parameters

    model_name = 'model_Dense'
    model_path = 'models/goodResults/2'
    data_path = '/home/kourosh/icub_ws/external/DataSet/HumanDataForActionMotionPrediction/' \
                'ActionRecognition/carefulAnnotation/2/Dataset_2021_08_19_20_06_39.txt'

    pop_list = ['time', 'label']
    features_list = []

    OUT_STEPS = 1
    INPUT_WIDTH = 15
    PLOT_COL = ['l_shoe_fz', 'r_shoe_fz', 'jLeftKnee_roty_val', 'jRightKnee_roty_val']
    MAX_SUBPLOTS = 5
    INPUT_FEATURE_SIZE = 144
    labels = ['None', 'Rotating', 'Standing', 'Walking']
    PLOT_PREDICTION = True

    # *** Create new threads ***
    # inferenceThread = Inference()
    # dataReceiverThread = GetHumanData()

    # # start the threads
    # inferenceThread.start()
    # dataReceiverThread.start()
    #
    # # Add threads to thread list
    # threads.append(inferenceThread)
    # threads.append(dataReceiverThread)
    #
    # # Wait for all threads to complete
    # for t in threads:
    #     t.join()
    # print("Exiting Main Thread")
    # sys.exit()


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
    train_mean, train_std, df = DatasetUtility(data_path=data_path, OUT_STEPS=OUT_STEPS,
                                               INPUT_WIDTH=INPUT_WIDTH,
                                               features_list=features_list, pop_list=pop_list,
                                               plot_figures=plot_figures, MAX_SUBPLOTS=MAX_SUBPLOTS)

    # window_dataset.plot(model, max_subplots=MAX_SUBPLOTS, plot_col=PLOT_COL)

    df_std = (df - train_mean) / train_std
    data = np.array(df_std)

    plot_index = np.array([])
    for feature in PLOT_COL:
        plot_index = np.append(plot_index, df.columns.get_loc(feature))

    # plt.figure(figsize=(12, 8))
    data_Tx = []
    human_data = []
    human_data = np.resize(human_data, INPUT_FEATURE_SIZE)
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
        for i in range(INPUT_FEATURE_SIZE):
            human_data[i] = (human_kin_dyn.get(i).asFloat64() - train_mean[i]) / train_std[i]
        # human_data_std = (human_data - train_mean) / train_std
        data_Tx.append(human_data)
        # print(np.shape(data_Tx))
        if np.shape(data_Tx)[0] == INPUT_WIDTH:

            # data_Tx= np.delete(data_Tx, 0, axis=0)
            # print(np.shape(data_Tx))
            human_data_Tx = np.array(data_Tx)
            # input array size: (batch_size, Tx, nx) // batch_size=1
            human_data_Tx = np.reshape(human_data_Tx, (1, INPUT_WIDTH, INPUT_FEATURE_SIZE))
            # output array size: (batch_size, Ty, nx) // batch_size=1
            tik = current_milli_time()
            # prediction = model.predict(human_data_Tx, workers=1, use_multiprocessing=True)
            prediction = model(human_data_Tx, training=False)

            tok = current_milli_time()

            # print('prediction time: {} ms', tok - tik)
            # print('prediction: {}', prediction)


            ## STREAM DATA
            bottle = prediction_port.prepare()
            bottle.clear()
            pred = np.float64(np.array(prediction))
            if np.size(pred.shape) > 2:
                pred = np.reshape(pred, (pred.shape[1] * pred.shape[2]))
            else:
                pred = np.reshape(pred, (pred.shape[0] * pred.shape[1]))
            # print("pred[0]: ", pred[0])
            for i in range(pred.size):
                bottle.push_back(pred[i])
            # print("Sending ...")
            prediction_port.write()

            data_Tx.pop(0)
            # argMax = np.argmax(pred)
            print("----- Predicted Action :{} ------, inference time[ms]: {}".format(labels[np.argmax(pred)], tok - tik))

            if PLOT_PREDICTION:
                plot_prediction(pred, labels=labels)
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
