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
import yarp
from yarp import Vector


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plot_figures = False

# def main():
if __name__ == "__main__":
    ## parameters
    model_name = 'myModel'
    model_path = '/home/kourosh/icub_ws/external/element_human-action-intention-recognition/scripts/' \
                  'Sequential_API_Implementation/models'
    data_path = '/home/kourosh/icub_ws/external/element_human-action-intention-recognition/dataset/HumanActionIntentionPrediction/RawData/Dataset01/Dataset_2021_03_23_13_45_06.txt'

    MAX_EPOCHS = 20
    OUT_STEPS = 240
    INPUT_WIDTH = 10
    HIDDEN_LAYER_SIZE = 256
    PATIENCE = 5
    PLOT_COL = ['l_shoe_fz', 'r_shoe_fz', 'jLeftKnee_roty_val', 'jRightKnee_roty_val']
    MAX_SUBPLOTS = 5
    features_list = []
    pop_list = ['time']
    total_window_size = INPUT_WIDTH + OUT_STEPS

    ## yarp
    yarp.Network.init()
    rf = yarp.ResourceFinder()
    rf.setDefaultContext("myContext");
    rf.setDefaultConfigFile("default.ini")

    p = yarp.BufferedPortBottle()
    p.open("/motionPrediction:o")


    ## model, data
    model = load_model_from_file(file_path=model_path, file_name=model_name)

    # features_list = ['jLeftKnee_roty_val', 'jRightKnee_roty_val', 'jLeftKnee_roty_vel', 'jRightKnee_roty_vel']
    window_dataset, train_mean, train_std, df = DatasetUtility(data_path=data_path, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH,
                           features_list=features_list, pop_list=pop_list,
                           plot_figures=plot_figures, MAX_SUBPLOTS=MAX_SUBPLOTS)

    # window_dataset.plot(model, max_subplots=MAX_SUBPLOTS, plot_col=PLOT_COL)

    df_std = (df - train_mean) / train_std
    data = np.array(df_std)

    plot_index = np.array([])
    for feature in PLOT_COL:
        plot_index = np.append(plot_index, df.columns.get_loc(feature))

    # plt.figure(figsize=(12, 8))
    n = 0
    mean_computation_time = 0.0
    for t in range(35000, 37200, 10):
        n += 1
        data_tmp = data[t:t+INPUT_WIDTH]
        # (batch_size, Tx, nx)
        data_tmp = np.reshape(data_tmp, (1, data_tmp.shape[0], data_tmp.shape[1]))
        tik = current_milli_time()
        prediction = model.predict(data_tmp)
        tok = current_milli_time()
        mean_computation_time += tok - tik
        labels = data[t+INPUT_WIDTH:t+INPUT_WIDTH+OUT_STEPS]
        plot_prediction(time=t, inputs=data_tmp, labels=labels, prediction=prediction,
                         plot_index=plot_index, PLOT_COL=PLOT_COL)

        bottle = p.prepare()
        bottle.clear()
        pred = np.float64(np.array(prediction))
        pred= np.reshape(pred, (pred.shape[1]*pred.shape[2]))
        vec = yarp.Vector(list(pred))
        bottle.addList().read(vec)
        print("Sending ...")
        p.write()
    p.close()
    mean_computation_time = mean_computation_time/n
    print('==> average time for computing for prediction is: {} ms'.format(mean_computation_time))

#    visualize_human()

# if __name__ == "__main__":
#     main()

