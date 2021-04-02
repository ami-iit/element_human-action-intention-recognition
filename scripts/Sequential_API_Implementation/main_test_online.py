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

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plot_figures= False

# def main():
if __name__ == "__main__":

    model_name = 'myModel'
    model_path = '/home/kourosh/icub_ws/external/element_human-action-intention-recognition/scripts/' \
                  'Sequential_API_Implementation/models'
    data_path = '/home/kourosh/icub_ws/external/element_human-action-intention-recognition/dataset/HumanActionIntentionPrediction/RawData/Dataset01/Dataset_2021_03_23_13_45_06.txt'

    MAX_EPOCHS = 20
    OUT_STEPS = 240
    INPUT_WIDTH = 10
    HIDDEN_LAYER_SIZE = 256
    PATIENCE = 5
    PLOT_COL = 'r_shoe_ty'
    MAX_SUBPLOTS = 5
    features_list = []
    pop_list = ['time']


    model = load_model_from_file(file_path=model_path, file_name=model_name)

    # features_list = ['jLeftKnee_roty_val', 'jRightKnee_roty_val', 'jLeftKnee_roty_vel', 'jRightKnee_roty_vel']
    window_dataset = DatasetUtility(data_path=data_path, OUT_STEPS=OUT_STEPS, INPUT_WIDTH=INPUT_WIDTH,
                           features_list=features_list, pop_list=pop_list,
                           plot_figures=plot_figures, MAX_SUBPLOTS=MAX_SUBPLOTS)


    window_dataset.plot(model, max_subplots=MAX_SUBPLOTS, plot_col=PLOT_COL)

    


# if __name__ == "__main__":
#     main()

