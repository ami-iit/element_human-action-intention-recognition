import sys
import os
import datetime
import numpy as np
import copy
# import matplotlib.pyplot as plt
import yarp
import matplotlib.animation as animation
import time
import math
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib

# timing
def current_milli_time():
    return round(time.time() * 1000)



# YARP related code
if not yarp.Network.checkNetwork():
    print("[main] Unable to find YARP network")
yarp.Network.init()
rf = yarp.ResourceFinder()
rf.setDefaultContext("myContext")
rf.setDefaultConfigFile("default.ini")

human_kin_dyn_port = yarp.BufferedPortBottle()
human_kin_dyn_port.open("/test_visualization/humanDynamics:i")
# action_prediction_port = yarp.BufferedPortVector()
# action_prediction_port.open("/test_visualization/actionRecognition:i")
motion_prediction_port = yarp.BufferedPortVector()
motion_prediction_port.open("/test_visualization/dynamicsPredictionAll:i")

is_connected_human_kindyn = yarp.Network.connect("/humanDataAcquisition/humanKinDyn:o",
                                                 "/test_visualization/humanDynamics:i")
# is_connected_action_recognition = yarp.Network.connect("/test_moe/actionRecognition:o",
#                                                        "/test_visualization/actionRecognition:i")
is_connected_motion_prediction = yarp.Network.connect("/test_moe/dynamicPredictionAll:o",
                                                      "/test_visualization/dynamicsPredictionAll:i")
print("human kindyn port is connected: {}".format(is_connected_human_kindyn))
# print("action recognition port is connected: {}".format(is_connected_action_recognition))
print("motion prediction port is connected: {}".format(is_connected_motion_prediction))
yarp.delay(0.001)

# fig, ax = plt.subplots()
# x = np.arange(0, 2 * np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))


class PlotInferenceResults:
    def __init__(self):
        # related to figure
        font = {'size': 15}
        matplotlib.rc('font', **font)

        self.variable_idx_prediction = 2
        self.variable_idx_ground_truth = 2 * 31 + 2

        self.variableName = 'l_shoe_fz'

        self.xmin = 0.0
        self.xmax = 6.5
        self.plot_front_time = 1.2

        self.f2 = figure(num=0, figsize=(8, 3.5))  # , dpi = 100)
        # self.f0.title("joint value vs time", fontsize=12)
        self.ax01 = self.f2.subplots() # 2grid((1, 1), (0, 0))
        # self.ax02 = subplots()
        self.ax01.set_title('Foot wrench vs time', fontsize=16)
        self.ax01.set_ylim(-100, 1000)
        self.ax01.set_xlim(self.xmin, self.xmax)

        self.t = np.zeros(0)
        self.t0 = current_milli_time() / 1000.0  # seconds
        self.joint_values = np.zeros(0)
        self.p1, = self.ax01.plot(self.t, self.joint_values, 'g-', linewidth=5)

        self.t_prediction = np.zeros(0)
        self.joint_predictions = np.zeros(0)
        # self.p2 = self.ax01.scatter(self.t_prediction, self.joint_predictions)
        self.p2, = self.ax01.plot(self.t_prediction, self.joint_predictions, 'o', color='k', markersize=4, alpha=0.1)

        self.ax01.grid(True)
        # self.ax01.set_xlabel("time[sec]", fontsize=20)
        # self.ax01.set_ylabel("{} [N]".format(self.variableName), fontsize=20)
        #
        # bar plot
        # prediction_time_idx = [0, 12, 24]
        # labels = ['None', 'Rotating', 'Standing', 'Walking']
        #
        # bars = []
        # colors = []
        # # print('prediction bars: {}'.format(bars))
        # width = 0.25
        # sampling_time = 0.04
        # x0 = np.arange(0, 2 * len(labels), 2)
        # x = x0 - (len(prediction_time_idx) // 2) * width
        #
        # for i in range(len(prediction_time_idx)):
        #     self.axs_action.bar(x, bars[i], width=width, color=colors[i],
        #                         label='t + {} [sec]'.format(prediction_time_idx[i] * sampling_time))
        #     x = x + width
        #
        # self.f1 = figure(num=1, figsize=(12, 8))  # , dpi = 100)
        # self.ax11 = self.f1.subplots() # 2grid((1, 1), (0, 0))
        # self.ax11.set_title('human action prediction')
        # self.ax11.set_xlabel('human actions')
        # self.ax11.set_ylabel(ylabel='Probability')
        # self.ax11.legend(loc='upper right')
        # self.ax11.set_ylim(0, 1.0)
        # width = 0.25
        # sampling_time = 0.04
        # self.t_prediction = np.zeros(0)
        # self.joint_predictions = np.zeros(0)
        # # self.p2 = self.ax01.scatter(self.t_prediction, self.joint_predictions)
        # self.p2, = self.ax01.plot(self.t_prediction, self.joint_predictions, 'o', color='k', markersize=4, alpha=0.1)
        ######################
        ######################

        self.x = 0.0

        # related to the data
        self.timer = current_milli_time()
        self.counter = 0
        self.time_length = 100
        self.human_kin_dyn_data = []

        self.prediction_horizon = 50
        self.time_step = 0.03
        self.output_size = 12

        return

    def animate(self, dummy):
        # read human current data:
        # data manipulation
        print('timer: {}'.format(current_milli_time() - self.timer))
        self.timer = current_milli_time()
        time_now = (current_milli_time() / 1000.0) - self.t0  # seconds

        # set the human current joint values
        human_kin_dyn = human_kin_dyn_port.read(False)
        if human_kin_dyn is not None:
            tmp_joint = human_kin_dyn.get(self.variable_idx_ground_truth).asFloat64()
        else:
            return self.p1, self.p2,

        # handle data to feed to plots
        self.joint_values = append(self.joint_values, tmp_joint)
        self.t = append(self.t, time_now)

        # get all the prediction results
        human_kin_dyn_prediction = motion_prediction_port.read(False)
        if human_kin_dyn_prediction is not None:
            human_kin_dyn_prediction_data = []
            for i in range(self.variable_idx_prediction, human_kin_dyn_prediction.size(), self.output_size):
                human_kin_dyn_prediction_data.append(human_kin_dyn_prediction.get(i))

            if len(human_kin_dyn_prediction_data) != self.prediction_horizon:
                print('prediction values size {} and prediction horizon size {} are not equal'.format(
                    len(human_kin_dyn_prediction_data),  self.prediction_horizon))
                return self.p1,

            new_time_prediction = [(time_now + i * self.time_step) for i in range(self.prediction_horizon)]
            self.t_prediction = append(self.t_prediction, new_time_prediction)
            self.joint_predictions = append(self.joint_predictions, human_kin_dyn_prediction_data)

            # print('prediction shape: {}'.format(np.shape(human_kin_dyn_prediction_data)))
            #
            # self.p2.scatter(self.output_time[i][0], self.output_prediction[i][0],
            #                            marker='o',
            #                            color=(alpha, 0.2, 0.2, alpha),
            #                            zorder=2)

        self.x += 0.05
        # handling figure
        self.p1.set_data(self.t, self.joint_values)
        self.p2.set_data(self.t_prediction, self.joint_predictions)

        if time_now >= self.xmax - self.plot_front_time:
            self.p1.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            self.p2.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)

            if human_kin_dyn_prediction is not None and (time_now - self.t[0] > self.xmax):
                self.t_prediction = self.t_prediction[self.prediction_horizon:]
                self.joint_predictions = self.joint_predictions[self.prediction_horizon:]

            if human_kin_dyn is not None:
                self.t = self.t[1:]
                self.joint_values = self.joint_values[1:]

        # human_kin_dyn = human_kin_dyn_port.read(False)
        # if human_kin_dyn is not None:
        #     # for j in range(human_kin_dyn.size()):
        #     self.human_kin_dyn_data.append(human_kin_dyn.get(idx).asFloat64())
        #     # last_T_human_kin_dyn_data = human_kin_dyn_data.copy()
        #     # if last_human_kin_dyn_data is not None:
        #         #     print('last human data: {}'.format(last_human_kin_dyn_data[17]))
        #         # if len(self.human_kin_dyn_data) > self.time_length:
        #         #     self.human_kin_dyn_data.pop(0)
        #
        #     length = len(self.human_kin_dyn_data)
        #     x = np.arange(self.counter-length, self.counter)
        #
        #     line.set_data(x, self.human_kin_dyn_data)  # update the data
        #             # line.set_xdata(self.counter)  # update the data
        #         # line.set_xdata(x)
        #
        #     print('timer: {}'.format(current_milli_time()-self.timer))
        #     self.timer = current_milli_time()
        #     self.counter = self.counter + 1
        return self.p1, self.p2,

    # Init only required for blitting to give a clean slate.
    def init(self):
        # z = np.arange(0, 5000)

        # line.set_ydata(np.ma.array(x, mask=True))
        # line.set_xdata(np.ma.array(z, mask=True))
        self.p1.set_data([], [])
        return self.p1,


plot_object = PlotInferenceResults()

ani = animation.FuncAnimation(plot_object.f2, plot_object.animate,
                              interval=20, blit=False, repeat=False)
plt.show()
