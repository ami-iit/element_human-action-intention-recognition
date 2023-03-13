import sys
import os
import datetime
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import yarp
import matplotlib.animation as animation
import time
import math
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot


# timing
def current_milli_time():
    return round(time.time() * 1000)


#labels = ["None", "standing", "stooping", "straightening",
#          "placing", "fetching", "stoop-lowering", "stoop-back"]
labels = ["rising", "squatting", "standing"]
number_categories = len(labels)

# YARP related code
if not yarp.Network.checkNetwork():
    print("[main] Unable to find YARP network")
yarp.Network.init()
rf = yarp.ResourceFinder()
rf.setDefaultContext("myContext")
rf.setDefaultConfigFile("default.ini")

action_prediction_port = yarp.BufferedPortVector()
action_prediction_port.open("/onlineAnimation/actionRecognition:i")

is_connected_human_action_prediction = yarp.Network.connect("/test_moe/actionRecognition:o",
                                                            "/onlineAnimation/actionRecognition:i")

print("action prediction port is connected: {}".format(is_connected_human_action_prediction))
yarp.delay(0.001)


class PlotInferenceResults:
    def __init__(self):
        # related to figure
        font = {'size': 15}
        matplotlib.rc('font', **font)

        self.xmin = 0.0
        self.xmax = 10.0
        self.plot_front_time = 1.2
        self.f0 = figure(num=0, figsize=(8, 3.5))#, dpi=100)

        # self.f0.title("joint value vs time", fontsize=12)
        self.ax01 = self.f0.subplots()  # 2grid((1, 1), (0, 0))
        # self.ax02 = subplots()
        self.ax01.set_title('Action recognition', fontsize=16)
        self.ax01.set_ylim(-0.1, 1.1)
        self.ax01.set_xlim(self.xmin, self.xmax)
        self.ax01.grid(True)
        self.t = np.zeros(0)
        self.t0 = current_milli_time() / 1000.0  # seconds

        self.t_prediction = np.zeros(0)
        # action: standing
        self.prediction_now0 = np.zeros(0)
        self.action_predictions0 = np.zeros(0)
        self.p1, = self.ax01.plot(self.t, self.prediction_now0, 'k-', linewidth=4, label='{}'.format(labels[0]))
        #self.p2, = self.ax01.plot(self.t_prediction, self.action_predictions0, 'o', color='k', markersize=4, alpha=0.05)

        # action: squatting
        self.prediction_now1 = np.zeros(0)
        self.action_predictions1 = np.zeros(0)
        self.p3, = self.ax01.plot(self.t, self.prediction_now1, 'b-', linewidth=4, label='{}'.format(labels[1]))
        #self.p4, = self.ax01.plot(self.t_prediction, self.action_predictions1, 'o', color='b', markersize=4, alpha=0.05)

        # action: rising
        self.prediction_now2 = np.zeros(0)
        self.action_predictions2 = np.zeros(0)
        self.p5, = self.ax01.plot(self.t, self.prediction_now2, 'r-', linewidth=4, label='{}'.format(labels[2]))
        #self.p6, = self.ax01.plot(self.t_prediction, self.action_predictions2, 'o', color='r', markersize=4, alpha=0.05)

        self.ax01.set_xlabel('Time[sec]')
        self.ax01.set_ylabel('Action Probabilities')
        # self.ax01.legend(["Rising", "Rising-future", 
        #                   "Squatting", "Squatting-future",
        #                   "Standing", "Standing-future"])
        self.ax01.legend(["Rising", 
                          "Squatting", 
                          "Standing"])
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

        # self.x = 0.0

        # related to the data
        self.timer = current_milli_time()
        self.counter = 0
        self.time_length = 100
        self.human_kin_dyn_data = []

        self.prediction_horizon = 50
        self.time_step = 0.03
        self.output_size = 3

        return

    def animate(self, dummy):
        # read human current data:
        # data manipulation
        print('timer: {}'.format(current_milli_time() - self.timer))
        self.timer = current_milli_time()
        time_now = (current_milli_time() / 1000.0) - self.t0  # seconds

        # set the human current joint values
        # joint_idx = 17
        # human_kin_dyn = human_kin_dyn_port.read(False)
        # if human_kin_dyn is not None:
        #     tmp_joint = human_kin_dyn.get(joint_idx).asFloat64()
        # else:
        #     return self.p1, self.p2,

        # get all the prediction results
        predicted_human_actions = action_prediction_port.read(False)
        if predicted_human_actions is not None:
            predicted_human_actions_data = []
    
            for i in range(predicted_human_actions.size()):
                predicted_human_actions_data.append(predicted_human_actions.get(i))

            predicted_actions_reshaped = np.reshape(predicted_human_actions_data, (-1, number_categories))

            if len(predicted_actions_reshaped) != self.prediction_horizon:
                print('prediction values size {} and prediction horizon size {} are not equal'.format(
                    len(predicted_actions_reshaped), self.prediction_horizon))
                return self.p1,
            else:
                print("[INFO] Length is: ", len(predicted_actions_reshaped))

            new_time_prediction = [(time_now + i * self.time_step) for i in range(self.prediction_horizon)]
            self.t_prediction = append(self.t_prediction, new_time_prediction)
            self.action_predictions0 = append(self.action_predictions0, predicted_actions_reshaped[:, 0])
            self.action_predictions1 = append(self.action_predictions1, predicted_actions_reshaped[:, 1])
            self.action_predictions2 = append(self.action_predictions2, predicted_actions_reshaped[:, 2])

            # handle data to feed to plots
            self.t = append(self.t, time_now)
            self.prediction_now0 = append(self.prediction_now0, predicted_actions_reshaped[0, 0])
            self.prediction_now1 = append(self.prediction_now1, predicted_actions_reshaped[0, 1])
            self.prediction_now2 = append(self.prediction_now2, predicted_actions_reshaped[0, 2])
           
    
        self.p1.set_data(self.t, self.prediction_now0)
        #self.p2.set_data(self.t_prediction, self.action_predictions0)

        self.p3.set_data(self.t, self.prediction_now1)
        #self.p4.set_data(self.t_prediction, self.action_predictions1)

        self.p5.set_data(self.t, self.prediction_now2)
        #self.p6.set_data(self.t_prediction, self.action_predictions2)

        


        if time_now >= self.xmax - self.plot_front_time:
            #self.p1.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            self.p1.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            #self.p2.axes.set_xlim(self.xmin, time_now + self.plot_front_time)

            self.p3.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            #self.p4.axes.set_xlim(self.xmin, time_now + self.plot_front_time)

            self.p5.axes.set_xlim(time_now - self.xmax + self.plot_front_time, time_now + self.plot_front_time)
            #self.p6.axes.set_xlim(self.xmin, time_now + self.plot_front_time)


            # pop the data to have faster visualization iff new data arrives
            if (predicted_human_actions is not None) and (time_now - self.t[0] > self.xmax):
                # print('deleting: ', self.t_prediction[slice(0, self.prediction_horizon)])
                self.t_prediction = np.delete(self.t_prediction, slice(0, self.prediction_horizon))
                self.action_predictions0 = self.action_predictions0[self.prediction_horizon:]
                self.action_predictions1 = self.action_predictions1[self.prediction_horizon:]
                self.action_predictions2 = self.action_predictions2[self.prediction_horizon:]
        

                self.t = self.t[1:]
                self.prediction_now0 = self.prediction_now0[1:]
                self.prediction_now1 = self.prediction_now1[1:]
                self.prediction_now2 = self.prediction_now2[1:]
               
     
        return self.p1, self.p3, self.p5

    # Init only required for blitting to give a clean slate.
    def init(self):
        # z = np.arange(0, 5000)

        # line.set_ydata(np.ma.array(x, mask=True))
        # line.set_xdata(np.ma.array(z, mask=True))
        self.p1.set_data([], [])
        return self.p1, self.p3, self.p5


plot_object = PlotInferenceResults()

ani = animation.FuncAnimation(plot_object.f0, plot_object.animate,
                              interval=20, blit=False, repeat=False, cache_frame_data=False, save_count=0)
plt.show()
