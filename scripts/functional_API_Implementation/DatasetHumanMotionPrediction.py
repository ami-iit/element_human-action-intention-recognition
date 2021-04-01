from __future__ import print_function

from Dataset import Dataset
import numpy as np
import math
import random
import matplotlib.pyplot as plt


class DatasetHumanMotionPrediction(Dataset):
    def __init__(self):
        self.data_raw = []
        return

    def prepare_data(self, batch_data, Tx, Ty, Tx0= 0, Ty0=1):

        batch_x = batch_data[:, range(Tx0,  Tx0+Tx), :]
        # batch_x = np.reshape(batch_x, (np.size(batch_data, 0), Tx, np.size(batch_data, 2)))
        batch_y = batch_data[:, range(Ty0, Ty0+Ty), :]
        # batch_y = np.reshape(batch_y, (np.size(batch_y, 1), np.size(batch_y, 0), np.size(batch_y, 2)))
        batch_y = np.swapaxes(batch_y, 0, 1)

        # batch_x shape: (batch_size, Tx, nx)
        # batch_y shape: (batch_size, ny, Ty)
        return batch_x, batch_y


    def prepare_data_batches(self, feature_list, data_raw, seq_length):
        batch_t = []
        batch_x = []
        if not feature_list:
            feature_list = list(data_raw[0].keys())
            feature_list.remove('time')

        for data in data_raw:
            for count in range(len(data[feature_list[0]]) - seq_length):
                x_ = []
                for j in range(count, count+seq_length):
                    x = []
                    for feature in feature_list:
                        x.append((data[feature])[j])
                    x_.append(x)
                batch_x.append(x_)

        # shape: (batch_size, seq_length, output_dim)
        return np.array(batch_t), np.array(batch_x)

    def plot_data(self, batch_t=[], batch_x=[], explanation=[]):  # , batch_x, batch_y
        # batch_x= list(range(1 , 5))
        print(np.size(batch_x, 0))
        print(np.size(batch_x, 1))
        seq_length = np.size(batch_x, 1)
        if batch_t.size==0:
            batch_t = list(range(0, np.size(batch_x, 1)))
        plt.figure()
        for i in range(np.size(batch_x, 0)):
            plt.plot(batch_t, batch_x[i,:,0])
        plt.title('time- {}'.format(explanation))
        plt.ylabel('{}'.format(explanation))
        plt.xlabel('time')
        plt.show()

    def plot_test_prediction_data(self, batch_t=[], batch_y_test=[], batch_y_prediction=[],
                                  explanation=[]):  # , batch_x, batch_y
        """
        :param batch_t: time series (m x seq_length)
        :param batch_y_test: real output (Ty x m_test x n_y)
        :param batch_y_prediction:  predicted output (Ty x m_test x n_y)
        :param explanation: the figure's title
        :return: return void
        """
        Ty = np.size(batch_y_test, 0)
        # batch_x= list(range(1 , 5))
        # swap the axis to have the shape ( m_test , Ty , n_y)
        batch_y_prediction_reshaped = np.swapaxes(batch_y_prediction, 0, 1)
        batch_y_test_reshaped = np.swapaxes(batch_y_test, 0, 1)

        # print(np.size(batch_y_test_reshaped, 0))
        # print(np.size(batch_y_test_reshaped, 1))
        n_outputs = np.size(batch_y_test_reshaped, 2)
        seq_length = np.size(batch_y_test_reshaped, 1)

        if batch_t.size==0:
            batch_t_test = list(range(0, np.size(batch_y_test_reshaped, 1)))
        else:
            # cut to have (m x Ty) shape
            batch_t_test = batch_t

        for j in range(0, n_outputs, Ty):
            plt.figure()
            plt.title('time- {}'.format(explanation))
            for i in range(0, np.size(batch_y_test_reshaped, 0), 40):
                plt.plot(batch_t_test[:], batch_y_test_reshaped[i, :, j])
                plt.plot(batch_t_test[:], batch_y_prediction_reshaped[i, :, j], '--')
            plt.ylabel('output {}'.format(str(j)))
            plt.xlabel('time')
            plt.show()
        return

    def plot_data_adv(self, data=[], idx_time=0, idx_features=0, idx_no_samples=0 ):  # no_samples= m
        # n_outputs = np.size(data, idx_features)
        # time = list(range(0, np.size(data, idx_time)))
        #
        # # data_swapped : [idx_time, idx_no_samples, idx_features]
        # data_swapped = np.swapaxes(data, 0, idx_time)
        # if idx_no_samples == 0:
        #     idx_no_samples = idx_time
        # data_swapped = np.swapaxes(data_swapped, 1, idx_no_samples)
        #
        # for j in range(n_outputs):
        #     plt.figure()
        #     for i in range(0, np.size(data_swapped, 1)):
        #         plt.plot(time, data_swapped[:, i, j])
        #     plt.ylabel('output {}'.format(str(j)))
        #     plt.xlabel('time step')
        #     plt.show()
        return