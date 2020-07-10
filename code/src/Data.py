import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os
import io
from sklearn.preprocessing import MinMaxScaler






class Data:
    def __init__(self):
        return



    def generate_sequence_data(self, m=1000, seq_length=100, seed_number=0, data_type='linear'):
        """
        Generate data for test

        Arguments:
        m -- number of training example
        seq_length -- length of the sequence data

        Returns:
        generated_data -- sequence data
        """

        batch_x = []
        batch_dx = []
        batch_t = []
        a = 1.0
        b = 2.0
        random.seed(seed_number)
        for _ in range(m):
            rand = random.random() * 2 * math.pi
            # print('rand', rand)

            ## sine/cosine funcitons
            if data_type == 'sin':
                print('data_type: (1) ', data_type)
                timeSteps = np.linspace(0.0 * math.pi + rand, 3.0 * math.pi + rand, seq_length)
                x1 = np.sin(timeSteps)
                x2 = np.cos(timeSteps)

                dx1 = np.cos(timeSteps)
                dx2 = -np.sin(timeSteps)

                x = (a + 0.1 * rand) * x1 + (b + 0.1 * rand) * x2
                dx = (a + 0.1 * rand) * dx1 + (b + 0.1 * rand) * dx2

            # linear data
            elif data_type == 'linear':
                print('data_type: (2) ', data_type)
                timeSteps = np.linspace(0.0, 10.0, seq_length)
                x = (2 + 0.1 * rand) * timeSteps
                dx = (2 + 0.1 * rand) * (timeSteps * 0.0 + 1)

                # print('x1:', x1)
                # print('x2:', x2)
                # print('x1+x2', x1+x2)

                # batch_x.append(x_)
            elif data_type=='amplitude-modulation':
                print('data_type: (3) ', data_type)
                timeSteps= np.linspace(0.0, np.pi/2.0, seq_length)
                wm = 5.0
                wc = 4.0
                mu = 0.0
                x = []
                dx = []
                for i in range(0, seq_length):
                    t = timeSteps[i]
                    m_t = np.sin(wm * t)
                    d_m_t = wm * np.cos(wm * t)
                    f_t = m_t * np.sin(wc * t)
                    d_f_t = d_m_t * np.sin(wc * t) + m_t* wc* np.cos(wc*t)
                    sigma = np.sqrt(0.02 + 0.02 * (1 - m_t) ** 2.0)
                    sigma2 = np.sqrt((0.02 + 0.02 * (1 - m_t) ** 2.0)/2.0)
                    epsilon = np.random.normal(mu, sigma, 1)
                    epsilon2 = np.random.normal(mu, sigma2, 1)
                    y = f_t + epsilon
                    dy = d_f_t + epsilon2
                    x.append(y)
                    dx.append(dy)
                x = np.array(x)
                dx = np.array(dx)

                x = x.ravel()
                dx = dx.ravel()

            elif data_type =='amplitude-modulation-noNoise':
                print('data_type: (4) ', data_type)
                timeSteps= np.linspace(0.0, np.pi/2.0, seq_length)
                wm = 5.0
                wc = 4.0
                mu = 0.0
                x = []
                dx=[]
                for i in range(0, seq_length):
                    t = timeSteps[i]
                    m_t = np.sin(wm * t)
                    d_m_t = wm * np.cos(wm * t)
                    f_t = m_t * np.sin(wc * t)
                    d_f_t = d_m_t * np.sin(wc * t) + m_t* wc* np.cos(wc*t)
                    sigma = np.sqrt(0.02 + 0.02 * (1 - m_t) ** 2.0)
                    epsilon = np.random.normal(mu, sigma, 1)
                    y = f_t
                    dy = d_f_t
                    x.append(y)
                    dx.append(dy)
                x = np.array(x)
                dx = np.array(dx)

                x = x.ravel()
                dx = dx.ravel()

            batch_t.append(timeSteps)
            # if (data_type == 'sin') or (data_type =='linear'):
            batch_x.append(np.array([x, dx]).T)  # [x, dx]
            # else:
            #     batch_x.append(np.array([x]).T)  # [x, dx]
            # batch_dx.append((a + 0.1 * rand) * dx1 + (b + 0.1 * rand) * dx2)
        batch_x = np.array(batch_x)
        batch_dx = np.array(batch_dx)

        # shape: (batch_size, seq_length, feature_dim) ## feature_dim= output_dim
        return batch_t, batch_x

    def prepare_data(self, batch_data, Tx, Ty, Tx0= 0, Ty0=1,  x_feature=[], y_feature=[]):

        if not x_feature:
            batch_x = batch_data[:, range(Tx0,  Tx0+Tx), :]
        else:
            # batch_x = batch_data[:, range(Tx0, Tx0 + Tx), 0:2]
            batch_x = batch_data[:, range(Tx0, Tx0 + Tx), 0] #0:3
            # batch_x[:, :, -1] = 3.0
            # batch_x = np.reshape(batch_x, (np.size(batch_data, 0), Tx, np.size(batch_data, 2)))
        if not y_feature:
            batch_y = batch_data[:, range(Ty0, Ty0+Ty), :]
        else:
            batch_y = batch_data[:, range(Ty0, Ty0 + Ty), 1]  #2:3]

        # batch_y = np.reshape(batch_y, (np.size(batch_y, 1), np.size(batch_y, 0), np.size(batch_y, 2)))
        batch_y = np.swapaxes(batch_y, 0, 1)

        batch_x = np.reshape(batch_x, (np.size(batch_data, 0), Tx, np.size(x_feature, 0)) )
        batch_y = np.reshape(batch_y, (Ty, np.size(batch_data, 0), np.size(y_feature, 0)) )

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        # batch_x shape: (batch_size, Tx, nx)
        # batch_y shape: (Ty, batch_size, ny)
        return batch_x, batch_y

    def load_data(self):
        return

    def plot_data(self, batch_t=[], batch_x=[], explanation=[]):  # , batch_x, batch_y
        # batch_x= list(range(1 , 5))
        print(np.size(batch_x, 0))
        print(np.size(batch_x, 1))
        seq_length = np.size(batch_x, 1)
        if batch_t == []:
            batch_t = list(range(1, np.size(batch_x, 1)))
        plt.figure()
        for i in range(np.size(batch_x, 0)):
            plt.plot(batch_t[i], batch_x[i])
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
        # batch_x= list(range(1 , 5))
        # swap the axis to have the shape ( m_test , Ty , n_y)
        batch_y_prediction_reshaped = np.swapaxes(batch_y_prediction, 0, 1)
        batch_y_test_reshaped = np.swapaxes(batch_y_test, 0, 1)

        # print(np.size(batch_y_test_reshaped, 0))
        # print(np.size(batch_y_test_reshaped, 1))
        n_outputs = np.size(batch_y_test_reshaped, 2)
        seq_length = np.size(batch_y_test_reshaped, 1)

        if batch_t == []:
            batch_t_test = list(range(1, np.size(batch_y_test_reshaped, 1)))
        else:
            # cut to have (m x Ty) shape
            batch_t_test = batch_t

        for j in range(n_outputs):
            plt.figure()
            plt.title('time- {}'.format(explanation))
            for i in range(np.size(batch_y_test_reshaped, 0)):
                plt.plot(batch_t_test[i], batch_y_test_reshaped[i, :, j])
                plt.plot(batch_t_test[i], batch_y_prediction_reshaped[i, :, j], '--')
            plt.ylabel('output {}'.format(str(j)))
            plt.xlabel('time')
            plt.show()

    def plot_data_adv(self, data=[], idx_time=0, idx_features=0, idx_no_samples=0 ):  # no_samples= m
        n_outputs = np.size(data, idx_features)
        time = list(range(0, np.size(data, idx_time)))

        # data_swapped : [idx_time, idx_no_samples, idx_features]
        data_swapped = np.swapaxes(data, 0, idx_time)
        if idx_no_samples == 0:
            idx_no_samples = idx_time
        data_swapped = np.swapaxes(data_swapped, 1, idx_no_samples)

        for j in range(n_outputs):
            plt.figure()
            for i in range(0, np.size(data_swapped, 1)):
                plt.plot(time, data_swapped[:, i, j])
            plt.ylabel('output {}'.format(str(j)))
            plt.xlabel('time step')
            plt.show()

    '''
        For the given path, get the List of all files in the directory tree
        source: https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/ 
    '''
    def getListOfFiles(self, dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)

        return allFiles

    def read_from_file(self, file_path):
        print('read_from_file: file_path', file_path)
        files = self.getListOfFiles(file_path)
        # files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        print('files: ', files)
        data_raw = []
        for file in files:
            # file_name = os.join(file_path, file)
            print('file: ', file)
            data_dict = {}
            input_file = io.open(file, 'r')
            # lines = input_file.readlines()
            count = 0
            keys_list = []
            while True:
                count += 1

                # Get next line from file
                line = input_file.readline()

                # if line is empty
                # end of file is reached
                if not line:
                    break

                line_elements = line.split()
                i = 0
                for element in line_elements:
                    if count == 1:
                        data_dict[element] = []
                        keys_list.append(element)
                    else:
                        data_dict[keys_list[i]].append(float(element))
                    i += 1
            input_file.close()
            data_raw.append(data_dict)
        print('key list: ', keys_list)
        return data_raw

    def prepare_data_batches(self, feature_list, data_raw, seq_length):
        batch_t = []
        batch_x = []
        # if abs(data['rssiinCollision'])(count)) > 0.01:
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

    def update_bracelet_data(self, data_raw, problem_type):

        for data in data_raw:
            updating_indices = []
            for i in range( len(data['rssiinCollision']) ):
                if abs( (data['rssiinCollision'])[i] ) < 0.01 :
                    updating_indices.append(i)
                else:
                    # update the range of rssi values brings close to 1.
                    (data['rssiinCollision'])[i]= (data['rssiinCollision'])[i] / 50.0

            # remove the lines of the data with zero rssi, start from the last index to the first of the list
            updating_indices.reverse()
            for index in updating_indices:
                for key in data:
                    (data[key]).pop(index)



            ## if there is the time part

            # update the time feature with the time difference instead of absolute timing
            # time_list = data['time_epoch_braclet']# copy by reference
            # for i in reversed(range(1, len(time_list))):
            #     time_list[i] = abs(time_list[i] - time_list[i-1])/0.05
            #
            # # remove also the first time we get time value
            # for key in data:
            #     (data[key]).pop(0)


            if problem_type =='classification':
                # try the classification problem
                distnace_threshold = 1.5
                for i in range(len(data['braceletsdistance'])):
                    if (data['braceletsdistance'])[i] < distnace_threshold:
                        (data['braceletsdistance'])[i] = 1
                    else:
                        (data['braceletsdistance'])[i] = 0

        return data_raw

    def plot_bracelet_predictions(self, batch_y_test=[], prediction=[],
                                  explanation=[]):  # , batch_x, batch_y
        """
        :param batch_t: time series (m x seq_length)
        :param batch_y_test: real output (Ty x m_test x n_y)
        :param prediction:  predicted output (Ty x m_test x n_y)
        :param explanation: the figure's title
        :return: return void
        """
        # batch_x= list(range(1 , 5))
        # swap the axis to have the shape ( m_test , Ty , n_y)
        batch_y_prediction_reshaped = np.swapaxes(prediction, 0, 1)
        batch_y_test_reshaped = np.swapaxes(batch_y_test, 0, 1)

        if not explanation:
            explanation = ' '

        # print(np.size(batch_y_test_reshaped, 0))
        # print(np.size(batch_y_test_reshaped, 1))
        n_outputs = np.size(batch_y_test_reshaped, 2)

        batch_t_test = list(range(0, np.size(batch_y_test_reshaped, 0)))

        # predThreshold = 0.5
        # predMap = lambda x: 0 if x <= predThreshold else 1

        # for j in range(n_outputs):
        #     plt.figure()
        #     plt.title('batches_sample- {}'.format(explanation))
        #     for i in range(np.size(batch_y_test_reshaped, 0)):
        #         pred = predMap(batch_y_prediction_reshaped[i, :, j])
        #         # print(j, ' ', i, ' ', np.size(batch_t_test, axis=0), ' ', np.size(batch_y_test_reshaped, axis=0), ' ',
        #         #       np.size(batch_y_test_reshaped, axis=2))
        #         plt.plot(batch_t_test[i], batch_y_test_reshaped[i, :, j], 'go')
        #         plt.plot(batch_t_test[i], pred, 'ro')
        #     plt.ylabel('output {}'.format(str(j)))
        #     plt.xlabel('batches')
        #     plt.show()

        for j in range(n_outputs):
            plt.figure()
            plt.title('batches_sample- {}'.format(explanation))
            # pred =[]
            # for i in range(np.size(batch_y_test_reshaped, 0)):
            #     pred.append( predMap(batch_y_prediction_reshaped[i, :, j]))
            # print(j, ' ', i, ' ', np.size(batch_t_test, axis=0), ' ', np.size(batch_y_test_reshaped, axis=0), ' ',
            #       np.size(batch_y_test_reshaped, axis=2))
            plt.plot(batch_t_test, batch_y_test_reshaped[:, :, j], 'go', markersize=10)
            plt.plot(batch_t_test, batch_y_prediction_reshaped[:, :, j], 'r*', markersize=8)
            plt.ylabel('output {}'.format(str(j)))
            plt.xlabel('batches')
            plt.show()

        return


    def bracelet_data_augmentation(self, data_raw):
        data_raw_reversred=[]
        for data in data_raw:
            reversed_data = {}
            for key in data:
                reversed_data[key] = (data[key])[::-1]
            data_raw_reversred.append(reversed_data)

        for reversed_data in data_raw_reversred:
            data_raw.append(reversed_data)
        return

