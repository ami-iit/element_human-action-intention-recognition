import numpy as np
import math
import random
import matplotlib.pyplot as plt



class Data:
    def __init__(self):
        return
    def generate_sequence_data(self, m =1000, seq_length=100, seed_number=0):
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
            #print('rand', rand)
            timeSteps = np.linspace(0.0 * math.pi + rand, 3.0 * math.pi + rand, seq_length );
            x1 = np.sin(timeSteps)
            x2 = np.cos(timeSteps)

            dx1 = np.cos(timeSteps)
            dx2 = -np.sin(timeSteps)
            # print('x1:', x1)
            # print('x2:', x2)
            # print('x1+x2', x1+x2)

            # batch_x.append(x_)
            batch_t.append(timeSteps)
            batch_x.append(np.array([(a + 0.1 * rand) * x1+ (b + 0.1 * rand) * x2, (a + 0.1 * rand) * dx1 + (b + 0.1 * rand) * dx2]).T )
            # batch_dx.append((a + 0.1 * rand) * dx1 + (b + 0.1 * rand) * dx2)
        batch_x = np.array(batch_x)
        batch_dx = np.array(batch_dx)

        # shape: (batch_size, seq_length, output_dim)
        return batch_t, batch_x

    def prepare_data(self , batch_data):
        seq_length=np.size(batch_data, 1)
        print('seq_length:',seq_length)
        batch_x = batch_data[:, range(0, seq_length - 1), :]
        batch_y = batch_data[:, range(1, seq_length), :]
        # batch_y = np.reshape(batch_y, (np.size(batch_y, 1), np.size(batch_y, 0), np.size(batch_y, 2)))
        batch_y = np.swapaxes(batch_y, 0, 1)

        return batch_x, batch_y

    def load_model(self):
        return


    def plot_data(self, batch_t=[], batch_x=[], explenation=[], Tx=0): #, batch_x, batch_y
        # batch_x= list(range(1 , 5))
        print(np.size(batch_x, 0))
        print(np.size(batch_x, 1))
        seq_length = np.size(batch_x, 1)
        if batch_t==[]:
            batch_t = list(range(1, Tx+1))
        plt.figure()
        for i in range(np.size(batch_x , 0)):
            plt.plot(batch_t[i], batch_x[i])
        plt.title('time- {}'.format(explenation))
        plt.ylabel('{}'.format(explenation))
        plt.xlabel('time')
        plt.show()
