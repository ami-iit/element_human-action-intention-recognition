from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras import backend as K

print(tf.__version__)


class TrainRNN:
    def __init__(self, n_a, n_y, n_x, Tx, m):

        # hidden state dimensions of each RNN/LSTM cell
        self.n_a = 64
        # output dimensions of each output layer
        self.n_y = 78
        # input dimensions of input layer
        self.n_x = 78
        # time series (sequence) length, the time horizon of prediction
        self.Tx = 10
        # number of training set
        self.m = 10


        # TODO: Training

        #self.learning_rate = 0.0025
        #self.lambda_loss_amount = 0.0015
        #self.training_iters = training_data_count * 300  # Loop 300 times on the dataset
        #self.batch_size = 1500
        #self.display_iter = 30000  # To show test set accuracy during training
        #self.time = 0.0


        ## utilities
        self.reshapor = Reshape((1, self.n_y))
        self.LSTM_cell = LSTM(self.n_a, return_state = True)
        self.densor = Dense(self.n_y, activation='softmax')

    def model(self):
        """
        Implement the model

        Arguments:
        Tx -- length of the sequence in a corpus
        n_a -- the number of activations used in our model
        n_values -- number of unique values in the music data

        Returns:
        model -- a keras instance model with n_a activations
        """

        # Define the input layer and specify the shape
        X = Input(shape=(self.Tx, self.n_y))

        # Define the initial hidden state a0 and initial cell state c0
        # using `Input`
        a0 = Input(shape=(self.n_a,), name='a0')
        c0 = Input(shape=(self.n_a,), name='c0')
        a = a0
        c = c0
        # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
        outputs = []

        # Step 2: Loop
        for t in range(self.Tx):
            # Step 2.A: select the "t"th time step vector from X.
            x = Lambda(lambda X: X[:, t, :])(X)
            # Step 2.B: Use reshapor to reshape x to be (1, n_y) (≈1 line)
            x = self.reshapor(x)
            # Step 2.C: Perform one step of the LSTM_cell
            a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])
            # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
            out = self.densor(a)
            # Step 2.E: add the output to "outputs"
            outputs = outputs + [out]
            # outputs=(ouputs+out)

        # Step 3: Create model instance
        model = Model(inputs=[X, a0, c0], outputs=outputs)
        return model

    def load_data(self, path):
        return

    def visualize(self):
        return



if __name__ == '__main__':
    n_a = 64
    n_y = 78
    n_x = 78
    Tx = 10
    m = 10

    rnn = TrainRNN(n_a=n_a, n_y=n_y, n_x=n_x, Tx=Tx, m=m)
    model = rnn.model()
    model.summary()
    print("1-")
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print("2-")
    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    model.fit([X, a0, c0], list(Y), epochs=100)

