from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K


n_a = 64
n_values = 78

reshapor = Reshape((1, n_values))
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_values, activation='softmax')

class RNN:
    def __init__(self):

        self.time=0.0
    def modelRNN(self, Tx, n_a, n_values):
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
        X = Input(shape=(Tx, n_values))

        # Define the initial hidden state a0 and initial cell state c0
        # using `Input`
        a0 = Input(shape=(n_a,), name='a0')
        c0 = Input(shape=(n_a,), name='c0')
        a = a0
        c = c0
        # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
        outputs = []

        # Step 2: Loop
        for t in range(Tx):
            # Step 2.A: select the "t"th time step vector from X.
            x = Lambda(lambda X: X[:, t, :])(X)
            # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
            x = reshapor(x)
            # Step 2.C: Perform one step of the LSTM_cell
            a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
            # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
            out = densor(a)
            # Step 2.E: add the output to "outputs"
            outputs = outputs + [out]
            # outputs=(ouputs+out)

        # Step 3: Create model instance
        model = Model(inputs=[X, a0, c0], outputs=outputs)

        return model


if __name__ == '__main__':
    rnn=RNN()
    model=rnn.modelRNN(Tx = 30 , n_a = 64, n_values = 78)
    model.summary()
    print("hello")

