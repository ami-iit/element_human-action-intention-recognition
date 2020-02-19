from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from TrainRNN import TrainRNN
from Data import Data
import array as arr



if __name__ == '__main__':

#################################
### STEP: Define Hyper-parameters
#################################
    seq_length = 20
    Tx = seq_length-1
    Ty=Tx
    n_a = 10
    n_y = 2
    n_x = 2
    m = 30
    epochs = 1000

#################################
### STEP: DATA
#################################
    data = Data()
    batch_t, batch_data = data.generate_sequence_data(m=m, seq_length=seq_length) # here m is the number of data sets
    # print(batch_x[0].shape)
    # print(batch_x[0])
    batch_x, batch_y = data.prepare_data(batch_data)
    print('batch_data shape', batch_data.shape)
    print('x shape: ', batch_x.shape)
    print('y shape: ', batch_y.shape)
    # print('batch_data:', batch_data)
    # print('batch_x:', batch_x)
    # print('batch_y:', batch_y)

    data.plot_data(batch_t, batch_data, 'x')
    # data.plot_data(batch_t, batch_dx, 'dx')

    # batch_y = batch_x([])
#################################
### TRAINING RNN
#################################
    rnn = TrainRNN(n_a=n_a, n_y=n_y, n_x=n_x, Tx=Tx, m=m)
    model = rnn.create_model()
    # model.summary()
    optimizer = rnn.create_optimizer()
    model = rnn.compile_model(model, optimizer)

    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    model, history = rnn.fit_model(model, batch_x, batch_y, a0, c0, epochs)
# list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['dense_accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['dense_accuracy'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['dense_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'dense-loss'], loc='upper left')
    plt.show()




#    m = 60
#    a0 = np.zeros((m, n_a))
#    c0 = np.zeros((m, n_a))

