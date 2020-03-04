from __future__ import print_function

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np


# return training data
def get_train():
    seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
    seq = np.array(seq)
    X, y = seq[:, 0], seq[:, 1]
    X = X.reshape((len(X), 1, 1))
    return X, y


# return validation data
def get_val():
    seq = [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]]
    seq = np.array(seq)
    X, y = seq[:, 0], seq[:, 1]
    X = X.reshape((len(X), 1, 1))
    return X, y


# def main():
if __name__ == '__main__':

    # define model
    model = Sequential()
    model.add(LSTM(10, input_shape=(1,1)))
    model.add(Dense(1, activation='linear'))
    # compile model
    model.compile(loss='mse', optimizer='adam')
    # fit model
    X,y = get_train()
    valX, valY = get_val()
    history = model.fit(X, y, epochs=1000, validation_data=(valX, valY), shuffle=False)
    # plot train and validation loss
    # plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    valY_predicted= model.predict(valX)
    # plt.figure()
    # plt.plot(valX)
    # plt.title('valX')
    # plt.show()
    # plt.figure()
    valX= np.reshape(valX,(5,1))
    valY = np.reshape(valY, (5, 1))
    valY_predicted = np.reshape(valY_predicted, (5, 1))
    print(valX.shape)
    print(valY.shape)
    print(valY_predicted.shape)
    plt.figure()
    plt.plot(valX, valY)
    plt.plot(valX, valY_predicted)
    plt.title('valY and valY_predicted')
    plt.legend(['Y', 'Y_predicted'], loc='upper right')
    plt.show()


# if __name__ == '__main__':
#     main()



