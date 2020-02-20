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
    seq_length = 50
    Tx = seq_length-1
    Ty = Tx
    n_a = 10
    n_y = 2
    n_x = 2
    m = 30
    epochs = 2000
    model_name = 'myModel'
    models_path = 'models'
    doTraining= True
    seed_number = 0

#################################
### STEP: DATA
#################################
    data = Data()
    batch_t, batch_data = data.generate_sequence_data(m=m, seq_length=seq_length,  seed_number=seed_number) # here m is the number of data sets
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
    rnn = TrainRNN(n_a=n_a, n_y=n_y, n_x=n_x, Tx=Tx, m=m, Ty=Ty)
    if doTraining == True:

        model = rnn.create_model()
        # model.summary()
        optimizer = rnn.create_optimizer()
        model = rnn.compile_model(model, optimizer)

        a0 = np.zeros((m, n_a))
        c0 = np.zeros((m, n_a))
        model, history = rnn.fit_model(model, batch_x, batch_y, a0, c0, epochs)
        rnn.save_model(model, models_path, model_name)
        # model=rnn.load_model(models_path, model_name)
        rnn.visualize_model(model, models_path, model_name)
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
    else:
        pass
        # model = rnn.load_model(models_path, model_name)

############

    inference_model = rnn.inference_model()
    inference_model.summary()
    x_initializer = np.zeros((1, 1, n_x))
    batch_t, batch_data_test = data.generate_sequence_data(m=1, seq_length=seq_length, seed_number=5)  # here m is the number of data sets
    batch_x_test, batch_y_test = data.prepare_data(batch_data_test)
    x_test = np.reshape(batch_x_test[0, 0, :], (1, 1, n_x))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))

    print('x_test: ', type(x_test), x_test.shape, x_test )
    print('x_initializer: ', type(x_initializer), x_initializer.shape, x_initializer)

    prediction = rnn.predict_motion(inference_model, x_test, a_initializer, c_initializer)

    # prediction = rnn.predict_motion_new(model, x_test, a_initializer, c_initializer)

    print('prediction type: ', type(prediction))
    prediction = np.array(prediction)
    prediction= np.reshape(prediction,(1, Ty, n_y))
    # batch_y_test = np.swapaxes(batch_y_test[:, 0, :], 0, 0)

    batch_t=np.array(batch_t)

    print('prediction shape: ', type(prediction), prediction.shape)
    print('prediction : ', prediction)
    y_test = np.reshape(batch_y_test[:, 0, :], (1, Ty, n_y))
    print('y shape: ', type(y_test), y_test.shape)
    print('y : ', y_test)
    print('batch_t ',     (batch_t).shape, batch_t)
    print('batch_t[1:] ', (batch_t[:, 1:]).shape, batch_t[1:])
    print('batch_t[2:] ', (batch_t[:, 2:]).shape, batch_t[2:])
    data.plot_data(batch_t[:,1:], prediction, 'prediction', Tx)
    data.plot_data(batch_t[:,1:], y_test,         'y',          Tx)



#    m = 60
#    a0 = np.zeros((m, n_a))
#    c0 = np.zeros((m, n_a))

