from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from RnnKeras import RnnKeras, PlotLosses
from Data import Data
import array as arr

if __name__ == '__main__':

#################################
### STEP: Define Hyper-parameters
#################################
    seq_length = 100
    Tx = seq_length-1
    Ty = Tx
    n_a = 32
    n_y = 2
    n_x = 2
    m = 200
    m_val = 20
    m_test = 1
    epochs = 50
    training_model_name = 'train_myModelNew'
    inference_model_name = 'inference_myModel'
    models_path = 'models'
    doTraining = False
    seed_number = 0
    ## use these for regression problem
    loss_function = 'mean_squared_error'
    model_metrics = ['mse']
    data_type ='sin'
    ## for classification problem use other methods

#################################
### STEP: DATA
#################################
    data = Data()
    ## Training set
    batch_t_train, batch_data_train = data.generate_sequence_data(m=m, seq_length=seq_length, seed_number=seed_number, data_type=data_type) # here m is the number of data sets
    # print(batch_x[0].shape)
    # print(batch_x[0])
    batch_x_train, batch_y_train = data.prepare_data(batch_data_train)
    print('batch_data_training shape', batch_data_train.shape)
    print('x_train shape: ', batch_x_train.shape)
    print('y_train shape: ', batch_y_train.shape)
    # print('batch_data:', batch_data)
    # print('batch_x:', batch_x)
    # print('batch_y:', batch_y)

    data.plot_data(batch_t_train, batch_data_train, 'x:train')
    # data.plot_data(batch_t, batch_dx, 'dx')

    ## validation set
    batch_t_val, batch_data_val = data.generate_sequence_data(m=m_val, seq_length=seq_length, seed_number=seed_number + 1, data_type=data_type)
    batch_x_val, batch_y_val = data.prepare_data(batch_data_val)
    print('batch_data_val shape', batch_data_val.shape)
    print('x_val shape: ', batch_x_val.shape)
    print('y_val shape: ', batch_y_val.shape)
    data.plot_data(batch_t_val, batch_data_val, 'x:val')

#################################
### TRAINING RNN
#################################

    plot_losses = PlotLosses()
    rnn = RnnKeras(n_a=n_a, n_y=n_y, n_x=n_x, Tx=Tx, m=m, Ty=Ty)
    if doTraining == True:

        # model = rnn.create_model()
        rnn.create_model()
        # model.summary()
        rnn.create_optimizer()
        rnn.compile_model(loss_function, model_metrics)

        a0 = np.zeros((m, n_a))
        c0 = np.zeros((m, n_a))
        a0val = np.zeros((m_val, n_a))
        c0val = np.zeros((m_val, n_a))

        history = rnn.fit_model(Xtrain=batch_x_train, Ytrain=batch_y_train, a0=a0, c0=c0, Xval=[batch_x_val, a0val, c0val], Yval=batch_y_val, epochs=epochs, plot_loss_value_obj=plot_losses)
        rnn.save_model(models_path, training_model_name)
        rnn.visualize_model(models_path, training_model_name)
        # list all data in history
        print(history.history.keys())

        # summarize history for accuracy
        # plt.figure()
        # plt.plot(history.history['dense_accuracy'])
        # # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['dense_accuracy'], loc='upper left')
        # plt.show()
        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train & validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='upper right')
        plt.show()
    else:
        # pass
        rnn.load_model(models_path, training_model_name)

############
### PREDICTION
############

    # inference_model = model
    # inference_model = rnn.create_model()
    # inference_model.summary()
    rnn.provide_model_summary()
    # rnn.visualize_model(model, models_path, inference_model_name)

    x_initializer = np.zeros((1, 1, n_x))
    batch_t_test, batch_data_test = data.generate_sequence_data(m=m_test, seq_length=seq_length, seed_number=5, data_type=data_type)  # here m is the number of data sets
    batch_x_test, batch_y_test = data.prepare_data(batch_data_test)
    # x_test = np.reshape(batch_x_test[0, 0, :], (1, 1, n_x))

    # batch_x_test, batch_y_test = data.prepare_data(batch_data_test)
    a_initializer = np.zeros((m_test, n_a))
    c_initializer = np.zeros((m_test, n_a))

    # print('x_test: ', type(x_test), x_test.shape, x_test )
    # print('x_initializer: ', type(x_initializer), x_initializer.shape, x_initializer)

    prediction = rnn.compute_prediction(batch_x_test, a_initializer, c_initializer)
    rmse_test = rnn.evaluate_prediction(batch_y_test, np.array(prediction))
    print('RMSE of the test: ', rmse_test)

    # prediction = rnn.predict_motion_new(model, x_test, a_initializer, c_initializer)

    print('prediction type: ', type(prediction))
    prediction = np.array(prediction)
    prediction_reshaped = np.swapaxes(prediction, 0, 1)
    batch_y_test_reshaped = np.swapaxes(batch_y_test, 0, 1)

# batch_y_test = np.swapaxes(batch_y_test[:, 0, :], 0, 0)

    batch_t_test = np.array(batch_t_test)

    print('prediction shape: ', type(prediction_reshaped), prediction_reshaped.shape)
    # print('prediction : ', prediction_reshaped)
    y_test = np.reshape(batch_y_test[:, 0, :], (1, Ty, n_y))
    print('y shape: ', type(y_test), y_test.shape)
    # print('y : ', y_test)
    # print('batch_t ',     (batch_t_test).shape, batch_t_test)
    # print('batch_t[1:] ', (batch_t_test[:, 1:]).shape, batch_t_test[1:])
    # print('batch_t[2:] ', (batch_t_test[:, 2:]).shape, batch_t_test[2:])
    data.plot_data(batch_t_test[:, 1:], prediction_reshaped, 'prediction', Tx)
    data.plot_data(batch_t_test[:, 1:], y_test,         'y',      Tx)
    data.plot_test_prediction_data(batch_t_test[:,1:], batch_y_test_reshaped, prediction_reshaped, 'test (line) prediction (dashed lines) ', Tx)

#    m = 60
#    a0 = np.zeros((m, n_a))
#    c0 = np.zeros((m, n_a))
