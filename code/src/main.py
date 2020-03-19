from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from RnnKeras import RnnKeras, PlotLosses
from Data import Data
import datetime
import array as arr

if __name__ == '__main__':

    now = datetime.datetime.now()
    year, month, day, hour, minute, second = now.year, now.month, now.day, now.hour, now.minute, now.second
    time_now_string = '_{}_{}_{}_{}_{}_{}'.format(year, month, day, hour, minute, second)
#################################
### STEP: Define Hyper-parameters
#################################
    seq_length =100 #20 100
    Tx = seq_length-1
    Ty = Tx
    n_a = 32 # 5 32
    n_y = 2
    n_x = 2
    m_train = 200 # 40 200
    m_val = 20 #2 20
    m_test = 5 #1 5
    epochs = 50 #20 50
    model_name = 'model'
    models_path = 'models' + time_now_string
    doTraining = True
    seed_number = 0
    ## use these for regression problem
    loss_function = 'mean_squared_error'
    model_metrics = ['mse']
    data_type ='sin'
    verbosity = False
    ## for classification problem use other methods

#################################
### STEP: DATA
#################################
    data = Data()
    # ---> Generate Data
    # Training set
    batch_t_train, batch_data_train = data.generate_sequence_data(m=m_train, seq_length=seq_length,
                                                                  seed_number=seed_number, data_type=data_type)
    batch_x_train, batch_y_train = data.prepare_data(batch_data_train)
    # validation set
    batch_t_val, batch_data_val = data.generate_sequence_data(m=m_val, seq_length=seq_length,
                                                              seed_number=seed_number + 1, data_type=data_type)
    batch_x_val, batch_y_val = data.prepare_data(batch_data_val)
    # Test Set
    batch_t_test, batch_data_test = data.generate_sequence_data(m=m_test, seq_length=seq_length,
                                                                seed_number=seed_number+2, data_type=data_type)
    batch_x_test, batch_y_test = data.prepare_data(batch_data_test)

    # --> Print & Plot Data
    if verbosity:
        # training set
        print('batch_data_training shape', batch_data_train.shape)
        print('x_train shape: ', batch_x_train.shape)
        print('y_train shape: ', batch_y_train.shape)
        data.plot_data(batch_t_train, batch_data_train, 'x:train')
        # validation set
        print('batch_data_val shape', batch_data_val.shape)
        print('x_val shape: ', batch_x_val.shape)
        print('y_val shape: ', batch_y_val.shape)
        data.plot_data(batch_t_val, batch_data_val, 'x:val')
        # test set
        print('batch_data_test shape', batch_data_test.shape)
        print('x_test shape: ', batch_x_test.shape)
        print('y_test shape: ', batch_y_test.shape)
        data.plot_data(batch_t_test, batch_data_test, 'x:test')

#################################
### STEP: TRAINING RNN
#################################

    plot_losses = PlotLosses(file_path=models_path, file_name=model_name)
    rnn = RnnKeras(n_a=n_a, n_y=n_y, n_x=n_x, Tx=Tx, Ty=Ty, m_train=m_train, m_val=m_val, m_test=m_test)

    if doTraining:
        rnn.create_model()
        rnn.create_optimizer()
        rnn.compile_model(loss_function, model_metrics)
        history = rnn.fit_model(x_train=batch_x_train, y_train=batch_y_train, x_val=batch_x_val, y_val=batch_y_val,
                                epochs=epochs, plot_loss_value_obj=plot_losses, verbosity=verbosity)
        rnn.save_model(models_path, model_name)
        rnn.visualize_model(models_path, model_name)
    else:
        rnn.load_model(models_path, model_name)

    if verbosity:
        rnn.provide_model_summary()

        print(history.history.keys())
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train & validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='upper right')
        plt.show()

#################################
### STEP: PREDICTION
#################################

    prediction = rnn.compute_prediction(batch_x_test)
    rmse_test = rnn.evaluate_prediction(batch_y_test, np.array(prediction))
    print('RMSE of the test: ', rmse_test)

    # change the type of value from list to array
    prediction = np.array(prediction)
    batch_t_test = np.array(batch_t_test)

    data.plot_test_prediction_data(batch_t_test, batch_y_test, prediction,
                                   'test (line), prediction (dashed lines)')

    if verbosity:
        y_test = np.reshape(batch_y_test[:, 0, :], (1, Ty, n_y))
        prediction_reshaped = np.swapaxes(prediction, 0, 1)
        data.plot_data(batch_t_test[:, 1:], prediction_reshaped, 'prediction', Tx)
        data.plot_data(batch_t_test[:, 1:], y_test,         'y',      Tx)
