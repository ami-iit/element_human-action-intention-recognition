from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from RnnKeras import RnnKeras, PlotLosses, Uncertainty
from DatasetHumanMotionPrediction import DatasetHumanMotionPrediction
import datetime
import array as arr

if __name__ == '__main__':

    now = datetime.datetime.now()
    year, month, day, hour, minute, second = now.year, now.month, now.day, now.hour, now.minute, now.second
    time_now_string = '_{}_{}_{}_{}_{}_{}'.format(year, month, day, hour, minute, second)
    #################################
    #   STEP: Define Hyper-parameters
    #################################
    seq_length = 100  # 20 100
    Tx = 10
    Tx0 = 0  # this is used to prepare the data, not a part of rnn
    Ty = 80
    Ty0 = 10  # this is used to prepare the data, not a part of rnn
    n_a = [64, 64]#[3, 2]#[32, 16]  # 5 32
    n_y = 144
    n_x = 144
    m_train = 200  # 40 200
    m_val = 100  # 2 20
    m_test = 1  # 1 5
    epochs = 30 #100  # 20 50
    model_name = 'model'
    models_path = 'models/models' + time_now_string
    # models_path = 'models/models_2020_3_26_19_10_54'# sin signal learned
    # models_path = 'models/models_2020_4_14_19_46_56'  # amplitude-modulation signal learned
    doTraining = True
    learnUncertainty = False
    predictFuture = True
    seed_number = 0
    # use these for regression problem
    loss_function = 'mean_squared_error'
    model_metrics = ['mse']
    data_type = 'amplitude-modulation'
    verbosity = False
    read_data_from_file = True
    feature_list = []


    # for classification problem use other methods

    #################################
    #   STEP: DATA
    #################################
    data = DatasetHumanMotionPrediction()
    if read_data_from_file:
        # ---> Read Data from File
        # Training set
        data_training = data.read_data_in_directory('../dataset/HumanActionIntentionPrediction/train_val_test_Data/training')
        batch_t_train, batch_data_train = data.prepare_data_batches(feature_list, data_training, seq_length)
        batch_x_train, batch_y_train = data.prepare_data(batch_data_train, Tx, Ty, Tx0, Ty0)

        # Validation set
        data_validation = data.read_data_in_directory('../dataset/HumanActionIntentionPrediction/train_val_test_Data/validation')
        batch_t_validation, batch_data_val = data.prepare_data_batches(feature_list, data_validation, seq_length)
        batch_x_val, batch_y_val = data.prepare_data(batch_data_val, Tx, Ty, Tx0, Ty0)

        # Test set
        data_test = data.read_data_in_directory('../dataset/HumanActionIntentionPrediction/train_val_test_Data/test')
        batch_t_test, batch_data_test = data.prepare_data_batches(feature_list, data_test, seq_length)
        batch_x_test, batch_y_test = data.prepare_data(batch_data_test, Tx, Ty, Tx0, Ty0)

        m_train = batch_x_train.shape[0]
        m_val = batch_x_val.shape[0]
        m_test = batch_x_test.shape[0]

    else:

        # ---> Generate Data
        # Training set
        batch_t_train, batch_data_train = data.generate_dataset(m=m_train, seq_length=seq_length,
                                                                  seed_number=seed_number, data_type=data_type)
        batch_x_train, batch_y_train = data.prepare_data(batch_data_train, Tx, Ty, Tx0, Ty0)
        # validation set
        batch_t_val, batch_data_val = data.generate_dataset(m=m_val, seq_length=seq_length,
                                                              seed_number=seed_number + 1, data_type=data_type)
        batch_x_val, batch_y_val = data.prepare_data(batch_data_val, Tx, Ty, Tx0, Ty0)
        # Test Set
        batch_t_test, batch_data_test = data.generate_dataset(m=m_test, seq_length=seq_length,
                                                                seed_number=seed_number + 2, data_type=data_type)
        batch_x_test, batch_y_test = data.prepare_data(batch_data_test, Tx, Ty, Tx0, Ty0)

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

    print('---> data preparation is done ...')

    #################################
    #   STEP: TRAINING RNN
    #################################
    print('---> Start rnn ...')

    plot_losses = PlotLosses(file_path=models_path, file_name=model_name)
    rnn = RnnKeras(n_a=n_a, n_y=n_y, n_x=n_x, Tx=Tx, Ty=Ty, m_train=m_train, m_val=m_val, m_test=m_test)

    if doTraining:
        print('---> Start training ...')
        rnn.create_model()
        rnn.create_optimizer()
        rnn.compile_model(loss_function, model_metrics)
        history = rnn.fit_model(x_train=batch_x_train, y_train=batch_y_train, x_val=batch_x_val, y_val=batch_y_val,
                                epochs=epochs, plot_loss_value_obj=plot_losses, verbosity=verbosity)
        rnn.save_model(models_path, model_name)
        rnn.visualize_model(models_path, model_name)
    else:
        print('---> Start loading model ...')
        rnn.load_model(models_path, model_name)

    print('---> model is trained or loaded ...')

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
    #   STEP: PREDICTION
    #################################
    if predictFuture:
        prediction = rnn.compute_prediction(batch_x_test)
        prediction = np.reshape(prediction, (Ty, m_test, n_y))  # in case  prediction does not have correct shaping
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
        data.plot_data(batch_t_test[:, 1:], prediction_reshaped, 'prediction')
        data.plot_data(batch_t_test[:, 1:], y_test, 'y')

    if learnUncertainty:
        uncertainty = Uncertainty()
        ##
        # Compute the data sets for uncertainty prediction
        ##
        # ## perfect output ##
        # # training set
        # batch_t_train_noNoise, batch_data_train_noNoise = data.generate_sequence_data(m=m_train, seq_length=seq_length,
        #                                                                               seed_number=seed_number,
        #                                                                               data_type='amplitude-modulation-noNoise')
        # batch_x_train_noNoise, batch_y_train_noNoise = data.prepare_data(batch_data_train_noNoise, Tx, Ty, Tx0, Ty0)
        # # validation set
        # batch_t_val_noNoise, batch_data_val_noNoise = data.generate_sequence_data(m=m_val, seq_length=seq_length,
        #                                                                           seed_number=seed_number + 1,
        #                                                                           data_type='amplitude-modulation-noNoise')
        # batch_x_val_noNoise, batch_y_val_noNoise = data.prepare_data(batch_data_val_noNoise, Tx, Ty, Tx0, Ty0)
        # # Test Set
        # batch_t_test_noNoise, batch_data_test_noNoise = data.generate_sequence_data(m=m_test, seq_length=seq_length,
        #                                                                             seed_number=seed_number + 2,
        #                                                                             data_type='amplitude-modulation-noNoise')
        # batch_x_test_noNoise, batch_y_test_noNoise = data.prepare_data(batch_data_test_noNoise, Tx, Ty, Tx0, Ty0)
        #
        # batch_y_uncertainty_train = uncertainty.compute_uncertainty(batch_y_train, batch_y_train_noNoise)
        # batch_y_uncertainty_val = uncertainty.compute_uncertainty(batch_y_val, batch_y_val_noNoise)
        # batch_y_uncertainty_test = uncertainty.compute_uncertainty(batch_y_test, batch_y_test_noNoise)

        # compute uncertainty for training set
        batch_y_predict_train = rnn.compute_prediction(batch_x_train, 'training')
        batch_y_predict_train = np.reshape(batch_y_predict_train, (Ty, m_train, n_y))
        batch_y_uncertainty_train = uncertainty.compute_uncertainty(batch_y_train, batch_y_predict_train)

        # compute uncertainty for validation set
        batch_y_predict_val = rnn.compute_prediction(batch_x_val, 'validation')
        batch_y_predict_val = np.reshape(batch_y_predict_val, (Ty, m_val, n_y))
        batch_y_uncertainty_val = uncertainty.compute_uncertainty(batch_y_val, batch_y_predict_val)

        # compute uncertainty for test set
        batch_y_predict_test = rnn.compute_prediction(batch_x_test, 'test')
        batch_y_predict_test = np.reshape(batch_y_predict_test, (Ty, m_test, n_y))
        batch_y_uncertainty_test = uncertainty.compute_uncertainty(batch_y_test, batch_y_predict_test)

        n_y_uncertainty = n_y
        # create the data structures
        model_name_uncertainty = 'uncertainty' + model_name
        plot_losses_uncertainty = PlotLosses(file_path=models_path, file_name=model_name)
        rnn_uncertainty = RnnKeras(n_a=n_a, n_y=n_y_uncertainty, n_x=n_x, Tx=Tx, Ty=Ty, m_train=m_train, m_val=m_val,
                                   m_test=m_test)

        # learn the model
        rnn_uncertainty.create_model()
        rnn_uncertainty.create_optimizer()
        rnn_uncertainty.compile_model(loss_function, model_metrics)
        history_uncertainty = rnn_uncertainty.fit_model(x_train=batch_x_train, y_train=batch_y_uncertainty_train,
                                                        x_val=batch_x_val, y_val=batch_y_uncertainty_val,
                                                        epochs=epochs, plot_loss_value_obj=plot_losses_uncertainty,
                                                        verbosity=verbosity)
        rnn_uncertainty.save_model(models_path, model_name_uncertainty)
        rnn_uncertainty.visualize_model(models_path, model_name)

        # prediction on test set
        prediction_uncertainty = rnn_uncertainty.compute_prediction(batch_x_test)
        prediction_uncertainty = np.reshape(prediction_uncertainty, (
            Ty, m_test, n_y_uncertainty))  # in case  prediction does not have correct shaping
        rmse_test_uncertainty = rnn.evaluate_prediction(batch_y_uncertainty_test, np.array(prediction_uncertainty))
        print('RMSE of the uncertainty test: ', rmse_test_uncertainty)

        # change the type of value from list to array
        prediction_uncertainty = np.array(prediction_uncertainty)
        batch_t_test = np.array(batch_t_test)

        data.plot_test_prediction_data(batch_t_test[:, Ty0:Ty0 + Ty], batch_y_uncertainty_test, prediction_uncertainty,
                                       'test (line), prediction (dashed lines) of uncertainty')
        data.plot_data_adv(data=batch_y_uncertainty_train, idx_time=0, idx_features=2, idx_no_samples=1)
