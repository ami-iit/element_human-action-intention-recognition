from fnmatch import filter
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from tensorflow.keras.layers import Lambda, Dense, Flatten, LSTM, Reshape, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint


def get_dense_model(number_categories, input_shape, reg_l2, dp_rate):
    model = Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        # Lambda(lambda x: x[:, -1:, :]),
        Flatten(input_shape=input_shape),
        # Shape => [batch, 1, dense_units]
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(256, activation='relu'),
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(number_categories, activation='softmax')
    ])
    return model


def get_cnn_model(number_categories, input_shape, reg_l2, dp_rate):
    model = Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
               input_shape=input_shape),
        BatchNormalization(),
        Dropout(dp_rate),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
              bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
              bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(number_categories, activation='softmax')
    ])
    return model


def get_lstm_model(number_categories, input_shape, reg_l2, dp_rate):
    model = Sequential([
        LSTM(128, return_sequences=True,  activation='relu', name='lstm_a',
             kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
             recurrent_regularizer=regularizers.l2(reg_l2), dropout=dp_rate, recurrent_dropout=dp_rate,
             input_shape=(input_shape[0], input_shape[1])),
        BatchNormalization(),
        LSTM(64, return_sequences=True, activation='relu', name='lstm_b',
             kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
             recurrent_regularizer=regularizers.l2(reg_l2), dropout=dp_rate, recurrent_dropout=dp_rate),
        BatchNormalization(),
        LSTM(32, return_sequences=True, activation='relu', name='lstm_c',
             kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
             recurrent_regularizer=regularizers.l2(reg_l2), dropout=dp_rate, recurrent_dropout=dp_rate),
        BatchNormalization(),
        LSTM(16, return_sequences=True, activation='relu', name='lstm_d',
             kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
             recurrent_regularizer=regularizers.l2(reg_l2), dropout=dp_rate, recurrent_dropout=dp_rate),
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
              bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
              bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(number_categories, activation='softmax')
    ])
    return model


def compile_model(model):
    model.compile(loss=CategoricalCrossentropy(from_logits=False),
                  optimizer=Adam(),
                  metrics=['accuracy'])
    # To help the numerical stability of the back propagation, I can remove the softmax from the last dense layer and
    # add from_logits=True to the loss function

    return model


def fit_model(model, window, patience=2, max_epochs=20, model_path='', model_name='myModel'):

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   min_delta=0.01,
                                   mode='min')
    callback_loss_accuracy_plot = CallbackPlotLossesAccuracy()

    checkpoint_best = ModelCheckpoint(filepath='{}/{}.h5'.format(model_path, model_name),
                                      save_weights_only=False,
                                      save_freq='epoch',
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

    history = model.fit(window.train,
                        epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping, callback_loss_accuracy_plot, checkpoint_best])
    return history


def plot_losses(history):
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return


def plot_accuracy(history):
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return

def compile_and_fit(model, window, plot_losses, ):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')


    # model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               optimizer=tf.optimizers.Adam(),
    #               metrics=[tf.metrics.Accuracy()])


    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val
                        , callbacks=[early_stopping
                            #, plot_losses
                            ],
                        use_multiprocessing=True)



    return history


def save_model(model, file_path='', file_name='myModel'):
    Path(file_path).mkdir(parents=True, exist_ok=True)
    model.save('{}/{}.h5'.format(file_path, file_name))  # creates a HDF5 file 'my_model.h5'
    print('the model is saved as: {}'.format('{}/{}.h5'.format(file_path, file_name)))
    return

def visualize_model(model, file_path='', file_name='myModel'):
    plot_model(model, to_file='{}/{}.png'.format(file_path, file_name), show_shapes=True)

    return


def load_model_from_file(file_path='', file_name='myModel'):
    model = load_model('{}/{}.h5'.format(file_path, file_name))
    return model


def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Log-loss (cost function)"}
    if x in translations:
        return translations[x]
    else:
        return x


class CallbackPlotLossesAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, figsize=None, file_path='', file_name='myModel'):
        # super(PlotLosses, self).__init__()
        self.plot_loss = plt.figure(1, figsize=(12, 8))
        self.plot_acc = plt.figure(2, figsize=(12, 8))
        # self.file_path = file_path
        # self.file_name = file_name

    # def on_train_begin(self, logs=None):
    #     print('self.params', self.params)
    #     self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        print('epoch: {} , loss: {} , val_loss : {}'.format(epoch, logs['loss'], logs['val_loss']))
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.accuracy.append(logs['accuracy'])
        self.val_accuracy.append(logs['val_accuracy'])

        plt.figure(1, figsize=(12, 8))
        self.plot_loss.clf()
        plt.plot(self.losses)
        plt.plot(self.val_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.pause(0.001)
        plt.tight_layout()

        plt.figure(2, figsize=(12, 8))
        self.plot_acc.clf()
        plt.plot(self.accuracy)
        plt.plot(self.val_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.pause(0.001)
        plt.tight_layout()


        # save the model

        # self.save_model(epoch, val_loss=logs['val_loss'])

        # for metric_id, metric in enumerate(self.base_metrics):
        #     if metric == 'loss':
        #
        #         plt.plot(range(1, len(self.logs) + 1),
        #                  [log[metric] for log in self.logs],
        #                  label="training")
        #         if self.params['do_validation']:
        #             plt.plot(range(1, len(self.logs) + 1),
        #                      [log['val_' + metric] for log in self.logs], '--',
        #                      label="validation")
        #         plt.title(translate_metric(metric))
        #         plt.xlabel('epoch')
        #         plt.legend(loc='center right')
        #



    def save_model(self, epoch, val_loss):
        # creates a HDF5 file 'my_model_epochNumber_valLoss.h5'
        Path(self.file_path).mkdir(parents=True, exist_ok=True)
        self.model.save('{}/{}_{}_{}.h5'.format(self.file_path, self.file_name, epoch, val_loss))
        return

