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
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, LSTM, Reshape, Dropout, BatchNormalization, Conv1D, \
    MaxPooling1D, Softmax, Multiply, Add, Layer, Concatenate, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from CustomLayers import ReducedSum
from CustomLayers import get_complex_gate_output, get_simple_gate_output
from CustomLayers import get_lstm_expert_output, get_gate_selector_output, get_dense_expert_output


def get_moe_model_four_experts(number_categories, number_outputs, output_steps, input_shape=None, reg_l2=None, dp_rate=None):
    # input
    inputs = Input(shape=input_shape)

    #############
    # gate NN
    h_gate = Flatten(name='gate_nn')(inputs)
    # h_gate = get_complex_gate_output(h_gate, number_categories, output_steps, reg_l2, dp_rate)
    h_gate = get_simple_gate_output(h_gate, number_categories, output_steps, reg_l2, dp_rate)
    gate_output = Layer(name='gate_output')(h_gate)

    #############
    # expert 1
    # h_expert1 = get_lstm_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=1)
    h_expert1 = get_dense_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=1)
    #############
    # expert 2
    # h_expert2 = get_lstm_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=2)
    h_expert2 = get_dense_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=2)

    #############
    # expert 3
    # h_expert3 = get_lstm_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=3)
    h_expert3 = get_dense_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=3)
    #############
    # expert 4
    # h_expert4 = get_lstm_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=4)
    h_expert4 = get_dense_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=4)

    #############
    # Gate Layer
    moe_output = get_gate_selector_output(h_gate, h_expert1, h_expert2, h_expert3, h_expert4, number_categories,
                                          number_outputs, output_steps)

    # print('moe_output shape: {}'.format(moe_output.shape))

    # moe_output = Layer(name='moe_output')(h_expert1)
    # moe_output = GateLayer(name='moe_output')([h_gate, h_expert1, h_expert2, h_expert3, h_expert4])

    model = Model(inputs=inputs, outputs=[gate_output, moe_output])

    return model


def get_moe_model_one_expert(number_categories, number_experts_outputs, output_steps, input_shape=None, reg_l1=None, reg_l2=None, dp_rate=None):
    # input
    inputs = Input(shape=input_shape)

    #############
    # gate NN
    h_gate = Flatten(name='gate_nn')(inputs)
    h_gate = get_complex_gate_output(h_gate, number_categories, output_steps, reg_l2, dp_rate)
    # h_gate = get_simple_gate_output(h_gate, number_categories, output_steps, reg_l2, dp_rate)
    gate_output = Layer(name='gate_output')(h_gate)

    #############
    # expert 1
    # h_expert1 = get_lstm_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=1)
    # h_expert1 = get_dense_expert_output(inputs, number_outputs, output_steps, reg_l2, dp_rate, expert_number=1)
    #
    # h_expert = Flatten(name='expert{}_nn'.format(1))(inputs)
    #
    # h_expert = Dense(512,
    #                  activation='relu',
    #                  kernel_regularizer=regularizers.l2(reg_l2),
    #                  bias_regularizer=regularizers.l2(reg_l2))(h_expert)
    # #
    h_expert = LSTM(128,
                    name='expert_nn',
                    return_sequences=True,
                    activation=LeakyReLU(),
                    kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
                    )(inputs)

    h_expert = BatchNormalization()(h_expert)

    h_expert = LSTM(64,
                    name='expert_nn',
                    return_sequences=False,
                    kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
                    )(inputs)

    # h_expert = Flatten()(inputs)

    h_expert = BatchNormalization()(h_expert)
    #
    # h_expert = LSTM(128,
    #                 return_sequences=False,
    #                 kernel_regularizer=regularizers.l1_l2(reg_l2, reg_l2),
    #                 )(inputs)

    # h_expert = Flatten()(inputs)

    # h_expert = BatchNormalization()(h_expert)

    # h_expert = Dense(number_outputs, activation=LeakyReLU())(h_expert)

    # h_expert = Dropout(dp_rate)(h_expert)

    h_expert = BatchNormalization()(h_expert)

    h_expert = Dense(output_steps * number_experts_outputs)(h_expert)

    # h_expert = BatchNormalization()(h_expert)

    # h_expert = Dropout(dp_rate)(h_expert)

    moe_output = Reshape([output_steps, number_experts_outputs], name='moe_output')(h_expert)

    #############
    # Gate Layer
    # h_expert1 = Reshape([output_steps, number_outputs, 1])(h_expert1)
    # h_expert2 = Reshape([output_steps, number_outputs, 1])(h_expert2)
    # h_expert3 = Reshape([output_steps, number_outputs, 1])(h_expert3)
    # h_expert4 = Reshape([output_steps, number_outputs, 1])(h_expert4)
    # experts = Concatenate(axis=-1)([h_expert1, h_expert2, h_expert3, h_expert4])
    print('experts shape: {}'.format(h_expert.shape))

    # h_gate = Reshape([output_steps, 1, number_categories])(h_gate)
    # print('h_gate shape: {}'.format(h_gate.shape))
    #
    # moe_output_ = Multiply()([h_expert, h_gate])
    # print('moe_output_ shape: {}'.format(moe_output_.shape))
    # moe_output = ReducedSum(name='moe_output', axis=3)(moe_output_)

    model = Model(inputs=inputs, outputs=[gate_output, moe_output])

    return model


def compile_model(model):
    model.compile(loss={'gate_output': CategoricalCrossentropy(from_logits=False),
                        'moe_output': tf.losses.MeanSquaredError()},
                  optimizer=Adam(),
                  loss_weights={'gate_output': 1.0, 'moe_output': 1.0},
                  metrics={'gate_output': ['accuracy'],
                           'moe_output': ['mae']})

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
                                      monitor='val_loss',
                                      save_best_only=True,
                                      verbose=1,
                                      shuffle=True)

    history = model.fit(window.train,
                        epochs=max_epochs,
                        validation_data=window.val,
                        # validation_split=0.15,
                        callbacks=[early_stopping, callback_loss_accuracy_plot, checkpoint_best],
                        verbose=1)
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


def save_nn_model(model, file_path='', file_name='myModel'):
    Path(file_path).mkdir(parents=True, exist_ok=True)
    model.save('{}/{}.h5'.format(file_path, file_name))  # creates a HDF5 file 'my_model.h5'
    print('Model is saved as: {}'.format('{}/{}.h5'.format(file_path, file_name)))
    return


def visualize_model(model, file_path='', file_name='myModel'):
    plot_model(model, to_file='{}/{}.png'.format(file_path, file_name), show_shapes=True)
    print('Model architecture is saved as: {}'.format('{}/{}.png'.format(file_path, file_name)))

    return


def load_model_from_file(file_path='', file_name='myModel'):
    model = load_model('{}/{}.h5'.format(file_path, file_name), custom_objects={'ReducedSum': ReducedSum,
                                                                                'LeakyReLU': LeakyReLU})
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
        self.plot_loss = plt.figure(1, figsize=(6, 8))
        self.axs_loss = self.plot_loss.subplots(3)

        self.plot_metrics = plt.figure(2, figsize=(6, 8))
        self.axs_metrics = self.plot_metrics.subplots(2)

        # self.file_path = file_path
        # self.file_name = file_name

        # def on_train_begin(self, logs=None):
        #     print('self.params', self.params)
        #     self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.losses = []
        self.val_losses = []

        self.losses = []
        self.val_losses = []
        self.gate_losses = []
        self.val_gate_losses = []
        self.moe_losses = []
        self.val_moe_losses = []

        self.gate_accuracy = []
        self.val_gate_accuracy = []

        self.moe_mae = []
        self.val_moe_mae = []

    def on_epoch_end(self, epoch, logs=None):
        print('[on_epoch_end] epoch: {} , loss: {} , val_loss : {}'.format(epoch, logs['loss'], logs['val_loss']))

        # update log values
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.gate_losses.append(logs['gate_output_loss'])
        self.val_gate_losses.append(logs['val_gate_output_loss'])
        self.moe_losses.append(logs['moe_output_loss'])
        self.val_moe_losses.append(logs['val_moe_output_loss'])

        self.gate_accuracy.append(logs['gate_output_accuracy'])
        self.val_gate_accuracy.append(logs['val_gate_output_accuracy'])
        self.moe_mae.append(logs['moe_output_mae'])
        self.val_moe_mae.append(logs['val_moe_output_mae'])

        if epoch == 0:
            return

        # plot 1: losses
        plt.figure(1, figsize=(6, 8))
        self.plot_loss.clf()
        self.axs_loss = self.plot_loss.subplots(3)
        self.axs_loss[0].plot(self.losses)
        self.axs_loss[0].plot(self.val_losses)
        self.axs_loss[0].set_title('model loss')
        self.axs_loss[0].legend(['train', 'validation'], loc='upper left')

        self.axs_loss[1].plot(self.gate_losses)
        self.axs_loss[1].plot(self.val_gate_losses)
        self.axs_loss[1].set_title('gate loss')
        self.axs_loss[1].legend(['train', 'validation'], loc='upper left')

        self.axs_loss[2].plot(self.moe_losses)
        self.axs_loss[2].plot(self.val_moe_losses)
        self.axs_loss[2].set_title('moe loss')
        self.axs_loss[2].legend(['train', 'validation'], loc='upper left')

        for ax in self.axs_loss.flat:
            ax.set(xlabel='epoch', ylabel='loss')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in self.axs_loss.flat:
            ax.label_outer()

        plt.show()
        plt.pause(0.001)
        plt.tight_layout()

        # plot 2: accuracy and mae
        plt.figure(2, figsize=(12, 8))
        self.plot_metrics.clf()
        self.axs_metrics = self.plot_metrics.subplots(2)
        self.axs_metrics[0].plot(self.gate_accuracy)
        self.axs_metrics[0].plot(self.val_gate_accuracy)
        self.axs_metrics[0].set_title('model accuracy')
        self.axs_metrics[0].set(ylabel='accuracy')
        self.axs_metrics[0].legend(['train', 'validation'], loc='upper left')

        self.axs_metrics[1].plot(self.moe_mae)
        self.axs_metrics[1].plot(self.val_moe_mae)
        self.axs_metrics[1].set_title('model mae')
        self.axs_metrics[1].set(xlabel='epoch', ylabel='mae')
        self.axs_metrics[1].legend(['train', 'validation'], loc='upper left')

        for ax in self.axs_metrics.flat:
            ax.label_outer()

        plt.show()
        plt.pause(0.001)
        plt.tight_layout()

    def save_model(self, epoch, val_loss):
        # creates a HDF5 file 'my_model_epochNumber_valLoss.h5'
        Path(self.file_path).mkdir(parents=True, exist_ok=True)
        self.model.save('{}/{}_{}_{}.h5'.format(self.file_path, self.file_name, epoch, val_loss))
        return

# =================================


def get_dense_model_classification(number_categories, output_steps, input_shape=None, reg_l2=None, dp_rate=None):
    model = Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        # Lambda(lambda x: x[:, -1:, :]),
        Flatten(input_shape=input_shape),
        # Shape => [batch, 1, dense_units]
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(256, activation='relu'),
        # Dense(512, activation='relu', kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        # BatchNormalization(),
        # Dropout(dp_rate),
        # Dense(128, activation='relu', kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2)),
        # BatchNormalization(),
        # Dropout(dp_rate),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
              bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
              bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
              bias_regularizer=regularizers.l2(reg_l2)),
        BatchNormalization(),
        Dropout(dp_rate),
        Dense(number_categories * output_steps),
        Reshape([output_steps, number_categories]),
        Softmax(name='gate_output')
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


def get_lstm_regression_model_sequential(number_outputs, output_steps, input_shape, reg_l2, dp_rate):
    model = Sequential([
        LSTM(256, return_sequences=False, activation='relu', name='lstm_a',
             kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
             recurrent_regularizer=regularizers.l2(reg_l2), dropout=dp_rate, recurrent_dropout=dp_rate,
             input_shape=(input_shape[0], input_shape[1])),
        BatchNormalization(),
        # LSTM(64, return_sequences=True, activation='relu', name='lstm_b',
        #      kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
        #      recurrent_regularizer=regularizers.l2(reg_l2), dropout=dp_rate, recurrent_dropout=dp_rate),
        # BatchNormalization(),
        # LSTM(32, return_sequences=True, activation='relu', name='lstm_c',
        #      kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
        #      recurrent_regularizer=regularizers.l2(reg_l2), dropout=dp_rate, recurrent_dropout=dp_rate),
        # BatchNormalization(),
        # LSTM(16, return_sequences=True, activation='relu', name='lstm_d',
        #      kernel_regularizer=regularizers.l2(reg_l2), bias_regularizer=regularizers.l2(reg_l2),
        #      recurrent_regularizer=regularizers.l2(reg_l2), dropout=dp_rate, recurrent_dropout=dp_rate),
        # Flatten(),
        # Dense(32, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
        #       bias_regularizer=regularizers.l2(reg_l2)),
        # BatchNormalization(),
        # Dropout(dp_rate),
        # Dense(16, activation='relu', kernel_regularizer=regularizers.l2(reg_l2),
        #       bias_regularizer=regularizers.l2(reg_l2)),
        # BatchNormalization(),
        # Dropout(dp_rate),
        Dense(output_steps * number_outputs),
        Reshape([output_steps, number_outputs])
    ])
    return model


def compile_and_fit_regression(model, window, patience, max_epochs):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.Accuracy()])
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val
                        , callbacks=[early_stopping
                                     # , plot_losses
                                     ],
                        use_multiprocessing=True)

    return history

