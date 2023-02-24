#######################
##### Customlayers #####
########################
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, Conv1D, Reshape, Dropout, BatchNormalization, \
                                    Softmax, Multiply, Layer, Concatenate, LeakyReLU
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import math
from datetime import datetime

max_epochs = 40

# gate nn architecture
def get_gate_nn_output(input_, number_categories, output_steps, reg_l1, reg_l2, dp_rate):

    output_ = Flatten(name='gate_nn')(input_)

    output_ = Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(output_)

    #output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
                    bias_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(output_)

    #output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Dense(number_categories * output_steps)(output_)

    # ! default: axis=-1, normalization is done over last axis
    output_ = Reshape([output_steps, number_categories])(output_)

    output_ = Softmax()(output_)

    ## try lstm for action recognition
    #output_ = Conv1D(filters=50, kernel_size=74, activation='relu', padding='same',
    #                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    #                 bias_regularizer=regularizers.l1_l2(reg_l1, reg_l2))(input_)

    #output_ = LSTM(3,
    #              name='gate_nn',
    #              return_sequences=False,
    #              activation=LeakyReLU(),
    #              kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    #              )(output_)
    #output_ = Dropout(dp_rate)(output_)
    #output_ = Dense(number_categories * output_steps)(output_)
    #output_ = Reshape([output_steps, number_categories])(output_)

    #output_ = Softmax()(output_)
    return output_

# expert nn architecture
def get_expert_nn_output(input_, number_experts_outputs, output_steps, reg_l1, reg_l2, dp_rate, expert_number):
    h_expert = LSTM(43,
                    name='expert_{}_nn01'.format(expert_number),
                    return_sequences=False,
                    activation=LeakyReLU(),
                    kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
                    )(input_)
    
    #h_expert = LSTM(78,
    #                name='expert_{}_nn02'.format(expert_number),
    #                return_sequences=False,
    #                activation=LeakyReLU(),
    #                kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
    #                )(h_expert)

    #h_expert = BatchNormalization()(h_expert)

    h_expert = Dropout(dp_rate)(h_expert)

    #h_expert = Dense(output_steps * number_experts_outputs)(h_expert)

    #h_expert = Dense(output_steps * number_experts_outputs)(h_expert)

    h_expert = Dense(output_steps * number_experts_outputs)(h_expert)

    h_expert = Reshape([output_steps, number_experts_outputs])(h_expert)

    return h_expert

# experts nn: associative(/competitive)
def get_gate_selector_output_associative(h_gate, h_expert1, h_expert2, h_expert3, 
                                         number_categories, number_experts_outputs, output_steps):

    h_expert1 = Reshape([output_steps, number_experts_outputs, 1])(h_expert1)
    h_expert2 = Reshape([output_steps, number_experts_outputs, 1])(h_expert2)
    h_expert3 = Reshape([output_steps, number_experts_outputs, 1])(h_expert3)

    experts = Concatenate(axis=-1)([h_expert1, h_expert2, h_expert3])
    print('experts shape: {}'.format(experts.shape))

    h_gate = Reshape([output_steps, 1, number_categories])(h_gate)
    print('h_gate shape: {}'.format(h_gate.shape))
    
    #weights = np.full((1, 200), 0.125, dtype=np.float64)
    #weights = Reshape([output_steps, 1, number_categories])(weights)
    moe_output_ = Multiply()([experts, h_gate])
    #moe_output_ = Multiply()([experts, weights])
    print('moe_output_ shape: {}'.format(moe_output_.shape))
    moe_output = ReducedSumLayer(name='moe_output', axis=3)(moe_output_)
    print('moe_output shape: {}'.format(moe_output.shape))

    return moe_output

# create a layer for multiple experts
class ReducedSumLayer(Layer):

    def __init__(self, axis=None, **kwargs):
        super(ReducedSumLayer, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        print('[build] input_shape: {}'.format(input_shape))
        print('[build] input_shape[0]: {}'.format(input_shape[0]))
        return

    def call(self, inputs):
        # return tf.matmul(inputs, self.w) + self.b
        # print('__call__')
        # print('input tensor shape: {}'.format(inputs.shape))
        output = tf.reduce_sum(inputs, axis=self.axis)
        # print('output tensor shape: {}'.format(output.shape))
        return output

    def get_config(self):
        return {"axis": self.axis}

# create GMoE (gate nn + experts nn)
def get_moe_nn_model(number_categories, number_experts_outputs, output_steps, input_shape=None,
                       reg_l1_gate=None, reg_l2_gate=None,
                       reg_l1_experts=None, reg_l2_experts=None, dp_rate=None):
    # input
    inputs = Input(shape=input_shape)

    # gate NN
    h_gate = get_gate_nn_output(inputs, number_categories, output_steps,
                                     reg_l1_gate, reg_l2_gate,
                                     dp_rate)
    gate_output = Layer(name='gate_output')(h_gate)
    
    # expert NN
    h_expert1 = get_expert_nn_output(inputs, number_experts_outputs, output_steps,
                                     reg_l1_experts, reg_l2_experts, dp_rate, 1)
    h_expert2 = get_expert_nn_output(inputs, number_experts_outputs, output_steps,
                                     reg_l1_experts, reg_l2_experts, dp_rate, 2)
    h_expert3 = get_expert_nn_output(inputs, number_experts_outputs, output_steps,
                                     reg_l1_experts, reg_l2_experts, dp_rate, 3)
          
    moe_output = get_gate_selector_output_associative(h_gate, h_expert1, h_expert2, h_expert3, 
                                                      number_categories, number_experts_outputs, output_steps)

    #model = Model(inputs=inputs, outputs=[gate_output, moe_output])
    model = Model(inputs=inputs, outputs=[gate_output, moe_output])

    return model

# loss function/optimizer/loss weights/metrics
def compile_model(model):
    model.compile(loss={'gate_output': CategoricalCrossentropy(from_logits=False),
                        'moe_output': tf.losses.MeanSquaredError()},
                  optimizer=Adam(epsilon=1e-06),
                  loss_weights={'gate_output': 1.0, 'moe_output': 0.5},
                  metrics={'gate_output': ['accuracy'],
                           'moe_output': ['mae']})
    return model

# early stopping/save best nn model/plot train&validation results
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

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1)

    history = model.fit(window.train,
                        epochs=max_epochs,
                        validation_data=window.val,
                        # validation_split=0.15,
                        callbacks=[early_stopping, callback_loss_accuracy_plot, checkpoint_best, lr_scheduler],
                        verbose=1)
    return history

def lr_step_decay(epoch, lr):
    initial_learning_rate = 0.01
    #k = 0.1
    drop_rate = 0.5
    epochs_drop = 5.0
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
    #return initial_learning_rate * math.exp(-k*epoch)

# plot train/valiadation/test performance metrics
class CallbackPlotLossesAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, figsize=None, file_path='', file_name='myModel'):
        # super(PlotLosses, self).__init__()
        self.plot_loss = plt.figure(1, figsize=(6, 8))
        self.axs_loss = self.plot_loss.subplots(3)

        self.plot_metrics = plt.figure(2, figsize=(6, 8))
        self.axs_metrics = self.plot_metrics.subplots(2)
        
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

        self.font_size = 16

    def on_epoch_end(self, epoch, logs=None):
        print('[on_epoch_end] epoch: {} , loss: {} , val_loss : {}'.format(epoch, logs['loss'], logs['val_loss']))

        ## loss ##
        # update log values
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])

        self.gate_losses.append(logs['gate_output_loss'])
        self.val_gate_losses.append(logs['val_gate_output_loss'])

        self.moe_losses.append(logs['moe_output_loss'])
        self.val_moe_losses.append(logs['val_moe_output_loss'])

        ## metrics ##
        self.gate_accuracy.append(logs['gate_output_accuracy'])
        self.val_gate_accuracy.append(logs['val_gate_output_accuracy'])

        self.moe_mae.append(logs['moe_output_mae'])
        self.val_moe_mae.append(logs['val_moe_output_mae'])
   
        while epoch < max_epochs-2:
            return

        # plot 1: losses
        plt.figure(1, figsize=(6, 8))
        self.plot_loss.clf()
        self.axs_loss = self.plot_loss.subplots(3)

        self.axs_loss[0].plot(self.losses)
        self.axs_loss[0].plot(self.val_losses)

        self.axs_loss[0].set_title('model loss', fontsize=self.font_size)
        self.axs_loss[0].legend(['train', 'validation'], loc='upper left', fontsize=self.font_size)
        self.axs_loss[0].set_ylim(0, 4.0)

        self.axs_loss[1].plot(self.gate_losses)
        self.axs_loss[1].plot(self.val_gate_losses)

        self.axs_loss[1].set_title('gate loss', fontsize=self.font_size)
        self.axs_loss[1].legend(['train', 'validation'], loc='upper left', fontsize=self.font_size)

        self.axs_loss[2].plot(self.moe_losses)
        self.axs_loss[2].plot(self.val_moe_losses)

        self.axs_loss[2].set_title('moe loss', fontsize=self.font_size)
        self.axs_loss[2].legend(['train', 'validation'], loc='upper left', fontsize=self.font_size)

        for ax in self.axs_loss.flat:
            # ax.set(xlabel='epoch', ylabel='loss')
            ax.set_xlabel('epoch', fontsize=self.font_size)
            ax.set_ylabel('loss', fontsize=self.font_size)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in self.axs_loss.flat:
            ax.label_outer()

        # plot 2: accuracy and mae
        plt.figure(2, figsize=(12, 8))
        self.plot_metrics.clf()
        self.axs_metrics = self.plot_metrics.subplots(2)

        self.axs_metrics[0].plot(self.gate_accuracy)
        self.axs_metrics[0].plot(self.val_gate_accuracy)

        self.axs_metrics[0].set_title('model accuracy', fontsize=self.font_size)
        self.axs_metrics[0].set_ylabel('accuracy', fontsize=self.font_size)
        self.axs_metrics[0].legend(['train', 'validation'], loc='upper left', fontsize=self.font_size)

        self.axs_metrics[1].plot(self.moe_mae)
        self.axs_metrics[1].plot(self.val_moe_mae)

        self.axs_metrics[1].set_title('model mae', fontsize=self.font_size)
        self.axs_metrics[1].set_xlabel('epoch', fontsize=self.font_size)
        self.axs_metrics[1].set_ylabel('mae', fontsize=self.font_size)
        self.axs_metrics[1].legend(['train', 'validation'], loc='upper left', fontsize=self.font_size)

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

    def on_train_end(self, logs=None):
        print('loss =', self.losses)
        print('val_loss =', self.val_losses)
        print('gate_output_loss =', self.gate_losses)
        print('val_gate_output_loss =', self.val_gate_losses)
        print('moe_output_loss =', self.moe_losses)
        print('val_moe_output_loss =', self.val_moe_losses)
        print('gate_output_accuracy =', self.gate_accuracy)
        print('val_gate_output_accuracy =', self.val_gate_accuracy)
        print('moe_output_mae =', self.moe_mae)
        print('val_moe_output_mae =', self.val_moe_mae)

        return

def get_time_now():
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return date_time

def save_nn_model(model, file_path='', file_name='myModel'):
    Path(file_path).mkdir(parents=True, exist_ok=True)
    model.save('{}/{}.h5'.format(file_path, file_name))  # creates a HDF5 file 'my_model.h5'
    print('Model is saved as: {}'.format('{}/{}.h5'.format(file_path, file_name)))
    return

def visualize_model(model, file_path='', file_name='myModel'):
    plot_model(model, to_file='{}/{}.png'.format(file_path, file_name), show_shapes=True)
    print('Model architecture is saved as: {}'.format('{}/{}.png'.format(file_path, file_name)))

    return