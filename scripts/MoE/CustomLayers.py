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
    MaxPooling1D, Softmax, Multiply, Add, Layer, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint


class ReducedSum(Layer):

    def __init__(self, axis=None, **kwargs):
        super(ReducedSum, self).__init__(**kwargs)
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


# models:
def get_complex_gate_output(input_, number_categories, output_steps, reg_l2, dp_rate):
    output_ = Dense(1024, activation='relu',
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(input_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(input_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    # output_ = Dense(64, activation='relu',
    #                 kernel_regularizer=regularizers.l2(reg_l2),
    #                 bias_regularizer=regularizers.l2(reg_l2))(output_)
    #
    # output_ = BatchNormalization()(output_)
    #
    # output_ = Dropout(dp_rate)(output_)
    #
    # output_ = Dense(16, activation='relu',
    #                 kernel_regularizer=regularizers.l2(reg_l2),
    #                 bias_regularizer=regularizers.l2(reg_l2))(output_)
    #
    # output_ = BatchNormalization()(output_)
    #
    # output_ = Dropout(dp_rate)(output_)

    output_ = Dense(number_categories * output_steps)(output_)

    # ! default: axis=-1, normalization is done over last axis
    output_ = Reshape([output_steps, number_categories])(output_)

    output_ = Softmax()(output_)

    return output_


def get_simple_gate_output(input_, number_categories, output_steps, reg_l2, dp_rate):
    output_ = Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(input_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Dense(number_categories * output_steps,
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(output_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    # ! default: axis=-1, normalization is done over last axis
    output_ = Reshape([output_steps, number_categories])(output_)

    output_ = Softmax()(output_)

    return output_


def get_dense_expert_output(input_, number_outputs, output_steps, reg_l2, dp_rate, expert_number):
    output_ = Flatten(name='expert{}_nn'.format(expert_number))(input_)

    output_ = Dense(output_steps * number_outputs,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(output_)
    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Dense(output_steps * number_outputs,
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(output_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Reshape([output_steps, number_outputs])(output_)

    return output_


def get_lstm_expert_output(input_, number_outputs, output_steps, reg_l2, dp_rate, expert_number):
    output_ = LSTM(64,
                   name='expert{}_nn'.format(expert_number),
                   return_sequences=True,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(reg_l2),
                   bias_regularizer=regularizers.l2(reg_l2),
                   recurrent_regularizer=regularizers.l2(reg_l2),
                   dropout=dp_rate,
                   recurrent_dropout=dp_rate)(input_)

    output_ = BatchNormalization()(output_)

    output_ = Flatten()(output_)

    output_ = Dense(output_steps * number_outputs,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(output_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Dense(output_steps * number_outputs,
                    kernel_regularizer=regularizers.l2(reg_l2),
                    bias_regularizer=regularizers.l2(reg_l2))(output_)

    output_ = BatchNormalization()(output_)

    output_ = Dropout(dp_rate)(output_)

    output_ = Reshape([output_steps, number_outputs])(output_)

    return output_


def get_gate_selector_output(h_gate, h_expert1, h_expert2, h_expert3, h_expert4, number_categories, number_outputs,
                             output_steps):
    h_expert1 = Reshape([output_steps, number_outputs, 1])(h_expert1)
    h_expert2 = Reshape([output_steps, number_outputs, 1])(h_expert2)
    h_expert3 = Reshape([output_steps, number_outputs, 1])(h_expert3)
    h_expert4 = Reshape([output_steps, number_outputs, 1])(h_expert4)
    experts = Concatenate(axis=-1)([h_expert1, h_expert2, h_expert3, h_expert4])
    print('experts shape: {}'.format(experts.shape))

    h_gate = Reshape([output_steps, 1, number_categories])(h_gate)
    print('h_gate shape: {}'.format(h_gate.shape))

    moe_output_ = Multiply()([experts, h_gate])
    print('moe_output_ shape: {}'.format(moe_output_.shape))
    moe_output = ReducedSum(name='moe_output', axis=3)(moe_output_)

    return moe_output


########################


class GateLayer(Layer):

    def __init__(self, **kwargs):
        super(GateLayer, self).__init__(**kwargs)
        # self.supports_masking = True  #???
        print(['__init__'])

    def build(self, input_shape):
        # purely used to check the layers size
        # the first input is always the gate outputs
        # all the rest are the experts outputs

        # gate output size: (, number of output time steps, number of categories)
        # the expert output size : (, number of output time steps, number of output features)
        # Consider:
        #   x = number of categories
        #   Then we should have x number of experts, so the size of the input tensor list should be x+1
        print('[build] input_shape: {}'.format(input_shape))
        print('[build] input_shape[0]: {}'.format(input_shape[0]))
        print('[build] input_shape[0] type: {}'.format(type(input_shape[0])))
        print('[build] input_shape is tuple: {}'.format(isinstance(input_shape[0], tuple)))

        # to Enable later
        # if not isinstance(input_shape[0], tuple):
        #     raise ValueError('A gate layer should be called on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A gate layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        batch_sizes = {s[0] for s in input_shape if s} - {None}
        print('[build] batch_sizes: {} , len(batch_sizes): {}'.format(batch_sizes, len(batch_sizes)))
        if len(batch_sizes) > 1:
            raise ValueError(
                'Can not merge tensors with different '
                'batch sizes. Got tensors with shapes : ' + str(input_shape))

        # check if number of categories and input data are equal
        # print('[build] gate shape : {}, {}, {}'.format(input_shape[0].aslist(), input_shape[0][1][2]))
        gate_shape = input_shape[0]
        categories_size = gate_shape[-1]
        print('[build] categories_size: {} '.format(categories_size, ))
        if categories_size + 1 != len(input_shape):
            raise ValueError(
                'Gate layer should have similar number of categories and input experts.'
                'Got categories size of : ' + str(categories_size) +
                ' , Got number of experts: ' + str(len(input_shape) - 1) +
                ' Total number of input tensors: ' + str(len(input_shape)))

        if input_shape[1] is None:
            output_shape = None
        else:
            output_shape = input_shape[1][1:]
        print('output_shape: {}'.format(output_shape))

    # Defines the computation from inputs to outputs
    def call(self, inputs):
        # return tf.matmul(inputs, self.w) + self.b
        print('__call__')
        if not isinstance(inputs, (list, tuple)):
            raise ValueError('A gate layer should be called on a list of inputs.')
        print('expert tensor shape: {}'.format(inputs[-1].shape))
        stacked_experts = tf.stack(inputs[1:], axis=-1)
        print('stacked expert tensors shape: {}'.format(stacked_experts.shape))
        gate_shape = inputs[0].shape  # ! gate shape [No. samples, No of Time Steps (Ty), No. categories]
        print('current gate shape: {}, dtype: {}'.format(gate_shape, type(gate_shape)))
        num_output_features = inputs[1].shape[-1]
        print('number of output features: {}, dtype: {}'.format(num_output_features, type(num_output_features)))
        # ! gate shape [No. samples, No of Time Steps (Ty), 1, No. categories]
        gate_output = inputs[0]
        print('gate_ shape: {}, dtype: {}'.format(gate_output.shape, type(gate_output)))

        reshaped_gate = tf.reshape(inputs[0], [gate_shape[0], gate_shape[1], 1, gate_shape[2]])
        new_gate_shape = reshaped_gate.shape
        print('new gate shape: {}, dtype: {}'.format(new_gate_shape, type(new_gate_shape)))
        print('old gate: {}'.format(inputs[0]))
        print('new gate: {}'.format(reshaped_gate))
        broadcast_gate = tf.broadcast_to(reshaped_gate,
                                         shape=[gate_shape[0], gate_shape[1], num_output_features, gate_shape[2]])
        print('broadcasted gate shape: {}'.format(broadcast_gate.shape))
        # print('broadcasted gate: {}'.format(broadcast_gate))
        # ! reduce sum about the categories, probability axis = 3 : the last one
        output = tf.reduce_sum(tf.multiply(broadcast_gate, stacked_experts), axis=3)
        # print('output_: {}'.format(output_))
        # print('output_.shape: {}'.format(output_.shape))
        return output


gate_ = tf.constant([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]])  # 2 batches, 3 time steps, 2 categories

e1 = tf.constant([[[1, 2, 1, 2], [3, 4, 3, 4], [5, 6, 5, 6]],
                  [[10, 20, 10, 20], [30, 40, 30, 40], [50, 60, 50, 60]]])  # 2 batches, 3 time steps, 4 features
e2 = tf.constant([[[100, 200, 100, 200], [300, 400, 300, 400], [500, 600, 500, 600]],
                  [[1000, 2000, 1000, 2000], [3000, 4000, 3000, 4000],
                   [5000, 6000, 5000, 6000]]])  # 2 batches, 3 time steps, 4 features

gl = GateLayer()
gl([gate_, e1, e2])
