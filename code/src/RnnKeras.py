from SequentialModel import SequentialModel
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.layers import Concatenate
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import numpy as np
from pathlib import Path
import copy


class RnnKeras(SequentialModel):
    def __init__(self, n_a, n_y, n_x, Tx, Ty, m_train, m_val, m_test):
        # # print the tensorflow version
        # print(tf.__version__)
        # # hidden state dimensions of each RNN/LSTM cell
        # self.n_a = n_a
        # # output dimensions of each output layer
        # self.n_y = n_y
        # # input dimensions of input layer
        # self.n_x = n_x
        # # time series (sequence) length, the time horizon of prediction
        # self.Tx = Tx
        # self.Ty= Ty
        # # number of training set
        # self.m = m
        super().__init__(n_a, n_y, n_x, Tx, Ty, m_train, m_val, m_test)

        # TODO: Training
        # self.learning_rate = 0.0025
        # self.lambda_loss_amount = 0.0015
        # self.training_iters = training_data_count * 300  # Loop 300 times on the dataset
        # self.batch_size = 1500
        # self.display_iter = 30000  # To show test set accuracy during training
        # self.time = 0.0

        ## utilities


        # self.lambda_input = lambda_inputLambda(lambda z: z)
        # lambda layer for one-2-many architecture
        self.lambda_layer = []
        for i in range(self.m_layers):
            self.lambda_layer.append(Lambda(lambda z: z))

        self.reshapor = [Reshape((1, self.n_x))]
        for i in range(self.m_layers-1):
            self.reshapor.append(Reshape((1, self.n_a[i])))
        # self.LSTM_cell=[]
        # for i in range(self.m_layers):
        #     self.LSTM_cell.append(LSTM(self.n_a[i], return_state=True))

        self.LSTM_cell = LSTM(self.n_a[0], return_state=True, name='lstm_{}_'.format(0))
        self.LSTM_cell1 = LSTM(self.n_a[1], return_state=True, name='lstm_{}_'.format(1))

        self.densor = Dense(self.n_y)  # , activation='softmax'

    def create_optimizer(self):
        self.opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
        return

    def compile_model(self, loss_function, model_metrics):
        self.model.compile(optimizer=self.opt, loss=loss_function, metrics=model_metrics)
        return

    def fit_model(self, x_train, y_train, x_val, y_val, epochs, plot_loss_value_obj, verbosity):
        #  for the validation set data I can use either validation_split=0.33 or validation_data=(Xval, Yval)
        train_inputs = [x_train]
        val_inputs = [x_val]
        for i in range(self.m_layers):
            train_inputs.append(self.a0_train[i])
            train_inputs.append(self.c0_train[i])
            val_inputs.append(self.a0_val[i])
            val_inputs.append(self.c0_val[i])

        # history = self.model.fit([x_train, self.a0_train, self.c0_train], list(y_train), epochs=epochs,
        #                          validation_data=([x_val, self.a0_val, self.c0_val], list(y_val)),
        #                          verbose=True, callbacks=[plot_loss_value_obj])
        history = self.model.fit(train_inputs, list(y_train), epochs=epochs,
                                 validation_data=(val_inputs, list(y_val)),
                                 verbose=True, callbacks=[plot_loss_value_obj])
        return history

    def load_data(self, path):
        return

    def save_model(self, file_path='', file_name='myModel'):
        Path(file_path).mkdir(parents=True, exist_ok=True)
        self.model.save('{}/{}.h5'.format(file_path, file_name))  # creates a HDF5 file 'my_model.h5'
        return

    def visualize_model(self, file_path='', file_name='myModel'):
        plot_model(self.model, to_file='{}/{}.png'.format(file_path, file_name), show_shapes=True)
        return

    def delete_model(self):
        del self.model  # deletes the existing model
        return

    def load_model(self, file_path='', file_name='myModel'):
        self.model = load_model('{}/{}.h5'.format(file_path, file_name))
        return

    #######################
    #######################
    def create_model(self):
        """
        Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

        Arguments:
        LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
        densor -- the trained "densor" from model(), Keras layer object
        n_values -- integer, number of unique values
        n_a -- number of units in the LSTM_cell
        Ty -- integer, number of time steps to generate

        Returns:
        inference_model -- Keras model instance
        """

        # Define the input of your model with a shape
        X = Input(shape=(self.Tx, self.n_x))
        X1= Input(shape=(self.Tx, self.n_a[0]))

        #  initial hidden state for the decoder LSTM
        a0 = []
        c0 = []
        a = []
        c = []
        for i in range(self.m_layers):
            a0.append(Input(shape=(self.n_a[i],), name='a0_{}'.format(i)))
            c0.append(Input(shape=(self.n_a[i],), name='c0_{}'.format(i)))
            a.append(a0[i])
            c.append(c0[i])

        # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
        outputs = []

        if self.Tx == 1:
            print('One-to-one (Tx=Ty=1) or One-to-many (Tx=1, Ty>1) :: Tx = {} , Ty = {}'.format(self.Tx, self.Ty))
            x = self.lambda_layer[0](X)
            # One-to-one (Tx=Ty=1), One-to-many (Tx=1, Ty>1)
            # Step 2: Loop over Ty and generate a value at every time step
            for t in range(self.Ty):
                x = self.reshapor[0](x)

                # do the process for each layer
                for l in range(self.m_layers):
                    # Step 2.A: Perform one step of LSTM_cell (≈1 line)
                    if l > 0:
                        input_layer_l = self.lambda_layer[l](a[l - 1])
                        input_layer_l = self.reshapor[l](input_layer_l)
                        a[l], _, c[l] = self.LSTM_cell1(inputs=input_layer_l, initial_state=[a[l], c[l]])
                    else:
                        # in case l=0
                        a[l], _, c[l] = self.LSTM_cell(inputs=x, initial_state=[a[l], c[l]])

                # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                # a[-1] last element of the list
                out = self.densor(a[-1])

                # Step 2.C: Append the prediction "out" to "outputs".
                outputs = outputs + [out]

                x = self.lambda_layer[0](out)

        else:
            if self.Ty == 1:
                print('Many-to-one  (Tx>1, Ty=1) :: Tx = {} , Ty = {}'.format(self.Tx, self.Ty))
                # Many-to-one  (Tx>1, Ty=1)
                for t in range(self.Tx):
                    # Step 2.A: Perform one step of LSTM_cell (≈1 line)
                    # do the process for each layer
                    for l in range(self.m_layers):
                        if l > 0:
                            input_layer_l = self.lambda_layer[l](a[l - 1])
                            input_layer_l = self.reshapor[l](input_layer_l)
                            a[l], _, c[l] = self.LSTM_cell1(inputs=input_layer_l, initial_state=[a[l], c[l]])
                        else:
                            x = self.lambda_layer[l](X[:, t, :])
                            x = self.reshapor[l](x)
                            a[l], _, c[l] = self.LSTM_cell(inputs=x, initial_state=[a[l], c[l]])

                # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                out = self.densor(a[-1])

                # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
                outputs = outputs + [out]

            elif self.Tx == self.Ty:
                print('Many-to-many (Tx=Ty>1) :: Tx = {} , Ty = {}'.format(self.Tx, self.Ty))
                # Many-to-many (Tx=Ty>1)
                # Step 2: Loop over Ty and generate a value at every time step
                for t in range(self.Ty):

                    for l in range(self.m_layers):
                        if l > 0:
                            input_layer_l = self.lambda_layer[l](a[l - 1])
                            input_layer_l = self.reshapor[l](input_layer_l)
                            a[l], _, c[l] = self.LSTM_cell1(inputs=input_layer_l, initial_state=[a[l], c[l]])
                        else:
                            x = self.lambda_layer[l](X[:, t, :])
                            x = self.reshapor[l](x)
                            a[l], _, c[l] = self.LSTM_cell(inputs=x, initial_state=[a[l], c[l]])

                    # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                    out = self.densor(a[-1])

                    # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
                    outputs = outputs + [out]

            else:
                print('Many-to-many (Tx!=Ty>1) :: Tx = {} , Ty = {}'.format(self.Tx, self.Ty))
                # Many-to-many (Tx!=Ty>1)
                for tx in range(self.Tx):
                    for l in range(self.m_layers):
                        if l > 0:
                            input_layer_l = self.lambda_layer[l](a[l - 1])
                            input_layer_l = self.reshapor[l](input_layer_l)
                            a[l], _, c[l] = self.LSTM_cell1(inputs=input_layer_l, initial_state=[a[l], c[l]])
                        else:
                            x = self.lambda_layer[l](X[:, tx, :])
                            x = self.reshapor[l](x)
                            a[l], _, c[l] = self.LSTM_cell(inputs=x, initial_state=[a[l], c[l]])

                out = self.densor(a[-1])
                # x = Lambda(lambda z: z)(out)
                x = self.lambda_layer[0](out)
                for ty in range(self.Ty):
                    if l > 0:
                        input_layer_l = self.lambda_layer[l](a[l - 1])
                        input_layer_l = self.reshapor[l](input_layer_l)
                        a[l], _, c[l] = self.LSTM_cell1(inputs=input_layer_l, initial_state=[a[l], c[l]])
                    else:
                        x = self.reshapor[l](x)
                        a[l], _, c[l] = self.LSTM_cell(inputs=x, initial_state=[a[l], c[l]])

                    # Step 3: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                    out = self.densor(a[-1])

                    # Step 4: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
                    outputs = outputs + [out]

                    x = self.lambda_layer[0](out)

        # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
        inputs = [X]
        for i in range(self.m_layers):
            inputs.append(a0[i])
            inputs.append(c0[i])

        self.model = Model(inputs=inputs, outputs=outputs)

        print('RNNKeras::create_model() finished')
        return

    def compute_prediction(self, x, data_type='test'):
        """
        Predicts the next value of values using the inference model.

        Arguments:
        inference_model -- Keras model instance for inference time
        Returns:
        results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
        indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
        """
        inputs = [x]
        if data_type == 'test':
            for i in range(self.m_layers):
                inputs.append(self.a0_test[i])
                inputs.append(self.c0_test[i])

            prediction = self.model.predict(inputs)
        elif data_type == 'validation':
            for i in range(self.m_layers):
                inputs.append(self.a0_val[i])
                inputs.append(self.c0_val[i])

            prediction = self.model.predict(inputs)
        elif data_type == 'training':
            for i in range(self.m_layers):
                inputs.append(self.a0_train[i])
                inputs.append(self.c0_train[i])

            prediction = self.model.predict(inputs)
        else:
            print("not implemented!")

        return prediction

    def evaluate_prediction(self, y_test, y_prediction):
        # y_test, y_prediction (Ty, m, n)
        evaluation = 0.0
        Ty = np.size(y_test, 0)
        m = np.size(y_test, 1)
        for i in range(Ty):
            for j in range(m):
                err_t_m = y_test[i, j, :] - y_prediction[i, j, :]
                evaluation = evaluation + np.sqrt(np.dot(err_t_m, err_t_m))
        evaluation = (1.0 / Ty) * (1.0 / m) * evaluation
        return evaluation

    def provide_model_summary(self):
        self.model.summary()
        return

    # def predict_motion_new(self, model, x_initializer, a_initializer, c_initializer):
    #     x_initializer = self.reshapor(x_initializer)
    #     pred = model.predict([x_initializer, a_initializer, c_initializer])
    #     return pred


#
# class PlotLosses(tf.keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.i = 0
#         self.x = []
#         self.losses = []
#         self.val_losses = []
#
#         self.fig = plt.figure()
#
#         self.logs = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         self.logs.append(logs)
#         self.x.append(self.i)
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.i += 1
#
#         plt.clf()
#         plt.plot(self.x, self.losses, label="loss")
#         plt.plot(self.x, self.val_losses, label="val_loss")
#         plt.legend()
#         plt.show()


def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Log-loss (cost function)"}
    if x in translations:
        return translations[x]
    else:
        return x


class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self, figsize=None, file_path='', file_name='myModel'):
        super(PlotLosses, self).__init__()
        self.figsize = figsize
        self.file_path = file_path
        self.file_name = file_name

    def on_train_begin(self, logs={}):

        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        # save the model
        # print(logs)
        self.save_model(epoch, val_loss=logs['val_loss'])

        plt.clf()
        for metric_id, metric in enumerate(self.base_metrics):
            if metric == 'loss':

                plt.plot(range(1, len(self.logs) + 1),
                         [log[metric] for log in self.logs],
                         label="training")
                if self.params['do_validation']:
                    plt.plot(range(1, len(self.logs) + 1),
                             [log['val_' + metric] for log in self.logs], '--',
                             label="validation")
                plt.title(translate_metric(metric))
                plt.xlabel('epoch')
                plt.legend(loc='center right')

        plt.pause(0.05)
        plt.tight_layout()
        plt.show()

    def save_model(self, epoch, val_loss):
        # creates a HDF5 file 'my_model_epochNumber_valLoss.h5'
        Path(self.file_path).mkdir(parents=True, exist_ok=True)
        self.model.save('{}/{}_{}_{}.h5'.format(self.file_path, self.file_name, epoch, val_loss))
        return


class Uncertainty:
    def __init__(self):
        return

    def compute_uncertainty(self, y, y_predict):
        """
        :param y:  real output (Ty x m_xx x n_y)
        :param y_predict: predicted output (Ty x m_xx x n_y)
        :return: y_uncertainty: uncertainty of the prediction (Ty x m_xx x n_y)
        """
        y_uncertainty = []
        for i in range(0, np.size(y, 0)):
            temp_uncertainty = []
            for j in range(0, np.size(y, 1)):
                temp_uncertainty2 = []
                for k in range(0, np.size(y,2)):
                    temp_uncertainty3 = (y[i, j, k] - y_predict[i, j, k]) ** 2
                    temp_uncertainty2.append(temp_uncertainty3)
                temp_uncertainty.append(temp_uncertainty2)
            y_uncertainty.append(temp_uncertainty)

        return np.array(y_uncertainty)
