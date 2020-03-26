from SequentialModel import SequentialModel
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
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
        self.reshapor = Reshape((1, self.n_x))
        self.LSTM_cell = LSTM(self.n_a, return_state=True)
        self.densor = Dense(self.n_y) #, activation='softmax'

    def create_optimizer(self):
        self.opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
        return

    def compile_model(self, loss_function, model_metrics):
        self.model.compile(optimizer=self.opt, loss=loss_function, metrics=model_metrics)
        return

    def fit_model(self, x_train, y_train, x_val, y_val, epochs, plot_loss_value_obj, verbosity):
        #  for the validation set data I can use either validation_split=0.33 or validation_data=(Xval, Yval)
        history = self.model.fit([x_train, self.a0_train, self.c0_train], list(y_train), epochs=epochs,
                                validation_data=([x_val, self.a0_val, self.c0_val], list(y_val)),
                                verbose=verbosity, callbacks=[plot_loss_value_obj])
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

        #  initial hidden state for the decoder LSTM
        a0 = Input(shape=(self.n_a,), name='a0')
        c0 = Input(shape=(self.n_a,), name='c0')
        a = a0
        c = c0

        ### START CODE HERE ###
        # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
        outputs = []

        if self.Tx == 1:
            print('One-to-one (Tx=Ty=1) or One-to-many (Tx=1, Ty>1) :: Tx = {} , Ty = {}'.format(self.Tx, self.Ty))
            x = Lambda(lambda z: z)(X)
            # One-to-one (Tx=Ty=1), One-to-many (Tx=1, Ty>1)
            # Step 2: Loop over Ty and generate a value at every time step
            for t in range(self.Ty):
                x = self.reshapor(x)
                # Step 2.A: Perform one step of LSTM_cell (≈1 line)
                a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])

                # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                out = self.densor(a)

                # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
                outputs = outputs + [out]

                # Step 2.D:
                # Select the next value according to "out",
                # Set "x" to be the one-hot representation of the selected value
                # See instructions above.
                x = Lambda(lambda z: z)(out)

        else:
            if self.Ty == 1:
                print('Many-to-one  (Tx>1, Ty=1) :: Tx = {} , Ty = {}'.format(self.Tx, self.Ty))
                # Many-to-one  (Tx>1, Ty=1)
                for t in range(self.Tx):
                    x = Lambda(lambda z: z[:, t, :])(X)
                    x = self.reshapor(x)
                    # Step 2.A: Perform one step of LSTM_cell (≈1 line)
                    a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])

                # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                out = self.densor(a)

                # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
                outputs = outputs + [out]

            elif self.Tx == self.Ty:
                print('Many-to-many (Tx=Ty>1) :: Tx = {} , Ty = {}'.format(self.Tx, self.Ty))
                # Many-to-many (Tx=Ty>1)
                # Step 2: Loop over Ty and generate a value at every time step
                for t in range(self.Ty):
                    x = Lambda(lambda z: z[:, t, :])(X)
                    x = self.reshapor(x)
                    # Step 2.A: Perform one step of LSTM_cell (≈1 line)
                    a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])

                    # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                    out = self.densor(a)

                    # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
                    outputs = outputs + [out]

            else:
                print('Many-to-many (Tx!=Ty>1) :: Tx = {} , Ty = {}'.format(self.Tx, self.Ty))
                # Many-to-many (Tx!=Ty>1)
                for tx in range(self.Tx):
                    x = Lambda(lambda z: z[:, tx, :])(X)
                    x = self.reshapor(x)
                    # Step 2: Perform one step of LSTM_cell
                    a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])

                out = self.densor(a)
                x = Lambda(lambda z: z)(out)
                for ty in range(self.Ty):
                    x = self.reshapor(x)
                    # Step 3: Perform one step of LSTM_cell (≈1 line)
                    a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])

                    # Step 3: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
                    out = self.densor(a)

                    # Step 4: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
                    outputs = outputs + [out]

                    x = Lambda(lambda z: z)(out)

        # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
        self.model = Model(inputs=[X, a0, c0], outputs=outputs)

        print('RNNKeras::create_model() finished')
        return

    def compute_prediction(self, x_test):
        """
        Predicts the next value of values using the inference model.

        Arguments:
        inference_model -- Keras model instance for inference time
        Returns:
        results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
        indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
        """
        prediction = self.model.predict([x_test, self.a0_test, self.c0_test])
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
        evaluation = (1.0/Ty) * (1.0/m) * evaluation
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
        # self.save_model(epoch, val_loss=logs['val_loss'])

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
        # creates a HDF5 file 'my_model_epchNumber_valLoss.h5'
        Path(self.file_path).mkdir(parents=True, exist_ok=True)
        self.model.save('{}/{}_{}_{}.h5'.format(self.file_path, self.file_name, epoch, val_loss))
        return
