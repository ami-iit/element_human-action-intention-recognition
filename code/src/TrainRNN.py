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



class TrainRNN:
    def __init__(self, n_a, n_y, n_x, Tx, m, Ty):
        # print the tensorflow version
        print(tf.__version__)
        # hidden state dimensions of each RNN/LSTM cell
        self.n_a = n_a
        # output dimensions of each output layer
        self.n_y = n_y
        # input dimensions of input layer
        self.n_x = n_x
        # time series (sequence) length, the time horizon of prediction
        self.Tx = Tx
        self.Ty= Ty
        # number of training set
        self.m = m

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

    def create_model(self):
        """
        Implement the model

        Arguments:
        Tx -- length of the sequence in a corpus
        n_a -- the number of activations used in our model
        n_values -- number of unique values in the music data

        Returns:
        model -- a keras instance model with n_a activations
        """

        # Define the input layer and specify the shape
        X = Input(shape=(self.Tx, self.n_x))

        # Define the initial hidden state a0 and initial cell state c0
        # using `Input`, batch_size will be given later, while feeding the input values
        a0 = Input(shape=(self.n_a,), name='a0')
        c0 = Input(shape=(self.n_a,), name='c0')
        a = a0
        c = c0
        # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
        outputs = []

        # Step 2: Loop
        for t in range(self.Tx):
            # Step 2.A: select the "t"th time step vector from X.
            x = Lambda(lambda X: X[:, t, :])(X)
            # Step 2.B: Use reshapor to reshape x to be (1, n_y) (≈1 line)
            x = self.reshapor(x)
            # Step 2.C: Perform one step of the LSTM_cell
            a, _, c = self.LSTM_cell(inputs=x, initial_state=[a, c])
            # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
            out = self.densor(a)
            # Step 2.E: add the output to "outputs"
            outputs = outputs + [out]
            # outputs=(ouputs+out)

        # Step 3: Create model instance
        model = Model(inputs=[X, a0, c0], outputs=outputs)
        return model

    def create_optimizer(self):
        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
        return opt

    def compile_model(self, model, opt, loss_function, model_metrics):
        model.compile(optimizer=opt, loss=loss_function, metrics=model_metrics)
        return model

    def fit_model(self, model, Xtrain, Ytrain, a0, c0, Xval, Yval, epochs, plot_loss_value):
        #  for the validation set data I can use either validation_split=0.33 or validation_data=(Xval, Yval)
        history = model.fit([Xtrain, a0, c0], list(Ytrain), epochs=epochs, validation_data=(Xval, list(Yval)), verbose=1)# callbacks=[plot_loss_value]
        return model, history

    def load_data(self, path):
        return

    def visualize(self):
        return

    def save_model(self, model, file_path='models', file_name='myModel'):
        model.save('{}/{}.h5'.format(file_path, file_name))  # creates a HDF5 file 'my_model.h5'
        plot_model(model, to_file='model.png')

        return
    def delete_model(self, model):
        del model  # deletes the existing model
        return

    def load_model(self, file_path='models', file_name='myModel'):
        model = load_model('{}/{}.h5'.format(file_path, file_name))
        return model

    def visualize_model(self, model, file_path='models', file_name='myModel'):
        plot_model(model, to_file='{}/{}.png'.format(file_path, file_name))
        return

    #######################
    #######################
    def inference_model(self):
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
        x0 = Input(shape=(1, self.n_x))

        #  initial hidden state for the decoder LSTM
        a0 = Input(shape=(self.n_a,), name='a0')
        c0 = Input(shape=(self.n_a,), name='c0')
        a = a0
        c = c0
        x = Lambda(lambda z: z)(x0)

        ### START CODE HERE ###
        # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
        outputs = []

        # Step 2: Loop over Ty and generate a value at every time step
        for t in range(self.Ty):
            print('time: ', t)
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

        # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
        inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
        return inference_model

    def predict_motion(self, inference_model, x_initializer, a_initializer, c_initializer):
        """
        Predicts the next value of values using the inference model.

        Arguments:
        inference_model -- Keras model instance for inference time
        x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
        a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
        c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

        Returns:
        results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
        indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
        """

        ### START CODE HERE ###
        # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
        pred = inference_model.predict([x_initializer, a_initializer, c_initializer])

        return pred

    def predict_motion_new(self, model, x_initializer, a_initializer, c_initializer):
        x_initializer = self.reshapor(x_initializer)
        pred = model.predict([x_initializer, a_initializer, c_initializer])
        return pred


class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.clf()
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

class TrainRNN2():
    def __init__(self, n_a, n_y, n_x, Tx, m, Ty):
        # print the tensorflow version
        print(tf.__version__)
        print('TrainRNN2: ', TrainRNN2)
        # hidden state dimensions of each RNN/LSTM cell
        self.n_a = n_a
        # output dimensions of each output layer
        self.n_y = n_y
        # input dimensions of input layer
        self.n_x = n_x
        # time series (sequence) length, the time horizon of prediction
        self.Tx = Tx
        self.Ty= Ty
        # number of training set
        self.m = m

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

    def create_model(self):
        model = Sequential()
        model.add(LSTM(10, input_shape=(1, 1)))
        model.add(Dense(1, activation='linear'))
        return model

    def compile_model(self, model, opt, loss_function, model_metrics):
        model.compile(optimizer=opt, loss=loss_function, metrics=model_metrics)
        return model

    def fit_model(self, model, Xtrain, Ytrain, a0, c0, Xval, Yval, epochs, plot_loss_value):
        #  for the validation set data I can use either validation_split=0.33 or validation_data=(Xval, Yval)
        history = model.fit([Xtrain, a0, c0], list(Ytrain), epochs=epochs, validation_data=(Xval, list(Yval)), verbose=1)# callbacks=[plot_loss_value]
        return model, history
