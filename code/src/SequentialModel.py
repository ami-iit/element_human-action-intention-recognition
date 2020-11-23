from abc import ABC, abstractmethod
import numpy as np


class SequentialModel(ABC):
    def __init__(self, n_a, n_y, n_x, Tx, Ty, m_train, m_val, m_test, activation_type):
        super().__init__()
        # hidden state dimensions of each RNN/LSTM cell
        self.n_a = n_a
        # output dimensions of each output layer
        self.n_y = n_y
        # input dimensions of input layer
        self.n_x = n_x
        # input time series (sequence) length
        self.Tx = Tx
        # output time series length (the horizon of prediction)
        self.Ty = Ty
        # number of training set
        self.m_train = m_train
        # number of validation set
        self.m_val = m_val
        # number of test set
        self.m_test = m_test
        # RNN model
        self.model = None
        # RNN optimizer
        self.opt = None
        # number of layers
        self.m_layers = len(n_a)

        self.a0_train =[]
        self.c0_train =[]
        self.a0_val   =[]
        self.c0_val   =[]
        self.a0_test  = []
        self.c0_test  = []

        for i in range(self.m_layers):
            self.a0_train.append(np.zeros((m_train, n_a[i])))
            self.c0_train.append(np.zeros((m_train, n_a[i])))
            self.a0_val.append(np.zeros((m_val, n_a[i])))
            self.c0_val.append(np.zeros((m_val, n_a[i])))
            self.a0_test.append(np.zeros((m_test, n_a[i])))
            self.c0_test.append(np.zeros((m_test, n_a[i])))

    @abstractmethod
    def create_model(self, recursive=False):
        pass

    @abstractmethod
    def create_optimizer(self):
        pass

    @abstractmethod
    def compile_model(self, model, opt_obj, loss_function, model_metrics):
        pass

    @abstractmethod
    def fit_model(self, x_train, y_train, x_val, y_val, epochs, plot_loss_value_obj, verbosity, recursive):
        pass

    @abstractmethod
    def save_model(self, model, file_path='', file_name='myModel'):
        pass

    @abstractmethod
    def visualize_model(self, model, file_path='', file_name='myModel'):
        pass

    @abstractmethod
    def load_model(self, file_path='', file_name='myModel'):
        pass

    @abstractmethod
    def compute_prediction(self, x_test, data_type):
        pass

    @abstractmethod
    def evaluate_prediction(self, y_test, y_prediction):
        pass

    def return_model(self):
        return self.model
