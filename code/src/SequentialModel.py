from abc import ABC, abstractmethod


class SequentialModel(ABC):
    def __init__(self, n_a, n_y, n_x, Tx, m, Ty):
        super().__init__()
        # hidden state dimensions of each RNN/LSTM cell
        self.n_a = n_a
        # output dimensions of each output layer
        self.n_y = n_y
        # input dimensions of input layer
        self.n_x = n_x
        # time series (sequence) length, the time horizon of prediction
        self.Tx = Tx
        self.Ty = Ty
        # number of training set
        self.m = m
        # RNN model
        self.model = None
        # RNN optimizer
        self.opt = None

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_optimizer(self):
        pass

    @abstractmethod
    def compile_model(self, model, opt_obj, loss_function, model_metrics):
        pass

    @abstractmethod
    def fit_model(self, model, Xtrain, Ytrain, a0, c0, Xval, Yval, epochs, plot_loss_value_obj):
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
    def compute_prediction(self, inference_model, x_initializer, a_initializer, c_initializer):
        pass

    @abstractmethod
    def evaluate_prediction(self, y_test, y_prediction):
        pass

    def return_model(self):
        return self.model
