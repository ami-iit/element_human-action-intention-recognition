import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path



def compile_and_fit(model, window, plot_losses, patience=2, MAX_EPOCHS=20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')


    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.Accuracy()])
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping
                            # , plot_losses
                                   ])

    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss : {}'.format(model.name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    return history


def save_model(model, file_path='', file_name='myModel'):
    Path(file_path).mkdir(parents=True, exist_ok=True)
    model.save('{}/{}.h5'.format(file_path, file_name))  # creates a HDF5 file 'my_model.h5'
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


class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self, figsize=None, file_path='', file_name='myModel'):
        super(PlotLosses, self).__init__()
        self.figsize = figsize
        self.file_path = file_path
        self.file_name = file_name
        plt.show()

    def on_train_begin(self, logs=None):
        print('self.params', self.params)
        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        # save the model

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


    def save_model(self, epoch, val_loss):
        # creates a HDF5 file 'my_model_epochNumber_valLoss.h5'
        Path(self.file_path).mkdir(parents=True, exist_ok=True)
        self.model.save('{}/{}_{}_{}.h5'.format(self.file_path, self.file_name, epoch, val_loss))
        return

