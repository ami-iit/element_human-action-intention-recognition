from __future__ import print_function
import sys
import os
sys.path.append("/Users/kdarvish/Desktop/Bracelet/Algorithm/nnom/scripts")
print(sys.path)

from tensorflow.keras import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from nnom import *
from DatasetGenerationForPrediction import Data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


save_dir = 'keras_trained_model.h5'

def load_bracelet_data():

	seq_length = 10  # 20 100
	Tx = 10
	Tx0 = 0  # this is used to prepare the data, not a part of rnn
	Ty = 1  #
	Ty0 = 9  # this is used to prepare the data, not a part of rnn : normally (Tx0+Tx)
	n_a = [4]  # [3, 2]#[32, 16]  # 5 32
	n_y = 1
	n_x = 2
	# m_train = 200  # 40 200
	# m_val = 100  # 2 20
	# m_test = 1  # 1 5
	epochs = 2  # 100  # 20 50
	# models_path = 'models/models_2020_3_26_19_10_54'# sin signal learned
	# models_path = 'models/models_2020_4_14_19_46_56'  # amplitude-modulation signal learned
	doTraining = True
	doScaling = False
	learnUncertainty = False
	predictFuture = True
	seed_number = 0

	# use these for regression problem

	recursive = False
	data_type = 'amplitude-modulation'
	verbosity = False
	read_data_from_file = True
	feature_list = ['time_epoch_braclet', 'rssiinCollision', 'braceletsdistance']
	x_feature = [0, 1]
	y_feature = [2]
	problem_type = 'regression'

	data = Data()
	if read_data_from_file:
		# train
		train_data_raw = data.read_from_file('dataset/train_val_test_Data_session_02_03/training')
		data.bracelet_data_augmentation(train_data_raw)
		train_data_raw_updated = data.update_bracelet_data(train_data_raw, problem_type)
		batch_t_train, batch_data_train = data.prepare_data_batches(feature_list, train_data_raw_updated,
																	seq_length)
		if doScaling:
			scalar_ = MinMaxScaler(feature_range=(0, 1))
			s0 = np.size(batch_data_train, 0)
			s1 = np.size(batch_data_train, 1)
			s2 = np.size(batch_data_train, 2)
			batch_data_train_reshaped = np.reshape(batch_data_train, (s0 * s1, s2))

			scalar_.fit(batch_data_train_reshaped)
			batch_data_train_reshaped = scalar_.transform(batch_data_train_reshaped)
			batch_data_train = np.reshape(batch_data_train_reshaped, (s0, s1, s2))

		batch_x_train, batch_y_train = data.prepare_data(batch_data_train, Tx, Ty, Tx0, Ty0, x_feature, y_feature)

		# validation
		val_data_raw = data.read_from_file('dataset/train_val_test_Data_session_02_03/validation')
		data.bracelet_data_augmentation(val_data_raw)

		val_data_raw_updated = data.update_bracelet_data(val_data_raw, problem_type)
		batch_t_val, batch_data_val = data.prepare_data_batches(feature_list, val_data_raw_updated,
																seq_length)

		if doScaling:
			s0 = np.size(batch_data_val, 0)
			s1 = np.size(batch_data_val, 1)
			s2 = np.size(batch_data_val, 2)
			batch_data_val_reshaped = np.reshape(batch_data_val, (s0 * s1, s2))
			batch_data_val_reshaped = scalar_.transform(batch_data_val_reshaped)
			batch_data_val = np.reshape(batch_data_val_reshaped, (s0, s1, s2))

		batch_x_val, batch_y_val = data.prepare_data(batch_data_val, Tx, Ty, Tx0, Ty0, x_feature, y_feature)

		# test
		test_data_raw = data.read_from_file('dataset/train_val_test_Data_session_02_03/test')
		test_data_raw_updated = data.update_bracelet_data(test_data_raw, problem_type)
		batch_t_test, batch_data_test = data.prepare_data_batches(feature_list, test_data_raw_updated,
																  seq_length)
		if doScaling:
			s0 = np.size(batch_data_test, 0)
			s1 = np.size(batch_data_test, 1)
			s2 = np.size(batch_data_test, 2)
			batch_data_test_reshaped = np.reshape(batch_data_test, (s0 * s1, s2))
			batch_data_test_reshaped = scalar_.transform(batch_data_test_reshaped)
			batch_data_test = np.reshape(batch_data_test_reshaped, (s0, s1, s2))

		batch_x_test, batch_y_test = data.prepare_data(batch_data_test, Tx, Ty, Tx0, Ty0, x_feature, y_feature)

		m_train = batch_x_train.shape[0]
		m_val = batch_x_val.shape[0]
		m_test = batch_x_test.shape[0]
	# m_test = 1

	X_train, Y_train = data.prepare_time_series_data_as_image(batch_x_train, batch_y_train)
	X_val, Y_val = data.prepare_time_series_data_as_image(batch_x_val, batch_y_val)
	X_test, Y_test   = data.prepare_time_series_data_as_image(batch_x_test, batch_y_test)

	# X_train, Y_train =batch_x_train, batch_y_train
	# X_val, Y_val = batch_x_val, batch_y_val
	# X_test, Y_test   = batch_x_test, batch_y_test

	return X_train, Y_train, X_test, Y_test




def build_model(x_shape):

    inputs = Input(shape=x_shape)
    # x = Flatten()(inputs)
    # x = Dense(10, activation='relu')(inputs)
    x = Conv2D(8, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="relu")(inputs)
    # x = BatchNormalization()(x)
    # x = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D((1, 1), strides=(1, 1), padding="same")(x)
    # x = Dropout(0.2)(x)

    # x = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    # x = Dropout(0.2)(x)

    # x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    # x = MaxPool2D((1, 1), strides=(1, 1), padding="same")(x)
    # x = Dropout(0.2)(x)
    x = Flatten()(x)

    # x = Dense(32, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    # x = Dense(20, activation='sigmoid')(x)
    # x = Dense(10, activation='relu')(x)
    # x = Dense(1, activation='relu')(x)
    # predictions = ReLU()(x)
    # x = Dropout(0.2)(x)
    x = Dense(1)(x)
    # predictions = Softmax()(x)
    predictions = x

    model = Model(inputs=inputs, outputs=predictions)
    return model

def train(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=50):

    opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mse'])
    model.summary()

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=False)

    save_model(model, save_dir)
    del model
    tf.keras.backend.clear_session()
    return history

if __name__ == "__main__":
    #physical_devices = tf.config.experimental.list_physical_devices("GPU")
    #if(physical_devices is not None):
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    epochs = 50
    num_classes = 1

    # The data, split between train and test sets:
    # (x_train, y_train), (x_test_original, y_test_original) = mnist.load_data()
    x_train, y_train, x_test_original, y_test_original = load_bracelet_data()

    x_test = x_test_original
    y_test = y_test_original
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    # y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    # y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # reshape to 4 d becaue we build for 4d?
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    print('x_train shape:', x_train.shape)

    # quantize the range to q7
    # x_test = x_test.astype('float32')/255
    # x_train = x_train.astype('float32')/255
    print("data range", x_test.min(), x_test.max())

    # build model
    model = build_model(x_test.shape[1:])

    # train model
    history = train(model, x_train, y_train, x_test.copy(), y_test.copy(), epochs=epochs, batch_size=x_train.shape[0])

    # -------- generate weights.h (NNoM model) ----------
    # get the best model
    model = load_model(save_dir)

    # only use 1000 for test
    # x_test = x_test[:1000]
    # y_test = y_test[:1000]
    # generate binary dataset for NNoM validation, 0~1 -> 0~127, q7
    generate_test_bin(x_test*127, y_test, name='test_data.bin')

    # evaluate in Keras (for comparision)
    scores = evaluate_model(model, x_test, y_test)

    # generate NNoM model, x_test is the calibration dataset used in quantisation process
    generate_model(model,  x_test, format='hwc', name="weights.h")
    y_test_estimated = model.predict(x_test)
    print('y_test:', y_test.shape)
    print('y_test_estimated:', y_test_estimated.shape)

    t_test = list(range(0, np.size(y_test, 0)))
    plt.figure()
    plt.title('predictions and real values')
    plt.plot(t_test, y_test, 'go', markersize=10)
    plt.plot(t_test, y_test_estimated, 'r+', markersize=8)
    plt.ylabel('output')
    plt.xlabel('batches')
    plt.show()


    # --------- for test in CI ----------
    # build NNoM
    # os.system("scons")

    # do inference using NNoM
    # cmd = ".\mnist.exe" if 'win' in sys.platform else "./mnist"
    # os.system(cmd)
    # try:
    #     # get NNoM results
    #     result = np.genfromtxt('result.csv', delimiter=',', dtype=np.int, skip_header=1)
    #     result = result[:,0]        # the first column is the label, the second is the probability
    #     label = y_test_original[:len(y_test)].flatten()     # use the original numerical label
    #     acc = np.sum(result == label).astype('float32')/len(result)
    #     if (acc > 0.5):
    #         print("Top 1 Accuracy on Keras %.2f%%" %(scores[1]*100))
    #         print("Top 1 Accuracy on NNoM  %.2f%%" %(acc *100))
    #         return 0
    #     else:
    #         raise Exception('test failed, accuracy is %.1f%% < 80%%' % (acc * 100.0))
    # except:
    #     raise Exception('could not perform the test with NNoM')
