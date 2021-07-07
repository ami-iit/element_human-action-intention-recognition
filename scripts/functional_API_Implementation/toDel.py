import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

for key in data_validation[0].keys():
    k = k+1
    if key=='l_shoe_fz':
        print(k)

batch_t=batch_t_test
batch_y_test=batch_y_test
batch_y_prediction=prediction
explanation = 'test (line), prediction (dashed lines)'

Ty = np.size(batch_y_test, 0)
# batch_x= list(range(1 , 5))
# swap the axis to have the shape ( m_test , Ty , n_y)
batch_y_prediction_reshaped = np.swapaxes(batch_y_prediction, 0, 1)
batch_y_test_reshaped = np.swapaxes(batch_y_test, 0, 1)

# print(np.size(batch_y_test_reshaped, 0))
# print(np.size(batch_y_test_reshaped, 1))
n_outputs = np.size(batch_y_test_reshaped, 2)
seq_length = np.size(batch_y_test_reshaped, 1)

if not batch_t:
    batch_t_test = list(range(0, np.size(batch_y_test_reshaped, 1)))
else:
    # cut to have (m x Ty) shape
    batch_t_test = batch_t

for j in range(0, n_outputs, 1):
    plt.figure()
    plt.title('time- {}'.format(explanation))
    for example_no in range(1, 50):
        plt.plot(batch_t_test[:], batch_y_test_reshaped[example_no, :, j])
        plt.plot(batch_t_test[:], batch_y_prediction_reshaped[example_no, :, j], '--')
    plt.ylabel('output {}'.format(str(j)))
    plt.xlabel('time')
    plt.show()
