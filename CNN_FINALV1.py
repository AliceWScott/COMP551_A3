import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, AveragePooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn import preprocessing
import pandas as pd

seed = 7
np.random.seed(seed)

# hyper parameters
batch_num = 256
epochs_num = 300
filter_sizes = [32, 32, 32]
nns = 512

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), padding='same', input_shape=(64, 64, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    for fs in filter_sizes:
        model.add(Convolution2D(fs, (1, 1)))
        model.add(Convolution2D(fs, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Convolution2D(fs, (1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(nns))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr=5e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # load from text
    x_train = np.loadtxt("./train_x.csv", delimiter=",")
    x_test = np.loadtxt("./test_x.csv", delimiter=",")
    y_train = np.loadtxt("./train_y.csv", delimiter=",")
    # x_train = x_train[:5000,]
    # x_test = x[45001:50000,]
    # y_train = y_train[:5000]
    # y_test = y[45000:]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255

    le = preprocessing.LabelEncoder()
    le.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
            21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81])

    y_train = le.transform(y_train)
    y_train = np_utils.to_categorical(y_train)

    x_train = x_train.reshape(-1,64,64,1)
    x_test = x_test.reshape(-1,64,64,1)



    #Run the model
    model = cnn_model()
    print(model.summary())


    print('Training ----------')
    model.fit(x_train, y_train, batch_size=batch_num, epochs=epochs_num, verbose=1, validation_split=0.1)


    # print('\nTesting ----------')
    # loss, accuracy = model.evaluate(x_test, y_test)
    # print('\ntest loss: ', loss)
    # print('\ntest accuracy: ', accuracy)

    y_prediction = model.predict(x_test)
    y_prediction = np.argmax(y_prediction, axis=1)
    pd.DataFrame(y_prediction, columns=['y_prediction']).to_csv('./cnn_pred.csv')
