from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from sklearn import preprocessing

TIME_STEPS = 64     # same as the height of the image
INPUT_SIZE = 64     # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 40
CELL_SIZE = 50
LR = 0.001

seed = 7
np.random.seed(seed)

x = np.loadtxt("C:/Users/mrgoo/Desktop/551Project3/train_x.csv", delimiter=",")# load from text
x[x < 255] = 0
#x_test = np.loadtxt("C:/Users/mrgoo/Desktop/551Project3/test_x.csv", delimiter=",")
y = np.loadtxt("C:/Users/mrgoo/Desktop/551Project3/train_y.csv", delimiter=",")

x_train = x[:40000,]
x_test = x[40000:50000,]
y_train = y[:40000]
y_test = y[40000:]

x_train = x_train.reshape(-1, 64, 64)/255       # normalize
x_test = x_test.reshape(-1, 64, 64)/255         # normalize

le = preprocessing.LabelEncoder()
le.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
        21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81])

y_train = le.transform(y_train)
y_test = le.transform(y_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    x_batch = x_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(x_batch, y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX
    if step % 500 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)