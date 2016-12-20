from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

#sgd = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=False)
#    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def ConvLayer(channels, size, input_shape=None):
    if input_shape == None:
        layer = Convolution2D(channels, size, size, border_mode='valid', activation='relu')
    else:
        layer = Convolution2D(channels, size, size, border_mode='valid', input_shape=input_shape, activation='relu')
    return layer

def PoolLayer():
    return MaxPooling2D(pool_size=(2, 2))

def DenseLayer(units):
    return Dense(units, activation='relu')

def OutLayer(num_classes):
    return Dense(num_classes, activation='softmax')

def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def dense_300(input_shape, num_classes=10):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(DenseLayer(300))
    model.add(OutLayer(num_classes))
    compile_model(model)
    return model

def dense_1000(input_shape, num_classes=10):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(DenseLayer(1000))
    model.add(OutLayer(num_classes))
    compile_model(model)
    return model

def dense_300_100(input_shape, num_classes=10):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(DenseLayer(300))
    model.add(DenseLayer(100))
    model.add(OutLayer(num_classes))
    compile_model(model)
    return model

def small_convnet(input_shape, num_classes=10):
    model = Sequential()
    # ZIP is 1x16x16
    model.add(ConvLayer(4, 5, input_shape=input_shape))
    # 4x12x12
    model.add(PoolLayer())
    # 4x6x6
    model.add(ConvLayer(12, 5))
    # 12x2x2
    model.add(ConvLayer(10, 2))
    # 10x1x1
    model.add(Flatten())
    model.add(OutLayer(num_classes))
    compile_model(model)
    return model

def lenet1(input_shape, num_classes=10, pooling=True, last_pooling=True):
    model = Sequential()
    model.add(ConvLayer(4, 5, input_shape=input_shape))
    if pooling:
        model.add(PoolLayer())
    model.add(ConvLayer(12, 5))
    if last_pooling:
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ConvLayer(10,5))
    model.add(Flatten())
    model.add(OutLayer(num_classes))
    compile_model(model)
    return model

def lenet5(input_shape, num_classes=10, pooling=True, last_pooling=True, filters=[6, 16, 120]):
    model = Sequential()
    # cifar10 | mnist
    # 3x32x32 | 1x28x28
    model.add(Convolution2D(filters[0], 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))  # layer C1
    # 6x28x28 | 6x24x24
    if pooling:
        model.add(MaxPooling2D(pool_size=(2, 2)))  # layer S2
    # 6x14x14 | 6x12x12
    model.add(Convolution2D(filters[1], 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))  # layer C3
    # 16x10x10 | 16x8x8
    if pooling:
        model.add(MaxPooling2D(pool_size=(2, 2)))  # layer S4
    # 16x5x5 | 16x4x4
    if last_pooling == pooling:  # zip and cifar10
        model.add(Convolution2D(filters[2], 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))  # layer C5
    else:
        model.add(Convolution2D(filters[2], 4, 4, border_mode='valid', input_shape=input_shape, activation='relu'))  # layer C5
    # 120x1x1 | 120x1x1
    model.add(Flatten())  # flatten output in order to use fully-connected layers next
    model.add(Dense(84, activation='relu'))  # layer F6
    model.add(Dense(num_classes, activation='softmax'))  # output layer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
