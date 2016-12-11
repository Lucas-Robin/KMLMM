import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from data_reading import get_zip
from data_reading import get_cifar10

# fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

def baseline_model(input_shape=(1, 16, 16), num_classes=10, n_train=50000, n_test=10000):
	# create model
	model = Sequential()
	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def larger_model(input_shape=(1, 16, 16)):
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def my_model(input_shape):
    # create model
	model = Sequential()
	model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	sgd = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def zip_trained_model():
    (X_train, y_train, X_test, y_test) = get_zip()
    # one hot encode output
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    # build the model
    input_shape = X_train.shape[1:]
    model = baseline_model(input_shape)
    
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=200, verbose=2)
    
    return model
 

def main():
    # load data
    ZIP = False
    if ZIP:
        (X_train, y_train, X_test, y_test) = get_zip()
    else:
        (X_train, y_train, X_test, y_test) = get_cifar10()

    n_train = 50000
    n_test = 10000
    X_train = X_train[:n_train, :, :, :]
    y_train = y_train[:n_train]


    # normalize inputs from 0-255 to 0-1
    if not ZIP:
        X_train = X_train / 255
        X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    # build the model
    input_shape = X_train.shape[1:]

    #model = baseline_model(input_shape)
    #model = larger_model(input_shape)
    model = my_model(input_shape)

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test[:n_test, :, :, :], y_test[:n_test]), nb_epoch=100, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Error: %.2f%%" % (100-scores[1]*100))
    
if __name__ == "__main__":
    main()
