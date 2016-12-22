import numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.callbacks import Callback

from data_reading import get_zip
from data_reading import get_mnist
from data_reading import get_cifar10

from architectures import *

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

def zip_trained_model():
    (X_train, y_train, X_test, y_test) = get_zip()
    # one hot encode output
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    # build the model
    input_shape = X_train.shape[1:]
    model = dense_300(input_shape)
    
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
    
    return model

def accuracy(model, X, y):
    scores = model.evaluate(X, y, verbose=0)
    return scores[1]

class Performance(Callback):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.train = []

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        train_acc = accuracy(self.model, self.X_train, self.y_train)
        self.train.append(train_acc)
        print("train_acc = %f" % train_acc)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return


def main():
    # load data
    dataset = 1  # 0 = zip, 1 = mnist, 2 = cifar10
    
    if dataset == 0:
        (X_train, y_train, X_test, y_test) = get_zip()
    elif dataset == 1:
        (X_train, y_train, X_test, y_test) = get_mnist()
    else:
        (X_train, y_train, X_test, y_test) = get_cifar10()

    n_train = min(60000, X_train.shape[0])
    n_test = min(10000, X_test.shape[0])
    X_train = X_train[:n_train, :, :, :]
    y_train = y_train[:n_train]

    # normalize inputs from 0-255 to 0-1
    if dataset != 0:
        X_train = X_train / 255
        X_test = X_test / 255

    fraction_valid = 1/3
    
    n_valid = int(fraction_valid * n_train)
    indices_valid = np.array(np.random.choice(range(n_train), n_valid, replace=False))
    indices_train = np.array([i for i in range(n_train) if i not in indices_valid])

    X_valid = X_train[indices_valid, :, :, :]
    y_valid = y_train[indices_valid]
    X_train = X_train[indices_train, :, :, :]
    y_train = y_train[indices_train]
    n_train = n_train - n_valid

    pooling = dataset != 0
    last_pooling = dataset == 2

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_valid = np_utils.to_categorical(y_valid)
    y_test = np_utils.to_categorical(y_test)
    
    num_classes = y_train.shape[1]
    
    # build the model
    input_shape = X_train.shape[1:]

    models = []
    model = dense_300(input_shape)
    models.append(model)
    model = dense_1000(input_shape)
    models.append(model)
    model = dense_300_100(input_shape)
    models.append(model)
    model = lenet1(input_shape, pooling=pooling, last_pooling=last_pooling)
    models.append(model)
    if dataset == 2:
        filters = [32,64,256]
    else:
        filters = [6,16,120]
    model = lenet5(input_shape, pooling=pooling, last_pooling=last_pooling, filters=filters)
    models.append(model)

    for model in models:
        # Fit the model
        perf = Performance(X_train, y_train)
        hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=100, batch_size=200, verbose=2, callbacks = [perf])
        # Final evaluation of the model
        acc = accuracy(model, X_test, y_test)
        print("Error: %.2f%%" % (100 - acc * 100))
        print(perf.train)
        print(hist.history["val_acc"])
    
if __name__ == "__main__":
    main()
