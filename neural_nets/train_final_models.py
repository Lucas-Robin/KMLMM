import numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.callbacks import Callback

from data_reading import get_zip
from data_reading import get_mnist
from data_reading import get_cifar10

from architectures import *
from mnist import accuracy
from mnist import Performance

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

def train_model(dataset, epochs):
    # load data
    if dataset == 0:
        (X_train, y_train, X_test, y_test) = get_zip()
    elif dataset == 1:
        (X_train, y_train, X_test, y_test) = get_mnist()
    else:
        (X_train, y_train, X_test, y_test) = get_cifar10()

    n_train = min(60000, X_train.shape[0])
    n_test = min(10000, X_test.shape[0])

    # normalize inputs from 0-255 to 0-1
    if dataset != 0:
        X_train = X_train / 255
        X_test = X_test / 255

    pooling = dataset != 0
    last_pooling = dataset == 2

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    num_classes = y_train.shape[1]
    
    # build the model
    input_shape = X_train.shape[1:]

    if dataset == 2:
        filters = [32,64,256]
    else:
        filters = [6,16,120]
    model = lenet5(input_shape, pooling=pooling, last_pooling=last_pooling, filters=filters)
    
    # Fit the model
    perf = Performance(X_train, y_train)
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=200, verbose=2, callbacks = [perf])
    # Final evaluation of the model
    acc = accuracy(model, X_test, y_test)
    print("Error: %.2f%%" % (100 - acc * 100))
    print(perf.train)
    print(hist.history["val_acc"])
    
    # save model
    filename = "model%i.h5" % dataset
    model.save(filename)

def main():
    epochs = [40, 70, 30]
    for dataset in range(0, 3):
        train_model(dataset, epochs[dataset])

if __name__ == "__main__":
    main()
