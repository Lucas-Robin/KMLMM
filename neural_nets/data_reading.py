import numpy as np
import pickle
from keras.datasets import mnist

zip_train_path = "../zip_train.dat"
zip_test_path = "../zip_test.dat"
cifar10_path = "../cifar-10-batches-py/"

def get_lines(path):
    with open(path) as f:
        return f.readlines()

def read_floats(path, sep):
    lines = get_lines(path)
    n = len(lines)
    d = len([l for l in lines[0].split(sep) if len(l) > 0 and l != "\n"])
    data = np.zeros([n, d])
    for i in range(n):
        nums = lines[i].split(sep)
        for j in range(d):
            data[i, j] = float(nums[j])
    return data

def read_y_x(path, sep=" "):
    data = read_floats(path, sep)
    n = data.shape[0]
    y = np.zeros(n, dtype="int")
    y[:] = data[:, 0]
    x = data[:, 1:].reshape(n, 1, 16, 16)
    return (y, x)

def get_zip():
    (y_train, X_train) = read_y_x(zip_train_path)
    (y_test, X_test) = read_y_x(zip_test_path)
    return (X_train, y_train, X_test, y_test)

def get_cifar10_batch(batch):
    if batch == 0:
        filename = cifar10_path + "test_batch"
    else:
        filename = cifar10_path + "data_batch_" + str(batch)
    with open(filename, "br") as f:
        dic = pickle.load(f, encoding = "bytes")
        X = dic[b"data"].reshape(dic[b"data"].shape[0], 3, 32, 32)
        y = dic[b"labels"]
    return (X, y)

def get_cifar10():
    (X_test, y) = get_cifar10_batch(0)
    y_test = np.zeros(10000, dtype="int")
    y_test[:] = y
    X_train = np.zeros([50000, 3, 32, 32])
    y_train = np.zeros(50000, dtype="int")
    for i in range(5):
        (X, y) = get_cifar10_batch(i + 1)
        X_train[(i * 10000):((i+1) * 10000), :, :, :] = X
        y_train[(i * 10000):((i+1) * 10000)] = y
    return (X_train, y_train, X_test, y_test)

def get_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 1, 28, 28)
    X_test = X_test.reshape(10000, 1, 28, 28)
    return (X_train, y_train, X_test, y_test)
