import numpy as np
import h5py


def relu(z):
    return np.maximum(0., z)


def drelu(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def dsigmoid(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def softmax(z):
    e_Z = np.exp(z)
    A = e_Z / e_Z.sum(axis=0)
    return A


def dsoftmax(z, eps=1e-3):
    return (softmax(z + eps) - softmax(z - eps)) / (2 * eps)


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def save_checkpoint(parameters, filename):
    np.savez(filename, **parameters)
