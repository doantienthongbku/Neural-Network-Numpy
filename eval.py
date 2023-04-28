from neural_network import NeuralNetwork
import utils
import numpy as np
import matplotlib.pyplot as plt

_, _, test_x_orig, test_y, classes = utils.load_data()

# Reshape the training and test examples
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
test_x = test_x_flatten / 255.0

layers = [[test_x.shape[0], 20, 'relu'],
          [20, 7, 'relu'],
          [7, 1, 'sigmoid']]

deep_nn = NeuralNetwork(layers)
deep_nn.load_checkpoint('checkpoint.npz')
eval_acc = deep_nn.evaluate(test_x, test_y)
print("=========================================")
print(f"Test dataset accuracy: {eval_acc*100.:.4f} %")