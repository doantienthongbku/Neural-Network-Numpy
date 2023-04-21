from neural_network import NeuralNetwork
import utils
import numpy as np
import matplotlib.pyplot as plt

_, _, test_x_orig, test_y, classes = utils.load_data()

# Reshape the training and test examples
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
test_x = test_x_flatten / 255.0

sample_idx = 6

plt.imsave('sample.png', test_x_orig[sample_idx])
sample = test_x[:, sample_idx]
sample = sample.reshape(sample.shape[0], 1)
print("Ground truth: ", classes[test_y[0, sample_idx]])

parameters = [[test_x.shape[0], 20, 'relu'],
              [20, 7, 'relu'],
              [7, 1, 'sigmoid']]

deep_nn = NeuralNetwork(parameters)
deep_nn.load_checkpoint('checkpoint.npz')
output = deep_nn.predict(sample)

if output > 0.5:
    print(f"Predict: {classes[1]} - Probability: {output*100.:.4f} %")
else:
    print(f"Predict: {classes[0]} - Probability: {(1 - output)*100.:.4f} %")
