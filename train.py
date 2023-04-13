from neural_network import NeuralNetwork
import utils

train_x_orig, train_y, test_x_orig, test_y, classes = utils.load_data()

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.0
test_x = test_x_flatten / 255.0

parameters = [[train_x.shape[0], 20, 'relu'],
              [20, 7, 'relu'],
              [7, 5, 'relu'],
              [5, 1, 'sigmoid']]

deep_nn = NeuralNetwork(parameters)
deep_nn.fit(X=train_x, Y=train_y, epochs=2500, learning_rate=0.001, print_cost=True)
deep_nn.plot_losses(learning_rate=0.001)
eval_acc = deep_nn.evaluate(test_x, test_y)
print("=========================================")
print(f"Test dataset accuracy: {eval_acc*100.:.4f} %")
