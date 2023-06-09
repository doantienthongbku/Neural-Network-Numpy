import numpy as np
import utils
import matplotlib.pyplot as plt

# input_nodes, output_nodes, activation
# layers = [[10 * 10, 10, 'relu'],
#               [10, 20, 'relu'],
#               [20, 2, 'softmax']]


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.W = {}
        self.B = {}
        self.Z = {}
        self.A = {}
        self.dW = {}
        self.dB = {}
        self.dZ = {}
        self.dA = {}
        self.activation = ['non']
        self.loss_list = []
        self.acc_list = []
        
        self._initialize_weights()

    def _initialize_weights(self):
        for idx, (in_nodes, out_nodes, act_fn) in enumerate(self.layers):
            self.W[idx + 1] = np.random.randn(out_nodes, in_nodes) * np.sqrt(2 / in_nodes)
            self.B[idx + 1] = np.zeros(shape=(out_nodes, 1))
            self.activation.append(act_fn)
            
    def load_checkpoint(self, filename):
        checkpoint = np.load(filename, allow_pickle=True)
        self.W = checkpoint['W'].item()
        self.B = checkpoint['B'].item()
        self.activation = checkpoint['activation']
    
        print("Checkpoint loaded successfully!")


    def forward(self, X):
        self.A[0] = X
        L = len(self.layers)
        for idx in range(1, L + 1):
            self.Z[idx] = self.W[idx] @ self.A[idx - 1] + self.B[idx]

            if (self.activation[idx] == "relu"):
                self.A[idx] = utils.relu(self.Z[idx])
            elif (self.activation[idx] == "sigmoid"):
                self.A[idx] = utils.sigmoid(self.Z[idx])
            elif (self.activation[idx] == "softmax"):
                self.A[idx] = utils.softmax(self.Z[idx])

    def loss_function(self, Y):
        m = Y.shape[1]
        L = len(self.layers)
        loss = - (1 / m) * np.sum(Y * np.log(self.A[L]) + (1 - Y) * np.log(1 - self.A[L]))
        return loss

    def backward(self, Y):
        m = Y.shape[1]
        L = len(self.layers)
        self.dA[L] = - (np.divide(Y, self.A[L]) - np.divide(1 - Y, 1 - self.A[L]))
        for idx in reversed(range(1, L + 1)):

            if (self.activation[idx] == "relu"):
                self.dZ[idx] = self.dA[idx] * utils.drelu(self.Z[idx])
            elif (self.activation[idx] == "sigmoid"):
                self.dZ[idx] = self.dA[idx] * utils.dsigmoid(self.Z[idx])
            elif (self.activation[idx] == "softmax"):
                self.dZ[idx] = self.dA[idx] * utils.dsoftmax(self.Z[idx])

            self.dW[idx] = (1 / m) * (self.dZ[idx] @ self.A[idx - 1].T)
            self.dB[idx] = (1 / m) * self.dZ[idx].sum(axis=1, keepdims=True)
            self.dA[idx - 1] = (self.W[idx].T @ self.dZ[idx])

    def optimizer(self, learning_rate):
        L = len(self.layers)
        for idx in range(1, L + 1):
            self.W[idx] = self.W[idx] - learning_rate * self.dW[idx]
            self.B[idx] = self.B[idx] - learning_rate * self.dB[idx]

    def predict(self, X):
        L = len(self.layers)
        self.forward(X)
        return float(self.A[L][0])
    
    def evaluate(self, X, Y):
        L = len(self.layers)
        m = Y.shape[1]
        self.forward(X)
        predictions = np.zeros((1, m))
        for i in range(self.A[L].shape[1]):
            if self.A[L][0, i] > 0.5:
                predictions[0, i] = 1
            else:
                predictions[0, i] = 0
        accuracy = np.sum((predictions == Y) / m)
        return accuracy

    def fit(self, X, Y, epochs, learning_rate, print_cost=False):
        best_loss = np.inf
        for i in range(epochs):
            self.forward(X)
            loss = self.loss_function(Y)
            self.backward(Y)
            self.optimizer(learning_rate=learning_rate)
            accuracy = self.evaluate(X, Y)

            if print_cost and i % 100 == 0 or i == epochs - 1:
                print(f"Iteration {i:4}: Loss - {loss:.7f}, Accuracy - {accuracy*100.:.4f}")
            if i % 100 == 0 or i == epochs - 1:
                self.loss_list.append(loss)
                self.acc_list.append(accuracy)
                if loss < best_loss:
                    best_loss = loss
                    self.save_checkpoint('checkpoint')
                
    def save_checkpoint(self, filename):
        checkpoint = {'W': self.W, 'B': self.B, 'activation': self.activation}
        np.savez(filename, **checkpoint)

    def plot_figs(self):
        plt.plot(self.loss_list)
        plt.plot(self.acc_list)
        plt.ylabel('loss - accuracy')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Losses - Accuracy")
        
        plt.savefig('figure.png')
    
