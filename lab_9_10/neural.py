import numpy as np
import random

class CrossEntropyCost(object):
    @staticmethod
    def entropy_cost(a, y):
        """
        Return the cost associated with an output a and real output y.
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y):
        """
        Return the error delta from the output layer.  
        """
        return (a-y)

class Network(object):

    def __init__(self, sizes: list, cost=CrossEntropyCost):
        """
        The list sizes contains the number of neurons in the resigmoid_derivativeective
        layers of the network.  
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost=cost

    def feedforward(self, a : np.ndarray) -> np.ndarray:
        """Return the output of the network for a as input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def backprop(self, x, y):
        """Return grad for single sample of bias and weight"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]
        z_list = [] # before activation

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_list.append(z)
            activation = sigmoid(z) # weight between <0,1>
            activations.append(activation)

        # Count error
        delta = self.cost.delta(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backprop for hidden layers
        for l in range(2, self.num_layers):
            z = z_list[-l]
            sigmoid_derivative = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sigmoid_derivative
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def gradient_descent(self, training_data, max_steps, learning_rate):
        """Gradient descent classic version"""
        n = len(training_data)
        for step in range(max_steps):
            total_nabla_b = [np.zeros(b.shape) for b in self.biases]
            total_nabla_w = [np.zeros(w.shape) for w in self.weights]

            # Add all gradients
            for x, y in training_data:
                nabla_b, nabla_w = self.backprop(x, y)
                total_nabla_b = [tb + nb for tb, nb in zip(total_nabla_b, nabla_b)]
                total_nabla_w = [tw + nw for tw, nw in zip(total_nabla_w, nabla_w)]

            self.weights = [w - (learning_rate / n) * dw
                            for w, dw in zip(self.weights, total_nabla_w)]
            self.biases = [b - (learning_rate / n) * db
                           for b, db in zip(self.biases, total_nabla_b)]

#            print(f"step {step+1} finished")

    def evaluate(self, test_data):
        """Accuracy evaluation"""
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        return sum(int(pred == label) for pred, label in results)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))