import random
import sys
import numpy as np

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
        The list sizes contains the number of neurons in the respective
        layers of the network.  
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost=cost

    def feedforward(self, a : list) -> list:
        """Return the output of the network if for a as input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    # implement grad_descent here

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))