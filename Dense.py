import numpy as np
import time

class Dense():
    def __init__(self, input_neurons, output_neurons):
        self.weights = np.random.normal(size=(input_neurons, output_neurons), loc=0, scale=np.sqrt(2/input_neurons))
        self.biases = np.zeros(shape=(output_neurons))
    def forward(self, input):
            self.input = input
            return np.add(np.dot(input, self.weights), self.biases)
    def backward(self, de_dy, learning_rate): 
            de_db = de_dy
            de_dw = np.dot(de_dy.T, np.mean(self.input, axis=0))
            de_dx = np.dot(de_dy, self.weights.T)
            self.weights = np.subtract(self.weights, np.multiply(learning_rate, de_dw))
            self.biases = np.subtract(self.biases, np.multiply(learning_rate, de_db))
            return de_dx


