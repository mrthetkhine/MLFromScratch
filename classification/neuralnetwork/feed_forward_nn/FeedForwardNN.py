from __future__ import division
import random
import math
from classification.neuralnetwork.Pattern import Pattern
from classification.neuralnetwork.Data import Data
from util.Util import NNPatternLoader
import numpy as np

class FeedForwardNN:
    """ Multilayer feedforward neural network that use backpropgation as training algorithm"""

    def __init__(self, no_of_input, no_of_hidden, no_of_output, data):
        self.no_of_input = no_of_input
        self.no_of_hidden = no_of_hidden
        self.no_of_output = no_of_output
        self.data = data
        self.learning_rate = 0.15

        self.input = np.zeros(self.no_of_input)
        self.hidden = np.zeros(self.no_of_hidden)
        self.output = np.zeros(self.no_of_output)
        self.actual_target_values = np.zeros(self.no_of_output)

        #initialize weight
        self.weight_input_to_hidden = np.random.uniform(0, 1, (self.no_of_input,self.no_of_hidden))-0.5
        self.weight_hidden_to_output = np.random.uniform(0, 1, (self.no_of_hidden,self.no_of_output))-0.5

        #initialize bias
        self.bias_hidden = np.random.uniform(0, 1, self.no_of_hidden) -0.5
        self.bias_output = np.random.uniform(0, 1, self.no_of_output) -0.5

        #error
        self.error_hidden = np.zeros(self.no_of_hidden)
        self.error_output = np.zeros(self.no_of_output)

        self.mean_square_error = np.zeros(self.no_of_output)

    def train(self,n):
        mse =0
        for i in range(n):
            mse =self.train_epoch()
        return mse

    def train_epoch(self):
        """
        Train all entire set of training example set in one time.
        Pattern are drawn randomly for training.
        """
        pattern_indexs = np.arange(0,len(self.data.patterns),dtype=int)
        np.random.shuffle(pattern_indexs)

        self.error = 0

        for i in range(len(self.data.patterns)):

            #set random pattern as training example
            random_pattern_index = pattern_indexs[i]
            pattern = self.data.patterns[ random_pattern_index]
            self.input = pattern.input
            self.actual_target_values = pattern.output

            self.feed_forward()
            self.error += self.back_propagate()

        self.error /= len(self.data.patterns)
        print("Iteration ", i , " MSE ",self.error)
        return self.error

    def feed_forward(self):
        #clear out hidden neuron output value to zero
        self.hidden = np.zeros(self.no_of_hidden)

        #compute hidden layer
        self.hidden = np.dot(self.input,self.weight_input_to_hidden)+self.bias_hidden
        self.hidden = self.sigmoid(self.hidden)

        #compute output layer
        self.output = np.dot(self.hidden,self.weight_hidden_to_output)+self.bias_output
        self.output = self.sigmoid(self.output)

    def back_propagate(self):

        self.mean_square_error = 0

        #calcluate error of output
        self.error_output = self.output * (1-self.output)*(self.actual_target_values-self.output)
        self.mean_square_error = (np.square(self.actual_target_values - self.output)).mean(axis=None)

        #calculate error of hidden
        self.error_hidden *= self.hidden * (1- self.hidden)

        #update weight
        for hid in range(self.no_of_hidden):
            for out in range(self.no_of_output):
                self.weight_hidden_to_output[hid][out] += self.learning_rate*self.error_output[out]*self.hidden[hid]

        for inp in range(self.no_of_input):
            for hid in range(self.no_of_hidden):
                self.weight_input_to_hidden += self.learning_rate * self.error_hidden[hid]*self.input[inp]

        for out in range(self.no_of_output):
            self.bias_output[out] += self.learning_rate * self.error_output[out]

        for hid in range(self.no_of_hidden):
            self.bias_hidden[hid] += self.learning_rate * self.error_hidden[hid]


        return self.mean_square_error


    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def recall(self,inp):
        self.input = inp
        self.feed_forward()

    def get_accuracy_for_training(self):
        correct = 0
        for i in range(len(self.data.patterns)):
            pattern = self.data.patterns[i]
            self.recall(pattern.input)
            n_output = self.output
            act_output = pattern.output
            n_neuron = self.get_fired_neuron(n_output)
            a_neuron = self.get_fired_neuron(act_output)

            if n_neuron == a_neuron:
                correct += 1

        accuracy = (float)(correct / len(self.data.patterns)) * 100
        return accuracy

    def get_fired_neuron(self, output):
        return np.argmax(output)

"""Create test data """
path = "../../../dataset/iris.csv"

loader = NNPatternLoader(path)
data = Data(loader.data_set, loader.unique_class_label)
nn = FeedForwardNN(4, 10, 3, data)
mse = nn.train(320)
#accuracy = rbf.get_accuracy_for_training()
#print("Total accuracy is ", accuracy)
print("Last MSE ", mse)
print("Accuracy on training set ",nn.get_accuracy_for_training())