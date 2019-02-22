import pandas as pd
import numpy as np
import math as math
"""
Naive Bayesian Classifier for continuous attribute
using Gaussian PDF

"""


class GaussianNaiveBayesClassifier:
    def __init__(self,data_set_path):
        self.data_set_path = data_set_path

        self.data_frame = pd.read_csv(self.data_set_path)
        self.no_of_instance = self.data_frame.shape[0]
        self.no_of_attributes = self.data_frame.shape[1] - 1
        self.data = self.data_frame.iloc[:, 0:self.no_of_attributes].values  # last attribute is class label
        self.class_label = self.data_frame.iloc[:, self.no_of_attributes:self.no_of_attributes + 1].values
        self.unique_class_label = np.unique(self.class_label).tolist()
        self.class_proab = {}
        self.means_by_class={}
        self.std_by_class = {}
        # print("data ",self.data)


        self.train()

    def train(self):
        #compute class probabilities
        for c in self.unique_class_label:
            self.class_proab[c] = (self.data_frame[ 'label'] == c).sum() / self.no_of_instance
            data_of_class = self.data_frame.loc[self.data_frame['label'] ==c]
            self.means_by_class[c] = data_of_class.mean(axis=0)
            self.std_by_class[c] = data_of_class.std(axis=0)


    @staticmethod
    def gaussian_proba(x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    #pattern is like input[5.1,3.5,1.4,0.2]
    def classify(self,pattern):
        probab ={}
        for c in self.unique_class_label:
            probab[c] = self.class_proab[c]
            for i in range(self.no_of_attributes):
                probab[c] *= self.gaussian_proba(pattern[i],self.means_by_class[c][i],self.std_by_class[c][i])

        return max(probab, key=lambda k: probab[k])

    def get_accuracy_of_training_data(self):
        print("No of instance ",self.no_of_instance)
        correct = 0
        for i in range(self.no_of_instance):
            pattern = self.data[i]
            pattern_output = self.classify(pattern)
            actual_output = self.class_label[i]

            if pattern_output == actual_output :
                correct += 1

        return (float)(correct / len(self.data)) * 100

bayesian_classifier = GaussianNaiveBayesClassifier("../../dataset/iris.csv")
print("Accuracy ",bayesian_classifier.get_accuracy_of_training_data())
print(bayesian_classifier.classify([5.1,3.5,1.4,0.2]))
