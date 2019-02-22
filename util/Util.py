import numpy as np
import pandas as pd

from  classification.neuralnetwork.Pattern import Pattern


class NNPatternLoader:
    def __init__(self,path):
        self.data_frame = pd.read_csv(path)
        self.no_of_instance = self.data_frame.shape[0]
        self.no_of_attributes = self.data_frame.shape[1] - 1
        self.data = self.data_frame.iloc[:, 0:self.no_of_attributes].values  # last attribute is class label
        self.class_label = self.data_frame.iloc[:, self.no_of_attributes:self.no_of_attributes + 1].values
        self.unique_class_label = np.unique(self.class_label).tolist()
        self.no_of_class = len(self.unique_class_label)
        self.max_values = np.max(self.data, axis=0)  # max value in each column
        self.min_values = np.min(self.data, axis=0)

        self.data_set = []
        for row in range(self.data.shape[0]):
            input = ( self.data[row] -self.min_values) / (self.max_values - self.min_values)
            output = np.zeros(self.no_of_class)
            output_index = self.unique_class_label.index(self.class_label[row])
            output[output_index] =1
            pattern = Pattern(row,input,output)
            self.data_set.append(pattern)

#path = "../dataset/iris.csv"
#loader = NNPatternLoader(path)
#print(loader.data_set)