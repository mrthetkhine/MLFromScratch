import pandas as pd
import numpy as np
"""
Assume dataset path is path of csv,
last column is class label in string.

"""
class KMeans:
    def __init__(self,data_set_path,k,max_iteration):
        self.data_set_path = data_set_path
        self.k = k
        self.max_iteration = max_iteration
        self.data_frame = pd.read_csv(self.data_set_path)
        self.no_of_instance = self.data_frame.shape[0]
        self.no_of_attributes = self.data_frame.shape[1]-1
        self.data = self.data_frame.iloc[:,0:self.no_of_attributes].values #last attribute is class label
        self.class_label = self.data_frame.iloc[:,:self.no_of_attributes+1].values
        self.max_values = np.max(self.data,axis=0) #max value in each column
        self.min_values = np.min(self.data,axis=0)

        print("Loading ",self.data_set_path)
        #print("data ",self.data)
        print("No of instance ", self.no_of_instance)
        print("No of attribute ",self.no_of_attributes)
        print("No of cluster ",self.k)
        print("Max iteration ",self.max_iteration)
        print("Max values ",self.max_values)
        print("Min values ",self.min_values)

        self.set_up_centroid()
        self.run()

    @staticmethod
    def euclidean_distance(x, y):
        return np.linalg.norm(x - y)

    def set_up_centroid(self):
        self.centroids = np.random.uniform(low=0., high=1., size=(self.k, self.no_of_attributes))
        self.centroids = self.centroids * (self.max_values - self.min_values) + self.min_values
        print("Initial Centroid ",self.centroids)

    def assign_memebers_to_clusters(self):
        self.clusters = [[] for c in range(self.k)]
        member_distance_to_centroid = np.zeros(self.k)
        # Assign members to nearest cluster
        for instance in range(self.no_of_instance - 1):
            distances = np.array([self.euclidean_distance(self.data[instance], c) for c in self.centroids])
            cluster_index = np.argmin(distances, axis=0)
            self.clusters[cluster_index].append(instance)  # just add member index
            member_distance_to_centroid[cluster_index] += distances[cluster_index]
        sse = 0
        for k in range(self.k):
            member_distance_to_centroid[k] /= len(self.clusters[k])
            sse += member_distance_to_centroid[k]
        print("SSE ",sse)

    def refine_centroid(self):
        self.centroids = np.zeros((self.k, self.no_of_attributes))
        for c in range(len(self.clusters)):
            for member in self.clusters[c]:
                for attr in range(self.no_of_attributes):
                    self.centroids[c][attr] += self.data[member][attr]
            self.centroids[c] = self.centroids[c] / len(self.clusters[c])

    def display_clusters(self):
        for c in range(len(self.clusters)):
            print("Cluster ",c)
            for member in self.clusters[c]:
                print("Member id", member, " Classs ",self.class_label[member])

    def run(self):
        for iter in range(self.max_iteration):
            print("Iteration ",iter)
            self.assign_memebers_to_clusters()
            self.refine_centroid()


kmeans = KMeans("../dataset/iris.csv",3,300)
kmeans.display_clusters()
