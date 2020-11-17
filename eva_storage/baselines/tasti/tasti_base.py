"""
In this file, we implement the basic pipeline for tasti
Method will be:
1. Extract features using vgg16
2. Cluster based on FPF (furthest point first) algorithm
3. Use KNN-neighbors on all other datapoints to perform label propagation
"""

from torchvision import models
import random
from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import torch
from sklearn.metrics import pairwise_distances




class Tasti:

    def __init__(self, num_clusters = 100, k = 1):
        self.vgg16 = models.vgg16(pretrained = True).cuda() ## it is cuda by default
        self.knn = KNeighborsClassifier(n_neighbors = k)
        self.num_clusters = num_clusters
        self.batch_size = 10

    def resize_input(self, X):
        ## we need this to be 300x300 for vgg16
        X_new = []

        for i in range(len(X)):
            X_new.append(cv2.resize(X[i], (300,300)))

        return np.stack(X_new, axis = 0)

    def preprocess_input(self, X):
        """
        We will have to resize the input images,
        then convert to tensor (set device, permute to channel first)
        so that we can use vgg16
        :param X:
        :return:
        """
        X = self.resize_input(X)
        X = torch.tensor(X, device = 'cpu').float()
        X = X.permute(0,3,1,2)
        return X


    def run_vgg(self, X):
        X = self.preprocess_input(X)
        results = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i+self.batch_size].cuda()
            result = self.vgg16.features(batch)
            result = result.detach().permute(0,2,3,1).cpu().numpy()
            results.append(result)
        results = np.vstack(results)
        return results

    def run(self, X, num_clusters = None):
        """

        :param X: original data
        :return: indices of the key frames w.r.t the original data array, cluster label for every frame
        """
        ## convert X to tensor,
        x_features = self.run_vgg(X)
        ### need to flatten the x_features to 2d
        n, h, w, channels = x_features.shape
        x_features = x_features.reshape(n, h*w*channels)
        if num_clusters is not None:
            self.num_clusters = num_clusters

        points, indices = self.run_fpf(x_features, self.num_clusters)

        labels, mappings = self.run_neighbor_search(x_features, indices)
        return indices, labels, mappings


    def run_neighbor_search(self, points, key_frame_indices):
        """
        Determine the nearest neighbor among the key frames for every point
        We return the labels as the indices of the key frames in the original array (array with all the points)
        :param points:
        :param key_frame_indices:
        :return:
        """
        distance_matrix = pairwise_distances(points)
        labels = [] ## this is labels respect to the original images
        mappings = [] ## this is labels respect to the sampled images
        for i in range(len(points)):
            my_distances = distance_matrix[i]
            my_key_distances = my_distances[key_frame_indices]
            my_choice_index = np.argmin(my_key_distances)
            my_choice_index_prop = key_frame_indices[my_choice_index]
            labels.append(my_choice_index_prop)
            mappings.append(my_choice_index)
        return np.array(labels), np.array(mappings)


    def run_fpf(self, points, k):
        """
        Implementation of fpf algorithm
        :param points: data
        :param k: number of examples to choose
        :return: selected point values, and their corresponding indices
        """
        solution_set_indices = []
        rand_int = random.randint(0, len(points))
        solution_set_indices.append(rand_int)
        distance_matrix = pairwise_distances(points)
        for _ in range(k - 1):
            relevant_distance_arrays = []
            for solution_index in solution_set_indices:
                relevant_distance_arrays.append(distance_matrix[solution_index])
            ## we find the minimum distances
            relevant_distance_arrays = np.array(relevant_distance_arrays)
            updated_distances = relevant_distance_arrays.min(axis=0)
            #### we find the index of the maximum value and append to solution_set
            solution_set_indices.append(np.argmax(updated_distances))

        solution_set_indices = np.array(solution_set_indices)
        return points[solution_set_indices], solution_set_indices




