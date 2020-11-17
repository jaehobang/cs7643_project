"""
In this file, we input the best methods found so far to do experiments
11/15/2020


"""

from eva_storage.temporalClusterModule import TemporalClusterModule
from eva_storage.featureExtractionMethods import DownSampleLanczosMethod, VGG16Method
from eva_storage.jvc.preprocessor import Preprocessor
from eva_storage.samplingMethods import FastMiddleEncounterMethod
from sklearn.cluster import AgglomerativeClustering
import numpy as np


class EKOBest:
    def __init__(self):
        pass


    def find_nearest(self, np_arr, val):
        idx = (np.abs(np_arr - val)).argmin()
        return np_arr[idx], idx


    def run(self, images):
        ## we don't input number of samples but rather output the number of samples used based on heuristics
        """
        Steps:
        1. Downsample the images -- for this the new method....
        :param images:
        :return: rep_indices wrt to the original images array, mapping
        """
        downsample_method = DownSampleLanczosMethod()
        images_downsampled = downsample_method.run(images, desired_vector_size = 100)
        tcm = TemporalClusterModule()
        connectivity = tcm.generate_connectivity_matrix(images_downsampled)
        cluster = AgglomerativeClustering(n_clusters=None, distance_threshold = 10, connectivity = connectivity, linkage = 'ward', compute_full_tree = True)
        cluster.fit(images_downsampled)
        distances = cluster.distances_.T
        ### okay now we computed the distances, the threshold we will use is
        mean = np.mean(distances)
        _, idx = self.find_nearest(distances, mean)
        n_samples = idx + 1 ## we add 1 because n_clusters start with 1
        ## we perform the clustering operation again
        cluster = AgglomerativeClustering(n_clusters=n_samples, connectivity=connectivity, linkage = 'ward')
        cluster.fit(images_downsampled)
        labels = cluster.labels_
        ### we should return the rep indices, mapping
        sampling_method = FastMiddleEncounterMethod()
        rep_indices = sampling_method.run(labels)
        ### how do we get mapping?
        mapping = tcm.get_mapping(rep_indices, labels)

        return rep_indices, mapping









