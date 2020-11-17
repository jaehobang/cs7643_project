"""
In this file, we perform
1. vgg16 feature extraction
2. clustering with temporal constraints
3. silhouette optimal number of clusters derivation
4. training loop for optimizing vgg16 features
"""

"""
In this file, we implement and test the method using silhouette (derivation of n clusters)
@Jaeho Bang
"""

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
from sklearn.metrics import silhouette_score


class EKOBackprop:
    def __init__(self):
        self.step_size = 10

    def run(self, images):
        ## we don't input number of samples but rather output the number of samples used based on heuristics
        """
        Steps:
        1. Downsample the images -- for this the new method....
        :param images:
        :return: rep_indices wrt to the original images array, mapping
        """
        downsample_method = VGG16Method()
        images_downsampled = downsample_method.run(images, desired_vector_size = 100)
        tcm = TemporalClusterModule()
        connectivity = tcm.generate_connectivity_matrix(images_downsampled)
        cluster = AgglomerativeClustering(n_clusters=1, connectivity=connectivity,
                                          linkage='ward', compute_full_tree=True)

        scores = []
        exit_point = min(10000, len(images_downsampled))
        for n in range(2, exit_point, self.step_size):
            cluster.n_clusters = n
            cluster.fit(images_downsampled)
            labels = cluster.labels_
            scores.append( silhouette_score(images_downsampled, labels) )

        best_n = np.argmax(scores) * self.step_size
        cluster.n_clusters = best_n
        cluster.fit(images_downsampled)
        labels = cluster.labels_

        ### we should return the rep indices, mapping
        sampling_method = FastMiddleEncounterMethod()
        rep_indices = sampling_method.run(labels)
        ### how do we get mapping?
        mapping = tcm.get_mapping(rep_indices, labels)

        return rep_indices, mapping











