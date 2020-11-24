"""
This file determines how many clusters we will use
"""
from abc import ABC, abstractmethod
import numpy as np


class ClusterNumModule(ABC):

    @abstractmethod
    def run(self, features):
        pass


class ClusterNumFix(ClusterNumModule):

    def run(self, features, num=10):
        return num


class ClusterNumLength(ClusterNumModule):


    def run(self, features):
        total_length = len(features)
        if total_length < 10:
            return total_length
        elif total_length < 100:
            return total_length // 10
        elif total_length < 1000:
            return total_length // 100
        else:
            return total_length // 1000


class ClusterNumSilhouette(ClusterNumModule):

    def __init__(self):
        self.step_size = 10

    def run(self, features):
        from eva_storage.temporalClusterModule import TemporalClusterModule
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        tcm = TemporalClusterModule()
        connectivity = tcm.generate_connectivity_matrix(features)
        cluster = AgglomerativeClustering(n_clusters=1, connectivity=connectivity,
                                          linkage='ward', compute_full_tree=True)

        scores = []
        exit_point = min(10000, len(features))
        for n in range(2, exit_point, self.step_size):
            cluster.n_clusters = n
            cluster.fit(features)
            labels = cluster.labels_
            scores.append(silhouette_score(features, labels))
            if n % 1000 == 2:
                print(f"Inside silhouette, current n is {n}")

        best_n = np.argmax(scores) * self.step_size
        return best_n