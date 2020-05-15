
from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix
import numpy as np


class SamplingMethod(ABC):

    @abstractmethod
    def run(self, cluster_labels, X = None):
        pass



class FirstEncounterMethod(SamplingMethod):

    def __str__(self):
        return "First Encounter Method"

    def run(self, cluster_labels, X = None):
        label_set = set()
        indices_list = []
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in label_set:
                label_set.add(cluster_label)
                indices_list.append(i)

        return indices_list

class MiddleEncounterMethod(SamplingMethod):

    def __str__(self):
        return "Middle Encounter Method"

    def run(self, cluster_labels, X = None):
        cluster_members_total_counts = {}
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in cluster_members_total_counts.keys():
                cluster_members_total_counts[cluster_label] = sum(cluster_labels == cluster_label)

        ## first count how many there are
        final_indices_list = []
        indices_dict2 = {}
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in indices_dict2.keys():
                indices_dict2[cluster_label] = 1
            elif indices_dict2[cluster_label] == -1:
                continue
            else: # not -1, already initialized
                indices_dict2[cluster_label] += 1

            if cluster_members_total_counts[cluster_label] // 2 == indices_dict2[cluster_label]:
                final_indices_list.append(i)

                indices_dict2[cluster_label] = -1

        return final_indices_list



class MeanEncounterMethod(SamplingMethod):

    def __str__(self):
        return "Mean Encounter Method -- we search for frame that is closest to centroid of the cluster"

    def _find_rep(self, centroid, cluster_members):
        ## we assume we already have a centroid, and we already have cluster_members
        centroid = centroid.reshape(1, -1) ## need to make it shape of (1, 100)
        dm = distance_matrix(cluster_members, centroid)
        index = np.argmin(dm) ## we find the member that is closest to centroid
        return cluster_members[index], index


    def _search_original_index(self, true_false_array, rep_index_within_members):
        true_count = 0
        for i,val in enumerate(true_false_array):
            if val:
                true_count += 1
            if true_count - 1 == rep_index_within_members:
                return i

        print("_search_original_index, something must be wrong, should not reach here")
        raise ArithmeticError


    def run(self, cluster_labels, X = None):
        if X is None:
            print("For this method, you need to supply X")
            raise ValueError

        from sklearn.neighbors.nearest_centroid import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X, cluster_labels)
        # clf.centroids_ (300, 100) clf.classes_ (300, ) gives the centroid for each corresponding cluster

        centroids = clf.centroids_
        classes = clf.classes_
        final_indices_list = []


        for i, class_ in enumerate(classes):
            rep_frame, rep_index_within_members = self._find_rep(centroids[i], X[X == classes[i]])
            ## TODO: we need a method of backtracing which index the frame actually is at
            real_rep_index = self._search_original_index(X == classes[i], rep_index_within_members)
            final_indices_list.append(real_rep_index)

        return final_indices_list



class MeanEncounterMethod(SamplingMethod):

    def __str__(self):
        return "Mean Encounter Method -- we search for frame that is closest to centroid of the cluster"

    def _find_rep(self, centroid, cluster_members):
        ## we assume we already have a centroid, and we already have cluster_members
        centroid = centroid.reshape(1, -1) ## need to make it shape of (1, 100)
        dm = distance_matrix(cluster_members, centroid)
        index = np.argmin(dm) ## we find the member that is closest to centroid
        return cluster_members[index], index


    def _search_original_index(self, true_false_array, rep_index_within_members):
        true_count = 0
        for i,val in enumerate(true_false_array):
            if val:
                true_count += 1
            if true_count - 1 == rep_index_within_members:
                return i

        print("_search_original_index, something must be wrong, should not reach here")
        raise ArithmeticError


    def run(self, cluster_labels, X = None):
        if X is None:
            print("For this method, you need to supply X")
            raise ValueError

        from sklearn.neighbors.nearest_centroid import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X, cluster_labels)
        # clf.centroids_ (300, 100) clf.classes_ (300, ) gives the centroid for each corresponding cluster

        centroids = clf.centroids_
        classes = clf.classes_
        final_indices_list = []


        for i, class_ in enumerate(classes):
            rep_frame, rep_index_within_members = self._find_rep(centroids[i], X[X == classes[i]])
            ## TODO: we need a method of backtracing which index the frame actually is at
            real_rep_index = self._search_original_index(X == classes[i], rep_index_within_members)
            final_indices_list.append(real_rep_index)

        return final_indices_list
