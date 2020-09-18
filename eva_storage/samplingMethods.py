
from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix
import numpy as np
from timer import Timer

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


class IFrameConstraintMethod(SamplingMethod):
    def __init__(self, i_frames):
        self.i_frames = i_frames
        self.preliminary = MiddleEncounterMethod()

    def __str__(self):
        return "I Frame Constraint Method"

    def run(self, cluster_labels, X = None):
        middle_labels = self.preliminary.run(cluster_labels)
        ### now we have to modify these labels with i frames
        final_labels = []
        lower_bound = 0

        did_it = False
        for i, label in enumerate(middle_labels):
            for j in range(lower_bound, len(self.i_frames) - 1):
                if label >= self.i_frames[j] and label < self.i_frames[j+1]:
                    lower_bound = j
                    did_it = True
                    if label - self.i_frames[j] < self.i_frames[j+1] - label:
                        final_labels.append(self.i_frames[j])
                    else:
                        final_labels.append(self.i_frames[j+1])
                elif label >= self.i_frames[j] and j+1 == len(self.i_frames) - 1:
                    print('We have reached the end')
                    final_labels.append(self.i_frames[j+1])
                    did_it = True
            if not did_it:
                print(f"We didn't get the replacing index lower bound: {self.i_frames[lower_bound]}, {self.i_frames[lower_bound + 1]}, label: {label}")
            did_it = False

        print(self.i_frames[-10:])
        print(middle_labels[-10:])
        print(len(final_labels))
        print(len(middle_labels))
        assert(len(final_labels) == len(middle_labels))
        return final_labels


class FastMiddleEncounterMethod(SamplingMethod):
    """
    Optimized over Middle Encounter Method in 2 ways:
    1.
    """
    def __init__(self):
        self.timer = Timer()
        self.cluster_members_total_counts = {}

    def __str__(self):
        return "Fast Middle Encounter Method"

    def run(self, cluster_labels, X = None):
        #self.timer.tic()
        max_label = int(max(cluster_labels))

        for cluster_label in range(max_label + 1):
            self.cluster_members_total_counts[cluster_label] = 0

        for cluster_label in cluster_labels:
            self.cluster_members_total_counts[cluster_label] += 1


        ## first count how many there are
        final_indices_list = []
        indices_dict2 = {}
        ### we can use while loop to skip if we already have found the middle point


        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in indices_dict2.keys():
                indices_dict2[cluster_label] = 0
            elif indices_dict2[cluster_label] == -1:
                continue
            else: # not -1, already initialized
                indices_dict2[cluster_label] += 1

            if self.cluster_members_total_counts[cluster_label] // 2 == indices_dict2[cluster_label]:
                final_indices_list.append(i)

                indices_dict2[cluster_label] = -1
        """
        i = 0
        while i < len(cluster_labels):
            cluster_label = cluster_labels[i]
            if cluster_label not in indices_dict2.keys():
                indices_dict2[cluster_label] = 0
            elif indices_dict2[cluster_label] == -1: ## we are done with this label
                continue
            else: # not -1, already initialized
                indices_dict2[cluster_label] += 1

            if self.cluster_members_total_counts[cluster_label] // 2 == indices_dict2[cluster_label]:
                final_indices_list.append(i)

                indices_dict2[cluster_label] = -1
                i += self.cluster_members_total_counts[cluster_label] // 2 ## we can skip a lot of the frames but we should check for correctness
            i += 1
        """

        return final_indices_list

class MiddleEncounterMethod(SamplingMethod):

    def __init__(self):
        self.timer = Timer()
        self.cluster_members_total_counts = {}


    def __str__(self):
        return "Middle Encounter Method"

    def run(self, cluster_labels, X = None):
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in self.cluster_members_total_counts.keys():
                self.cluster_members_total_counts[cluster_label] = sum(cluster_labels == cluster_label)
        ## first count how many there are
        final_indices_list = []
        indices_dict2 = {}
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in indices_dict2.keys():
                indices_dict2[cluster_label] = 0
            elif indices_dict2[cluster_label] == -1:
                continue
            else: # not -1, already initialized
                indices_dict2[cluster_label] += 1

            if self.cluster_members_total_counts[cluster_label] // 2 == indices_dict2[cluster_label]:
                final_indices_list.append(i)

                indices_dict2[cluster_label] = -1


        return final_indices_list


##TODO: mapping seems to be wrong when I use this method, we need to make fixes
class MeanEncounterMethod(SamplingMethod):

    def __str__(self):
        return "Mean Encounter Method -- we search for frame that is closest to centroid of the cluster"

    def _find_rep(self, centroid, cluster_members):
        ## we assume we already have a centroid, and we already have cluster_members
        centroid = centroid.reshape(1, -1) ## need to make it shape of (1, 100)
        #print(cluster_members.shape)
        #print(centroid.shape)
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
            #print(i, class_)
            #print(centroids.shape)
            #print(centroids[i].shape)
            #print(X[cluster_labels == classes[i]].shape)
            #print("------------")
            rep_frame, rep_index_within_members = self._find_rep(centroids[i], X[cluster_labels == classes[i]])
            ## TODO: we need a method of backtracing which index the frame actually is at
            real_rep_index = self._search_original_index(cluster_labels == classes[i], rep_index_within_members)
            final_indices_list.append(real_rep_index)

        return final_indices_list


