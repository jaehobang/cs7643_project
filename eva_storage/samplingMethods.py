
from abc import ABC, abstractmethod



class SamplingMethod(ABC):

    @abstractmethod
    def run(self, cluster_labels):
        pass



class FirstEncounterMethod(SamplingMethod):

    def __str__(self):
        return "First Encounter Method"

    def run(self, cluster_labels):
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

    def run(self, cluster_labels):
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


