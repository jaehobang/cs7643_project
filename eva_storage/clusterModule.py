
from __future__ import annotations
import time
from sklearn.cluster import AgglomerativeClustering
from logger import Logger
from abc import ABC, abstractmethod
import numpy as np

# Assume you have compressed images: images_compressed
# original_images -> images_compressed (output of the encoder network)




class ClusterModule:

    def __init__(self):
        self.ac = None
        self.logger = Logger()

    def run(self, image_compressed, number_of_clusters):
        """
        :param image_compressed:
        :param fps:
        :return: sampled frames, corresponding cluster numbers for each instance
        """
        self.logger.info("Cluster module starting....")
        n_samples = len(image_compressed)
        self.ac = AgglomerativeClustering(n_clusters=number_of_clusters)
        start_time = time.perf_counter()
        labels = self.ac.fit_predict(image_compressed)
        self.logger.info(f"Time to fit {n_samples}: {time.perf_counter() - start_time} (sec)")
        method = FirstEncounterMethod()
        self.logger.info(f"Sampling frames based on {str(method)} strategy")
        rep_indices = method.run(labels) ## we also need to get the mapping from this information


        return image_compressed[rep_indices], rep_indices, labels


    def get_mapping(self, rep_indices, cluster_labels):
        """
        When evaluating whether the clustering method is correct, we need a way to propagate to all the frames

        :param rep_indices: indices that are chosen as representative frames (based on the original images_compressed array
        :param cluster_labels: the cluster labels that are outputted by the algorithm (basically labels)
        :return: mapping from rep frames to all frames (basically tells us what each chosen frames is representing (normally used for evaluation purposes)
        """

        mapping = np.zeros(len(cluster_labels))
        for i, value in enumerate(rep_indices):
            corresponding_cluster_number = cluster_labels[value]
            members_in_cluster_indices = cluster_labels == corresponding_cluster_number
            mapping[members_in_cluster_indices] = i

        return mapping


    def plot_distribution(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(self.ac.labels_, bins=max(self.ac.labels_) + 1)
        plt.xlabel("Cluster Numbers")
        plt.ylabel("Number of datapoints")




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





if __name__ == "__main__":
    from loaders.uadetrac_loader import UADetracLoader
    from eva_storage.UNet import UNet

    loader = UADetracLoader()
    images = loader.load_cached_images(name = 'uad_train_images.npy', vi_name = 'uad_train_vi.npy')

    network = UNet()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    images_compressed, _ = network.execute(images, load_dir = '/nethome/jbang36/eva_jaeho/data/models/plain/unet_plain-epoch60.pth')

    cm = ClusterModule()
    rep_frames, rep_indices, all_cluster_labels = cm.run(images_compressed, number_of_clusters=len(images) / 30)


    mapping = cm.get_mapping(rep_indices, all_cluster_labels)

    """to make sure that mappings are generated correctly, the following property must hold
        1. for all i,j in indices of all_cluster_labels 
        if all_cluster_labels[i] == all_cluster_labels[j], then mapping[i] == mapping[j]
        2. to make sure at least and at most 1 frame is representing each cluster,
        len(rep_indices) == set(values of mapping)
    """

    print(len(rep_indices))
    print(len(set(mapping)))
    assert(len(rep_indices) == len(set(mapping)))


    for i in range(len(all_cluster_labels)):
        for j in range(len(all_cluster_labels)):
            if all_cluster_labels[i] == all_cluster_labels[j]:
                if mapping[i] != mapping[j]:
                    print(mapping[i])
                    print(mapping[j])
                    assert(False)

