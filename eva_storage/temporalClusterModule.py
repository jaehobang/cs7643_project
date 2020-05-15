
from __future__ import annotations
import time
from sklearn.cluster import AgglomerativeClustering
from logger import Logger
from eva_storage.samplingMethods import MiddleEncounterMethod
from eva_storage.featureExtractionMethods import DownSampleMeanMethod, DownSampleSkippingMethod
import numpy as np

# Assume you have compressed images: images_compressed
# original_images -> images_compressed (output of the encoder network)




class TemporalClusterModule:

    def __init__(self, downsample_method= DownSampleSkippingMethod(), sampling_method=MiddleEncounterMethod()):
        self.ac = None
        self.sampling_method = sampling_method
        self.downsample_method = downsample_method
        self.vector_size = 100
        self.logger = Logger()

    def run(self, images, number_of_clusters, number_of_neighbors = 3):
        """
        :param image_compressed:
        :param fps:
        :return: sampled frames, corresponding cluster numbers for each instance
        """
        self.logger.info("Cluster module starting....")
        n_samples = len(images)

        start_time = time.perf_counter()

        image_compressed = self.downsample_method.run(images, self.vector_size)

        connectivity = self.generate_connectivity_matrix(image_compressed, number_of_neighbors)
        self.ac = AgglomerativeClustering(n_clusters=number_of_clusters, connectivity=connectivity,
                                          linkage='ward')
        labels = self.ac.fit_predict(image_compressed)
        self.logger.info(f"Time to fit {n_samples}: {time.perf_counter() - start_time} (sec)")
        self.logger.info(f"Sampling frames based on {str(self.sampling_method)} strategy")
        rep_indices = self.sampling_method.run(labels) ## we also need to get the mapping from this information


        return image_compressed[rep_indices], rep_indices, labels


    def generate_connectivity_matrix(self, image_compressed, number_of_neighbors = 5):
        index_list = [i for i in range(len(image_compressed))]
        index_list_np = np.array(index_list).reshape(-1, 1)
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(index_list_np, number_of_neighbors , mode='connectivity', include_self=True)

        return A.toarray()



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


    def reorder_cluster_labels(self, cluster_labels):
        """
        We are reordering cluster labels just for visualization purposes

        :param cluster_labels:
        :return:
        """
        mapping = cluster_labels
        ordered_mapping = np.zeros(len(mapping))
        last_seen_num = 0
        seen_num_dict = {}
        for i in range(len(mapping)):
            if mapping[i] not in seen_num_dict.keys():
                seen_num_dict[mapping[i]] = last_seen_num
                ordered_mapping[i] = last_seen_num
                last_seen_num += 1
            else:
                ordered_mapping[i] = seen_num_dict[mapping[i]]

        assert(len(set(ordered_mapping)) == len(set(mapping)))
        return ordered_mapping




    def plot_distribution(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(self.ac.labels_, bins=max(self.ac.labels_) + 1)
        plt.xlabel("Cluster Numbers")
        plt.ylabel("Number of datapoints")





if __name__ == "__main__":
    from loaders.uadetrac_loader import UADetracLoader
    from eva_storage.UNet import UNet

    loader = UADetracLoader()
    images = loader.load_cached_images(name = 'uad_train_images.npy', vi_name = 'uad_train_vi.npy')

    network = UNet()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    images_compressed, _ = network.execute(images, load_dir = '/nethome/jbang36/eva_jaeho/data/models/plain/unet_plain-epoch60.pth')

    cm = TemporalClusterModule()
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

