"""

The preprocessor encapsulates the important i frame selection process + meta data generation / transfer processes

Interface should be run(images)

@Jaeho Bang


"""

import numpy as np
import os



from eva_storage.temporalClusterModule import TemporalClusterModule
from eva_storage.featureExtractionMethods import DownSampleMeanMethod
from eva_storage.samplingMethods import MiddleEncounterMethod

class Preprocessor:


    def __init__(self, children_save_dir = None):

        self.cluster = TemporalClusterModule(downsample_method=DownSampleMeanMethod(),
                                             sampling_method=MiddleEncounterMethod())
        self.children_save_dir = children_save_dir
        if not self.children_save_dir:
            self.children_save_dir = '/nethome/jbang36/eva_jaeho/data/clusters'
        self.children = None


    def get_tree(self):
        return self.children



    def run(self, images, video_filename):
        ### we need to compute the full tree and save the model
        number_of_neighbors = 3
        linkage = 'ward'

        cluster_count = len(images) // 100 ##
        _, rep_indices, all_cluster_labels = self.cluster.run(images, number_of_clusters=cluster_count,
                                                                  number_of_neighbors=number_of_neighbors,
                                                                  linkage=linkage, compute_full_tree = True)
        ## the algorithm automatically computes the full tree

        children = self.cluster.ac.children_
        assert(len(children) == len(images) - 1)
        save_dir = os.path.join(self.children_save_dir, video_filename)

        os.makedirs(save_dir, exist_ok=True)
        np.save(save_dir, children)
        self.children = children

        return rep_indices

