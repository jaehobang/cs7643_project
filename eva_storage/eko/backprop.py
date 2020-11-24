"""
In this file, we perform
1. vgg16 feature extraction
2. clustering with temporal constraints
3. silhouette optimal number of clusters derivation
4. training loop for optimizing vgg16 features

First pipeline specifics would be as follows:
1. vgg16 -> cluster -> derive n -> compute loss -> backprop -> vgg16 -> cluster -> compute loss -> REPEAT


"""
import sys

sys.argv = ['']
sys.path.append('/nethome/jbang36/eva_jaeho')



from eva_storage.temporalClusterModule import TemporalClusterModule
from eva_storage.featureExtractionMethods import DownSampleLanczosMethod, VGG16Method, VGG16Method_train
from eva_storage.jvc.preprocessor import Preprocessor
from eva_storage.samplingMethods import FastMiddleEncounterMethod
from eva_storage.clusterNumModule import ClusterNumLength

from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import silhouette_score

from torch.utils.data import DataLoader
import torch
import time
import os

import copy


class EKODataset(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, X, target):
        self.X = X
        self.y = target
        assert(len(self.X) == len(self.y))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            image -- in the format of original images
            target -- in the format of already processed by vgg16 network
        """
        image = self.X[index]
        ##image /= 255.0 ## normalize
        target = self.y[index]
        return torch.tensor(image, dtype=torch.float).permute(2, 0, 1), torch.tensor(target, dtype=torch.float)

    def __len__(self):
        return len(self.X)


class EKOBackprop:
    def __init__(self):
        self.step_size = 10
        self.tcm = TemporalClusterModule()
        self.clusterNumModule = ClusterNumLength()
        self.sampling_method = FastMiddleEncounterMethod()


    def create_dataloader(self, images, targets):
        ## cut the train and val to 9:1
        division_point = len(images) * 9 // 10
        train_images = images[:division_point]
        val_images = images[division_point:]
        train_targets = targets[:division_point]
        val_targets = targets[division_point:]

        train_dataset = EKODataset(train_images, train_targets)
        val_dataset = EKODataset(val_images, val_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle = False)
        val_dataloader = DataLoader(val_dataset, batch_size = 16, shuffle = False)

        return {'train': train_dataloader, 'val':val_dataloader}



    def train(self, images):
        """
        train the custom vgg16 network
        pipeline:
        TODO: for now we won't be using silhouette method since it takes too long to optimize
        1. outer train loop:
            - get_features -> cluster -> get_target -> model_parameter_update

        :param images:
        :return:
        """
        ### options
        num_epochs = 24
        save_directory = '/nethome/jbang36/eva_jaeho/data/vgg_backprop'
        downsample_method = VGG16Method_train()

        for i in range(num_epochs):
            images_downsampled = downsample_method.run(images, desired_vector_size = 100)
            images_features = downsample_method.image_features
            print(f"images downsampled shape: {images_downsampled.shape}, images_features shape: {images_features.shape}")

            connectivity = self.tcm.generate_connectivity_matrix(images_downsampled, number_of_neighbors = 3)

            cluster_num = self.clusterNumModule.run(images_downsampled)
            print(f"Cluster num is : {cluster_num}")

            cluster = AgglomerativeClustering(n_clusters=cluster_num, connectivity=connectivity, linkage='ward', compute_full_tree = True)
            cluster.fit(images_downsampled)
            labels = cluster.labels_

            rep_indices = self.sampling_method.run(labels)
            mapping = self.tcm.get_mapping(rep_indices, labels)


            target_features = images_features[mapping]
            print(f"target features shape is {target_features.shape}")
            dataloaders = self.create_dataloader(images, target_features)

            downsample_method.train(dataloaders)

            downsample_method.save(save_directory)
        return



    def run(self, images, load_directory = None):
        ## we don't input number of samples but rather output the number of samples used based on heuristics
        """
        Steps:
        1. Downsample the images -- for this the new method....
        :param images:
        :return: rep_indices wrt to the original images array, mapping
        """
        if load_directory is None or os.exists(load_directory) == False:
            load_directory = '/nethome/jbang36/eva_jaeho/data/vgg_backprop'
        downsample_method = VGG16Method_train(load_directory = load_directory)
        images_downsampled = downsample_method.run(images, desired_vector_size = 100)
        connectivity = self.tcm.generate_connectivity_matrix(images_downsampled)
        cluster_num = self.clusterNumModule(images_downsampled)

        cluster = AgglomerativeClustering(n_clusters=cluster_num, connectivity=connectivity,
                                          linkage='ward', compute_full_tree=True)
        cluster.fit(images_downsampled)
        labels = cluster.labels_

        ### we should return the rep indices, mapping
        sampling_method = FastMiddleEncounterMethod()
        rep_indices = sampling_method.run(labels)
        ### how do we get mapping?
        mapping = self.tcm.get_mapping(rep_indices, labels)

        return rep_indices, mapping



if __name__ == "__main__":
    ### let's try training the network
    from loaders.seattle_loader import SeattleLoader

    loader = SeattleLoader()
    #video_dir = os.path.join('/nethome/jbang36/eva_jaeho/data/seattle', 'seattle2_10000.mp4')
    video_dir = os.path.join('/nethome/jbang36/eva_jaeho/data/seattle', 'seattle2.mp4')
    images = loader.load_images(video_dir)
    eko = EKOBackprop()
    eko.train(images)









