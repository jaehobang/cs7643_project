"""
This file is used to evaluate the compression method of the pipeline
@Jaeho Bang
"""


import numpy as np
from loaders.uadetrac_loader import UADetracLoader
from eva_storage.preprocessingModule import PreprocessingModule
from eva_storage.UNet import UNet
from eva_storage.clusterModule import ClusterModule
from filters.minimum_filter import FilterMinimum


def get_rep_frames(images:np.ndarray, labels, image_cluster_labels):
    visited_cluster_nums = set()
    n_samples, height, width ,channels = images.shape
    rep_images = np.zeros(shape = (max(image_cluster_labels) + 1, height, width, channels))
    rep_labels = np.zeros(shape = (max(image_cluster_labels) + 1))

    for i in range(len(image_cluster_labels)):
        if image_cluster_labels[i] not in visited_cluster_nums:
            visited_cluster_nums.add(image_cluster_labels[i])
            rep_images[image_cluster_labels[i]] = images[i]
            rep_labels[image_cluster_labels[i]] = labels[i]

    return rep_images, rep_labels




if __name__ == "__main__":

    ### deprecated... moved to ipynb file
    """
    loader = LoaderUADetrac()
    images = loader.load_cached_images()
    labels = loader.load_cached_labels()
    video_start_indices = loader.get_video_start_indices()
    pm = PreprocessingModule()
    seg_images = pm.run(images,video_start_indices)
    unet = UNet()
    unet.train(images, seg_images)
    unet_compressed_images, unet_segmented_images = unet.execute()
    cm = ClusterModule()
    image_cluster_labels = cm.run(unet_compressed_images)

    rep_images, rep_labels = get_rep_frames(images, labels['vehicle'], image_cluster_labels)
    ## TODO: Chose the representative frames... now need to do custom_code with filters




    # init a filter instance that is trained on all images
    fm_everyframe = FilterMinimum()
    fm_everyframe.train(images, labels['vehicle'])

    fm_repframe = FilterMinimum()
    fm_repframe.train(rep_images, rep_labels)
    """

    


