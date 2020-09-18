"""
In this file, we investigate how much overlap in representative frames when we go from 1 - 5000
"""


import os
import sys
sys.argv=['']
sys.path.append('/nethome/jbang36/eva_jaeho')


import numpy as np
#import utils.helpers as helpers
import utils as helpers
from loaders.uadetrac_loader import UADetracLoader
from eva_storage.preprocessingModule import PreprocessingModule
from eva_storage.UNet import UNet
from eva_storage.clusterModule import ClusterModule
from filters.minimum_filter import FilterMinimum

from loaders.seattle_loader import SeattleLoader
from eva_storage.sampling_experiments.sampling_utils import *
from eva_storage.analysis.sampling_analysis_tools import *
from eva_storage.featureExtractionMethods import *
from eva_storage.temporalClusterModule import *
from eva_storage.samplingMethods import *
from eva_storage.sampling_experiments.noscope_sample_ssd_evaluation import *
from loaders.pp_loader import PPLoader


if __name__ == "__main__":
    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_10000.mp4'
    #video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'

    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    images = images[:100000]
    from eva_storage.featureExtractionMethods import DownSampleMeanMethod
    from eva_storage.temporalClusterModule import TemporalClusterModule

    ds = DownSampleMeanMethod()
    images_ds = ds.run(images, desired_vector_size=100)
    i_frames_list = []
    from eva_storage.samplingMethods import FastMiddleEncounterMethod

    sampling_method = FastMiddleEncounterMethod()
    import time

    st = time.perf_counter()
    tcluster = TemporalClusterModule()
    number_of_neighbors = 3
    linkage = 'ward'
    connectivity = tcluster.generate_connectivity_matrix(images_ds, number_of_neighbors)

    print('starting to cluster')
    for i in range(1,500):
        ac = AgglomerativeClustering(n_clusters=i, connectivity=connectivity, linkage=linkage)
        labels = ac.fit_predict(images_ds)
        rep_indices = sampling_method.run(labels)
        i_frames_list.append(rep_indices)
        if i % 10 == 0:
            print(f"finished {i}/5000...")

    print(f"total time taken to run is {time.perf_counter() - st} (seconds)")
    ### just in case we want to play around, we need to save the i_frames_list
    import pickle
    save_directory = '/nethome/jbang36/eva_jaeho/eva_storage/jvc/ideas/i_frames_list.txt'
    with open(save_directory, 'wb') as fp:
        pickle.dump(i_frames_list, fp)

    ### later you can load with
    #    with open(save_directory, 'rb') as fp:
    #        i_frames_list = pickle.load(fp)

    i_set = set()
    for i_frames in i_frames_list:
        i_set = i_set.union(i_frames)

    print(f"length of i_set is {len(i_set)}")
    results_directory = os.path.dirname(save_directory)
    results_directory = os.path.join(results_directory, 'results.txt')
    with open(results_directory, 'a+') as fp:
        fp.write(f"{__file__} length of i_frame_set is {len(i_set)} -- total time taken to process is {time.perf_counter() - st} (seconds) \n")


