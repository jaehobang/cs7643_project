"""
In this file, we will write a script for measuring the speed of end-to-end pipeline execution
"""
import sys
import os
import time
import random



sys.argv = ['']
sys.path.append('/nethome/jbang36/eva_jaeho')



from loaders.uadetrac_loader import UADetracLoader
from loaders.seattle_loader import SeattleLoader
from loaders.ssd_loader import SSDLoader
from loaders.pp_loader import PPLoader
from loaders.uadetrac_label_converter import UADetracConverter

from eva_storage.sampling_experiments.sampling_utils import sample3_middle
from eva_storage.sampling_experiments.sampling_utils import evaluate_with_gt5
from eva_storage.sampling_experiments.noscope_sample_ssd_evaluation import set_frame_count, get_rep_indices_noscope
from eva_storage.jvc.preprocessor import Preprocessor
from eva_storage.jvc.jvc import JVC

import sklearn.metrics as metrics


def load_seattle2():
    seattle_loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'
    seattle_images = seattle_loader.load_images(directory)
    label_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_annotations'
    seattle_labels, seattle_boxes = seattle_loader.load_labels(label_directory,
                                                               relevant_classes=['car', 'van', 'others'])
    seattle_gt_labels = {}
    seattle_gt_labels['car'] = UADetracConverter.convert2limit_queries(seattle_labels, car=1)
    #seattle_gt_labels['others'] = UADetracConverter.convert2limit_queries(seattle_labels, others=1)
    #seattle_gt_labels['van'] = UADetracConverter.convert2limit_queries(seattle_labels, bus=1)
    for label in seattle_gt_labels.keys():
        seattle_gt_labels[label] = UADetracConverter.replaceNoneWithZeros(seattle_gt_labels[label])

    return ('seattle2', seattle_images, seattle_gt_labels, seattle_boxes)


def udf_pipeline(file_descriptor):
    st = time.perf_counter()
    seattle_loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'
    seattle_images = seattle_loader.load_images(directory)
    file_descriptor.write(f"udf data loading time: {time.perf_counter() - st} (sec)\n")
    video_name = 'seattle2'
    query_name = 'car=1'
    ssd_loader = SSDLoader()
    results = ssd_loader.predict(seattle_images, cuda = True)
    file_descriptor.write(f"  NO SAMPLING TOTAL TIME: {time.perf_counter() - st} (sec)\n")
    return results

def sample_pipeline(file_descriptor, num_samples):
    st = time.perf_counter()
    seattle_loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'
    seattle_images = seattle_loader.load_images(directory)
    seattle_images = seattle_images[:num_samples]
    file_descriptor.write(f"udf data loading time: {time.perf_counter() - st} (sec)\n")
    ssd_loader = SSDLoader()
    results = ssd_loader.predict(seattle_images, cuda = True)
    file_descriptor.write(f"  SAMPLING TOTAL TIME for {num_samples} IMAGES: {time.perf_counter() - st} (sec)\n")




def pp_pipeline(file_descriptor):
    st = time.perf_counter()
    seattle_loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'
    seattle_images = seattle_loader.load_images(directory)
    file_descriptor.write(f"pp data loading time: {time.perf_counter() - st} (sec)\n")

    video_name = 'seattle2'
    query_name = 'car=1'
    st = time.perf_counter()
    pp_loader = PPLoader(video_name, query_name)
    results = pp_loader.predict(seattle_images, cache=False)
    file_descriptor.write(f"pp prediction loading time: {time.perf_counter() - st} (sec)\n")
    return results


def us_pipeline(number_of_samples, file_descriptor):

    st = time.perf_counter()
    seattle_loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'
    seattle_images = seattle_loader.load_images(directory)
    file_descriptor.write(f"us data loading time: {time.perf_counter() - st} (sec)\n")
    st = time.perf_counter()
    skip_rate = len(seattle_images) // number_of_samples
    sampled_images = seattle_images[::skip_rate]
    video_name = 'seattle2'
    query_name = 'car=1'
    file_descriptor.write(f"us sampling time: {time.perf_counter() - st} (sec)\n")
    st = time.perf_counter()
    pp_loader = PPLoader(video_name, query_name)
    results = pp_loader.predict(sampled_images, cache = False)
    file_descriptor.write(f"us prediction loading time: {time.perf_counter() - st} (sec)\n")

    return results



def noscope_pipeline(number_of_samples, file_descriptor):
    st = time.perf_counter()
    seattle_loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'
    seattle_images = seattle_loader.load_images(directory)
    file_descriptor.write(f"noscope data loading time: {time.perf_counter() - st} (sec)\n")
    st = time.perf_counter()
    t_diff = 1
    delta_diff = 100
    noscope_rep_indices, mapping = set_frame_count(number_of_samples, seattle_images, t_diff, delta_diff)
    sampled_images = seattle_images[noscope_rep_indices]
    file_descriptor.write(f"noscope sampling time: {time.perf_counter() - st} (sec)\n")
    st = time.perf_counter()
    video_name = 'seattle2'
    query_name = 'car=1'
    pp_loader = PPLoader(video_name, query_name)
    results = pp_loader.predict(sampled_images, cache = False)
    file_descriptor.write(f"noscope prediction time: {time.perf_counter() - st} (sec)\n")

    return results

def jvc_pipeline(number_of_samples, file_descriptor):
    st = time.perf_counter()
    jvc = JVC()
    dataset_name = 'seattle'
    video_name = 'jvc_seattle2_100000'
    hierarchy_name = 'seattle2_100000'
    sampled_images = jvc.decode(dataset_name, video_name, hierarchy_name, sample_count = number_of_samples)
    file_descriptor.write(f"jvc data loading time: {time.perf_counter() - st} (sec)\n")
    st = time.perf_counter()
    video_name = 'seattle2'
    pp_loader = PPLoader(video_name, 'car=1')
    results = pp_loader.predict(sampled_images, cache = False)
    file_descriptor.write(f"jvc prediction time: {time.perf_counter() - st} (sec)\n")

    return results


if __name__ == "__main__":
    ## we should try for seattle2_100000 (100k frames) video
    """
    Steps:
    1. Load the dataset
    2. """
    number_of_samples = [100,500,1000,5000]
    base_directory = '/nethome/jbang36/eva_jaeho/data/benchmark_results/cs7643'
    os.makedirs(base_directory, exist_ok = True)

    save_directory = os.path.join(base_directory, 'speed_sampling_vs_nosampling.txt')
    file_descriptor = open(save_directory, 'a+')

    #st = time.perf_counter()
    #udf_pipeline(file_descriptor)
    #file_descriptor.write(f"total time taken for udf is {time.perf_counter() - st} (seconds) \n")
    #st = time.perf_counter()
    for number in [100, 500, 1000, 5000]:
        sample_pipeline(file_descriptor, number)
    #file_descriptor.write(f"total time taken for pp is {time.perf_counter() - st} (seconds) \n")



    """
    
    for number in number_of_samples:
        st = time.perf_counter()
        us_pipeline(number, file_descriptor)
        us_time = time.perf_counter() - st

        st = time.perf_counter()
        noscope_pipeline(number, file_descriptor)
        noscope_time = time.perf_counter() - st

        st = time.perf_counter()
        jvc_pipeline(number, file_descriptor)
        jvc_time = time.perf_counter() - st

        file_descriptor.write(f"number of samples: {number}\n")
        file_descriptor.write(f"time taken for us: {us_time} (sec)\n")
        file_descriptor.write(f"time taken for noscope: {noscope_time} (sec)\n")
        file_descriptor.write(f"time taken for jvc: {jvc_time} (sec)\n")
    """



