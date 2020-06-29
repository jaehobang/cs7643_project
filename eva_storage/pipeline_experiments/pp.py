"""
We want to evaluate multiple pipelines and how they perform on an end-to-end basis
The scenarios we want to deal with are as follows:
UDF - DONE (udf.py)
Sample -> UDF - DONE(udf.py)
Filter -> UDF (PP) - CURRENT
Sample -> Filter -> UDF - CURRENT

"""

from loaders.seattle_loader import SeattleLoader
from loaders.ssd_loader import SSDLoader
from eva_storage.sampling_experiments.sampling_utils import evaluate_with_gt4, evaluate_with_gt5, sample3_middle
from others.amdegroot.data.seattle import SEATTLE_CLASSES as labelmap
from timer import Timer
from eva_storage.featureExtractionMethods import *
from eva_storage.samplingMethods import *
from eva_storage.temporalClusterModule import *
from loaders.pp_loader import PPLoader
import yaml
import os

timer = Timer()


def no_sampling():
    timer.tic()
    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2.mov'

    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    labels, boxes, _ = loader.load_labels(video_directory, relevant_classes=['car', 'others', 'van'])

    #### example_query = 'select * from Seattle where car == 1'


    ## we need to invoke the ssd method for evaluation and return the labels to all these frames
    pp = PPLoader()
    plabels = pp.detect(images)
    total_time = timer.toc()
    ## we want to report the accuracy as well
    data_pack = evaluate_with_gt4(images, labels, boxes, images, plabels, pboxes, labelmap)
    ### the numbers should be printed
    data_pack['time'] = total_time
    return data_pack



def uniform_sampling():
    timer.tic()
    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2.mov'
    sampling_rate = 30 ## fps

    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    labels, boxes, _ = loader.load_labels(video_directory, relevant_classes=['car', 'others', 'van'])

    images_us, labels_us, boxes_us, mapping = sample3_middle(images, labels, boxes, sampling_rate=sampling_rate)

    #### example_query = 'select * from Seattle where car == 1'

    ## we need to invoke the ssd method for evaluation and return the labels to all these frames
    model = SSDLoader()
    plabels, pboxes = model.predict(images_us)
    total_time = timer.toc()
    data_pack = evaluate_with_gt5(labels, plabels, mapping)
    data_pack['time'] = total_time

    return data_pack


def uniform_sampling(total_eval_count = 1000):
    timer.tic()
    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2.mov'

    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    labels, boxes, _ = loader.load_labels(video_directory, relevant_classes=['car', 'others', 'van'])

    sampling_rate = len(images) // total_eval_count ## fps

    images_us, labels_us, boxes_us, mapping = sample3_middle(images, labels, boxes, sampling_rate=sampling_rate)

    #### example_query = 'select * from Seattle where car == 1'

    ## we need to invoke the ssd method for evaluation and return the labels to all these frames
    model = SSDLoader()
    plabels, pboxes = model.predict(images_us)
    total_time = timer.toc()
    data_pack = evaluate_with_gt5(labels, plabels, mapping)
    data_pack['time'] = total_time

    return data_pack

def jnet_sampling(total_eval_count):
    timer.tic()
    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2.mov'


    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    labels, boxes, _ = loader.load_labels(video_directory, relevant_classes=['car', 'others', 'van'])

    cluster_count = total_eval_count
    number_of_neighbors = 3

    # feature_extraction_method = DownSampleMeanMethod()
    feature_extraction_method = DownSampleMeanMethod()
    rep_selection_method = MeanEncounterMethod()
    temporal_cluster = TemporalClusterModule(downsample_method=feature_extraction_method,
                                             sampling_method=rep_selection_method)
    _, rep_indices, all_cluster_labels = temporal_cluster.run(images, number_of_clusters=cluster_count,
                                                              number_of_neighbors=number_of_neighbors)
    ## we need to get rep labels, rep_boxes as well
    rep_images = images[rep_indices]
    rep_labels = np.array(labels)[rep_indices]
    rep_boxes = np.array(boxes)[rep_indices]

    mapping = temporal_cluster.get_mapping(rep_indices, all_cluster_labels)
    mapping = mapping.astype(np.int)

    #### example_query = 'select * from Seattle where car == 1'

    ## we need to invoke the ssd method for evaluation and return the labels to all these frames
    model = SSDLoader()
    plabels, pboxes = model.predict(rep_images)
    total_time = timer.toc()
    data_pack = evaluate_with_gt5(labels, plabels, mapping)
    data_pack['time'] = total_time

    return data_pack


if __name__ == "__main__":
    ## We want to save all this data so that we can output later. How about we save in a yaml format?
    data = no_sampling()
    directory = '/nethome/jbang36/eva_jaeho/eva_storage/pipeline_experiments/'
    file = 'udfonly.yml'
    full = os.path.join(directory, file)
    total_eval_count = 1000

    with open(full, 'a+') as outfile:
        outfile.write('no sampling')
        yaml.dump(data, outfile, default_flow_style=False)
        outfile.write('--------------------')

    data = uniform_sampling(total_eval_count)
    with open(full, 'a+') as outfile:
        outfile.write('uniform sampling')
        yaml.dump(data, outfile, default_flow_style=False)
        outfile.write('--------------------')

    data = jnet_sampling(total_eval_count)
    with open(full, 'a+') as outfile:
        outfile.write('jnet sampling')
        yaml.dump(data, outfile, default_flow_style=False)
        outfile.write('--------------------')








