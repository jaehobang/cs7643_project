"""
In this file, we perform accuracy experiments with benchmarks
same content as paper_experiments.ipynb

TODO: caching model results....


"""


import os
import sys
import numpy as np


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

import sklearn.metrics as metrics


datasets = []
benchmarks = []

class Writer:

    @staticmethod
    def write(method_name, dataset_name, benchmark_results):
        write_directory = os.path.join('/nethome/jbang36/eva_jaeho/data/benchmark_results/preprocessing_methods', dataset_name, method_name+'.txt')
        base = os.path.dirname(write_directory)
        os.makedirs(base, exist_ok=True)
        file_descriptor = open(write_directory, 'a+')
        from datetime import datetime
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")


        file_descriptor.write(f"Timestamp: {date_time}")
        for label, value in benchmark_results.items():
            f1_score, precision, recall = value
            file_descriptor.write(f"{label}: F1 - {f1_score}, P - {precision}, R - {recall}\n")
        file_descriptor.close()

        return

######################################
#### Loading Datasets ################
######################################

def load_uadetrac():
    uadetrac_loader = UADetracLoader()
    ua_image_directory = '/nethome/jbang36/eva_jaeho/data/ua_detrac/train_images'
    image_size = (300, 300)
    uad_images = uadetrac_loader.load_images(dir=ua_image_directory, image_size=image_size)
    ua_label_directory = os.path.join(os.path.dirname(ua_image_directory), 'train_xml')
    ua_labels, ua_boxes = uadetrac_loader.load_labels(dir=ua_label_directory)
    # convert the labels to evaluation format
    gt_labels = {}
    gt_labels['car'] = UADetracConverter.convert2limit_queries(ua_labels['vehicle'], car=1)
    gt_labels['others'] = UADetracConverter.convert2limit_queries(ua_labels['vehicle'], others=1)
    gt_labels['bus'] = UADetracConverter.convert2limit_queries(ua_labels['vehicle'], van=1)
    gt_labels['van'] = UADetracConverter.convert2limit_queries(ua_labels['vehicle'], bus=1)
    for label in gt_labels.keys():
        gt_labels[label] = UADetracConverter.replaceNoneWithZeros(gt_labels[label])
    return ('ua_detrac', uad_images, gt_labels, ua_boxes)


def load_seattle2():
    seattle_loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2.mp4'
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


def load_seattle2_short():
    seattle_loader = SeattleLoader()
    directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'
    seattle_images = seattle_loader.load_images(directory)
    label_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_annotations'
    seattle_labels, seattle_boxes = seattle_loader.load_labels(label_directory,
                                                               relevant_classes=['car', 'van', 'others'])
    cutoff = 100000
    seattle_gt_labels = {}
    seattle_gt_labels['car'] = UADetracConverter.convert2limit_queries(seattle_labels, car=1)
    # seattle_gt_labels['others'] = UADetracConverter.convert2limit_queries(seattle_labels, others=1)
    # seattle_gt_labels['van'] = UADetracConverter.convert2limit_queries(seattle_labels, bus=1)
    for label in seattle_gt_labels.keys():
        seattle_gt_labels[label] = UADetracConverter.replaceNoneWithZeros(seattle_gt_labels[label])[:cutoff]
    seattle_boxes = seattle_boxes[:cutoff]

    assert(len(seattle_images) == cutoff)
    assert(len(seattle_boxes) == cutoff)
    assert(len(seattle_gt_labels['car']) == cutoff)
    return ('seattle2', seattle_images, seattle_gt_labels, seattle_boxes)


#######################################
########### Loading Datasets ##########
#######################################


def perform_benchmark_udf(gt_labels, ssd_predictions, dataset_name, query_name):
    benchmark_results = {}
    #for label, predicted_labels in ssd_predictions.items():
    predicted_labels = ssd_predictions
    print(gt_labels[:100])
    print('-------------')
    print(predicted_labels[:100])

    f1_score = metrics.f1_score(gt_labels, predicted_labels)
    precision_score = metrics.precision_score(gt_labels, predicted_labels)
    recall_score = metrics.recall_score(gt_labels, predicted_labels)
    benchmark_results[query_name] = ( f1_score, precision_score, recall_score )

    Writer.write('udf', dataset_name, benchmark_results)
    return benchmark_results


def perform_benchmark_pp(gt_labels, pp_predictions, dataset_name, query_name):
    benchmark_results = {}
    predicted_labels = pp_predictions
    f1_score = metrics.f1_score(gt_labels, predicted_labels)
    precision_score = metrics.precision_score(gt_labels, predicted_labels)
    recall_score = metrics.recall_score(gt_labels, predicted_labels)
    benchmark_results[query_name] = (f1_score, precision_score, recall_score)

    Writer.write('pp', dataset_name, benchmark_results)
    return benchmark_results


def perform_benchmark_us_pp(images, gt_labels, pp_predictions, dataset_name, query_name, number_of_samples):
    skip_rate = len(images) // number_of_samples
    sampled_images, _, _, mapping = sample3_middle(images, None, None, sampling_rate = skip_rate)

    sampled_pp_predictions = pp_predictions[::skip_rate]

    benchmark_results = {}

    data_pack = evaluate_with_gt5(gt_labels, sampled_pp_predictions, mapping)
    ### data_pack has keys ( accuracy, precision, recall, f1_score
    f1_score = data_pack['f1_score']
    precision_score = data_pack['precision']
    recall_score = data_pack['recall']
    benchmark_results[query_name] = (f1_score, precision_score, recall_score)

    Writer.write(f'us_pp_{number_of_samples}', dataset_name, benchmark_results)
    return benchmark_results


def perform_benchmark_noscope_pp(images, gt_labels, pp_predictions, dataset_name, query_name, number_of_samples):
    t_diff = 1
    delta_diff = 100
    noscope_rep_indices, mapping = set_frame_count(number_of_samples, images, t_diff, delta_diff)
    noscope_sample_labels = pp_predictions[noscope_rep_indices]

    benchmark_results = {}

    data_pack = evaluate_with_gt5(gt_labels, noscope_sample_labels, mapping)
    f1_score = data_pack['f1_score']
    precision_score = data_pack['precision']
    recall_score = data_pack['recall']
    benchmark_results[query_name] = (f1_score, precision_score, recall_score)

    Writer.write(f'noscope_pp_{number_of_samples}', dataset_name, benchmark_results)



def perform_benchmark_jvc_pp(images, gt_labels, pp_predictions, dataset_name, query_name, number_of_samples):
    preprocessor = Preprocessor()
    ### we need to supply the number of samples here.....
    jvc_indices = preprocessor.run(images, cluster_count = number_of_samples)
    jvc_mapping = preprocessor.get_mapping()
    jvc_sample_labels = pp_predictions[jvc_indices]

    benchmark_results = {}

    data_pack = evaluate_with_gt5(gt_labels, jvc_sample_labels, jvc_mapping)
    f1_score = data_pack['f1_score']
    precision_score = data_pack['precision']
    recall_score = data_pack['recall']
    benchmark_results[query_name] = (f1_score, precision_score, recall_score)

    Writer.write(f'jvc_pp_{number_of_samples}', dataset_name, benchmark_results)


def perform_benchmark_hierarchy_pp(images, gt_labels, pp_predictions, dataset_name, query_name, number_of_samples):
    preprocessor = Preprocessor()
    preprocessor.run_debug(images, stopping_point = number_of_samples)
    jvc_hierarchy, jvc_mapping = preprocessor.get_hierarchy_debug(stopping_point = number_of_samples)
    assert(len(jvc_hierarchy) == number_of_samples)
    assert(len(jvc_hierarchy) == max(jvc_mapping) + 1)

    jvc_sample_labels = pp_predictions[jvc_hierarchy]

    benchmark_results = {}

    data_pack = evaluate_with_gt5(gt_labels, jvc_sample_labels, jvc_mapping)
    f1_score = data_pack['f1_score']
    precision_score = data_pack['precision']
    recall_score = data_pack['recall']
    benchmark_results[query_name] = (f1_score, precision_score, recall_score)

    Writer.write(f'hierarchy_pp_{number_of_samples}', dataset_name, benchmark_results)


def perform_benchmarks(dataset, ssd_predictions, pp_predictions, label_name, query_name, number_of_samples):
    """
    In this function, we load all the benchmarks we want to use for evaluation
    UDF
    filter + UDF
    US + filter + UDF
    noscope + filter + UDF
    jvc + filter + UDF
    hierarchy + filter + UDF

    :return:
    """
    name, images, labels, boxes = dataset
    ## benchmark_functions
    #perform_benchmark_udf(labels[label_name], ssd_predictions, name, query_name)
    #perform_benchmark_pp(labels[label_name], pp_predictions, name, query_name)
    #perform_benchmark_us_pp(images, labels[label_name], pp_predictions, name, query_name, number_of_samples)
    #perform_benchmark_noscope_pp(images, labels[label_name], pp_predictions, name, query_name, number_of_samples)
    perform_benchmark_jvc_pp(images, labels[label_name], pp_predictions, name, query_name, number_of_samples)
    #perform_benchmark_hierarchy_pp(images, labels[label_name], pp_predictions, name, query_name, number_of_samples)







if __name__ == "__main__":

    ### we will run it from tmux
    print(f"Running file {__file__}...")

    datasets = []
    #datasets.append(load_uadetrac())
    datasets.append(load_seattle2_short())
    print(f"finished loading datasets")
    ssd_loader = SSDLoader()
    number_of_samples = [5000, 500, 100, 50]

    for dataset in datasets:
        name, images, labels, boxes = dataset
        print(f"starting evaluation for dataset {name}")

        ssd_predictions = ssd_loader.predict(images, name=name, cuda = True)
        """MAKING MODIFICATIONS JUST FOR SEATTLE2 SHORT!!!!!"""
        if name == 'seattle2':
            for label in ssd_predictions:
                ssd_predictions[label] = ssd_predictions[label][:100000]

        """DONE MAKING MODIFICATIONS!!! REMEMBER TO MOVE THIS LINE"""

        print(f"Done with ssd predictions for {name}")
        queries = []
        label_query_set = []
        for label in labels:
            query = label + '=1'
            label_query_set.append( (label, query) )

        for label, query in label_query_set:
            print(f">> Name of dataset: {name}, Query being processed: {query}")
            pp_loader = PPLoader(name, query)
            pp_predictions = pp_loader.predict(images) ## this is for specific query whereas ssd_predictions are not
            print(f"Done with pp predictions for {name} / {query}")
            for number in number_of_samples:
                perform_benchmarks(dataset, ssd_predictions[label], pp_predictions, label, query, number)

        print(f"Done with benchmark predictions for {name}")







