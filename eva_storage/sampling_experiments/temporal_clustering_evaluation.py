## we need to add base
import sys
sys.path.append('/nethome/jbang36/eva_jaeho')
sys.argv = ['']

from loaders.uadetrac_loader import UADetracLoader
from loaders.jackson_loader import JacksonLoader
from eva_storage.sampling_experiments.sampling_utils import create_dummy_boxes, evaluate_with_gt
from eva_storage.temporalClusterModule import TemporalClusterModule
from eva_storage.featureExtractionMethods import DownSampleMeanMethod, DownSampleMeanMethod2, DownSampleMaxMethod
from eva_storage.samplingMethods import *
from others.amdegroot.data.jackson import JACKSON_CLASSES
from others.amdegroot.eval_uad2 import * ## we import all the functions from here and perform our own evaluation






if __name__ == "__main__":


    total_eval_num = 2000

    loader = JacksonLoader()
    images = loader.load_images()

    ## we want to filter out only the ones that we want to use

    labels = loader.load_labels(relevant_classes=JACKSON_CLASSES)

    images, labels = loader.filter_input(images, labels)
    boxes = create_dummy_boxes(labels)

    st = time.perf_counter()

    cluster_count = total_eval_num
    number_of_neighbors = 3

    feature_extraction_method = DownSampleMeanMethod()
    rep_selection_method = MeanEncounterMethod()
    temporal_cluster = TemporalClusterModule(downsample_method=feature_extraction_method, sampling_method=rep_selection_method)
    _, rep_indices, all_cluster_labels = temporal_cluster.run(images, number_of_clusters=cluster_count,
                                                              number_of_neighbors=number_of_neighbors)
    ## we need to get rep labels, rep_boxes as well
    rep_images = images[rep_indices]
    rep_labels = np.array(labels)[rep_indices]
    rep_boxes = np.array(boxes)[rep_indices]

    mapping = temporal_cluster.get_mapping(rep_indices, all_cluster_labels)
    mapping = mapping.astype(np.int)


    sys.stdout = open("/nethome/jbang36/eva_jaeho/eva_storage/sampling_experiments/eval_block_avg_output.txt", "a")
    print(__file__)
    print(f"total eval num: {total_eval_num}")
    print(f"finished whole process in {time.perf_counter() - st} (secs)")
    print(f"Feature Extraction: {temporal_cluster.downsample_method}")
    print(f"Sampling Method: {temporal_cluster.sampling_method}")

    evaluate_with_gt(images, labels, boxes, rep_images, rep_labels, rep_boxes, mapping, JACKSON_CLASSES)

    print("\n\n")
