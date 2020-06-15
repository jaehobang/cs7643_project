"""
This script is organized from the jupyter notebook
Reflects the experiment done for creating an environment where Uniform Sampling does not perform well

Changed the query type.
Accuracy of US depends on how many samples we extract to evaluate, but for things that are 80% or below,
JNET performs better

Distribution:
ALL: 22277, TRUE: 4674, FALSE: 17603, NAN: 0 (around 20%)

100 samples - 0.90 (US) vs 0.89
50 samples - 0.835 vs 0.865

30 samples
key: foo, accuracy: 0.7790546303362212, precision: 0.4721473495058401, recall: 0.449721865639709, f1_score: 0.4606618452772299
key: foo, accuracy: 0.8464335413206446, precision: 0.8594377510040161, recall: 0.3204963628583654, f1_score: 0.4668848371513168

50 samples
US: key: foo, accuracy: 0.8359743232930825, precision: 0.5955056179775281, recall: 0.6803594351732991, f1_score: 0.6351108448172558
JNET: key: foo, accuracy: 0.865287067378911, precision: 0.8696420680512594, recall: 0.42105263157894735, f1_score: 0.567392244486089

100 samples
US: key: foo, accuracy: 0.9057323697086681, precision: 0.722972972972973, recall: 0.8928112965340179, f1_score: 0.7989661114302125
JNET: key: foo, accuracy: 0.8890784216905329, precision: 0.9180265654648956, recall: 0.5175438596491229, f1_score: 0.6619236557668627

Analysis:
Some big clusters have high accuracy, some big clusters have low accuracy. There are small clusters that have low accuracy
-- conclusion is that cluster size doesn't have much correlation with the accuracies (although we assume that the big clusters
with high accuracy were returning 0 (the dominant label in this set)

JNET method is better at precision, worse in terms of recall. What does this mean???

"""

from loaders.uadetrac_loader import UADetracLoader
from loaders.uadetrac_label_converter import UADetracConverter
from eva_storage.sampling_experiments.sampling_utils import *
from eva_storage.sampling_experiments.temporal_clustering_evaluation import *


loader = UADetracLoader()
images = loader.load_images(dir='/nethome/jbang36/eva_storage/data/ua_detrac/test_images')
test_labels, test_boxes = loader.load_labels(dir='/nethome/jbang36/eva_storage/data/ua_detrac/test_xml')
labels = test_labels['vehicle']
images, labels, boxes = loader.filter_input3(images, labels, test_boxes)
limit_labels = UADetracConverter.convert2limit_queries(labels, car = 5, bus = 1, van = 1) ## let's try this setting for US


print(f"dataset balance: {UADetracConverter.getTrueFalseCount(limit_labels)}")

### we evaluate on US

from eva_storage.sampling_experiments.uniform_sample_ssd_evaluation import *

total_eval_num = 100
sampling_rate = int(len(images) / total_eval_num)

images_us, labels_us, boxes_us, mapping_us = sample3_middle(images, limit_labels, boxes, sampling_rate = sampling_rate)
## we need to derive the 'cluster_labels' for sampling
sample_cluster_labels = get_cluster_labels(mapping_us)

predicted_labels_sample, gt_labels_sample = evaluate_with_gt2(limit_labels, labels_us, mapping_us)


####################### JNET Clustering #########################
### let's try out my clustering method
cluster_count = total_eval_num

st = time.perf_counter()
number_of_neighbors = 3

#feature_extraction_method = DownSampleMeanMethod()
feature_extraction_method = DownSampleMeanMethod()
rep_selection_method = MeanEncounterMethod()
temporal_cluster = TemporalClusterModule(downsample_method=feature_extraction_method, sampling_method=rep_selection_method)
_, rep_indices, all_cluster_labels = temporal_cluster.run(images, number_of_clusters=cluster_count,
                                                          number_of_neighbors=number_of_neighbors)
## we need to get rep labels, rep_boxes as well
rep_images = images[rep_indices]
rep_labels = np.array(limit_labels)[rep_indices]
rep_boxes = np.array(boxes)[rep_indices]

mapping = temporal_cluster.get_mapping(rep_indices, all_cluster_labels)
mapping = mapping.astype(np.int)


print(f"total eval num: {total_eval_num}")
print(f"finished whole process in {time.perf_counter() - st} (secs)")
print(f"Feature Extraction: {temporal_cluster.downsample_method}")
print(f"Sampling Method: {temporal_cluster.sampling_method}")

predicted_labels, gt_labels = evaluate_with_gt2(limit_labels, rep_labels, mapping)

print("\n\n")





