"""
Implementation of Noscope's time diff detector using background image as reference frame

General Process:
1. We divide the dataset according to the video
2. We generate a background image depending on the video (examples are given noscoperep_experiment.ipynb
3. Threshold is generated to match the amount of images that need to be examined for other methods
4. Evaluation is done by propagating the labels to the neighboring frames
"""



import os
import sys
sys.argv=['']
sys.path.append('/nethome/jbang36/eva_jaeho')
from others.amdegroot.eval_uad2 import *  ## we import all the functions from here and perform our own evaluation
from others.amdegroot.data.uad import UAD_ROOT, UADAnnotationTransform, UADDetection
from others.amdegroot.data.uad import UAD_CLASSES as labelmap
from eva_storage.sampling_experiments.no_sample_ssd_evaluation import do_python_eval_for_uniform_sampling, \
    convert_labels_to_binary, propagate_labels

from loaders.uadetrac_loader import UADetracLoader


import numpy as np


def get_ref_frames_per_video(test_images, test_labels, test_vi):
    # get frames that don't have annotation -- we will assume that these frames are the ones that don't have labels
    empty_annotations = []
    for i in range(len(test_labels)):
        if test_labels[i] is None:
            empty_annotations.append(i)
    empty_annotations = np.array(empty_annotations)

    empty_images = test_images[empty_annotations]

    ####discard consecutive numbers
    filtered = []
    previous = -1
    for i in range(len(empty_annotations)):
        if previous == -1:
            filtered.append(empty_annotations[i])
            previous = empty_annotations[i]
        elif empty_annotations[i] == previous + 1:
            previous += 1
        else:
            filtered.append(empty_annotations[i])
            previous = empty_annotations[i]

    # number of reference images = len(test_vi) - 1
    ref_images = np.ndarray(shape=(len(test_vi) - 1, test_images.shape[1], test_images.shape[2], test_images.shape[3]),
                            dtype=np.uint8)

    for i in range(len(test_vi) - 1):
        start_index = test_vi[i]
        end_index = test_vi[i + 1]
        no_anno_frames = []
        for j in range(len(empty_annotations)):
            if empty_annotations[j] > start_index and empty_annotations[j] < end_index:
                no_anno_frames.append(empty_annotations[j])

        ## if we have unannotated frames, avg them and use them as reference
        if len(no_anno_frames) > 0:
            chosen_images = test_images[np.array(no_anno_frames)]
            ref_images[i] = np.mean(chosen_images, axis=0).astype(np.uint8)

        ## if we don't have, avg all frames within video
        else:
            chosen_images = test_images[start_index:end_index, :, :, :]
            ref_images[i] = np.mean(chosen_images, axis=0).astype(np.uint8)

    return ref_images

def get_eval_indices(skipped_images, skipped_vi, ref_images):
    ### make threshold around 76
    ### now we need to derive the frames that will be evaluated using DL and the corresponding mappings
    frames_to_eval_indices = []
    #video_it_refers_to = []
    threshold = 90

    for i in range(len(skipped_images)):
        for j in range(len(skipped_vi) - 1):
            if i >= skipped_vi[j] and i < skipped_vi[j + 1]:
                mse_error = np.square(np.subtract(skipped_images[i], ref_images[j])).mean()
                if mse_error > threshold:
                    frames_to_eval_indices.append(i)
                    #video_it_refers_to.append(j)
                break

    return frames_to_eval_indices

def get_rep_propagation_mapping(skipped_images, frames_to_eval_indices):
    initial_mapping = np.zeros(len(skipped_images))
    curr_index = 0
    for i in range(len(initial_mapping)):
        if i < frames_to_eval_indices[curr_index]:
            initial_mapping[i] = curr_index
        else:  # i == frames_to_eval_indices[curr_index]
            initial_mapping[i] = curr_index
            if curr_index < len(frames_to_eval_indices) - 1:
                curr_index += 1

    return initial_mapping

def get_skip_propagation_mapping(filtered_images, initial_mapping, sampling_rate):
    reference_indexes = np.zeros(len(filtered_images))
    length = len(initial_mapping)
    for i in range(length):
        for j in range(sampling_rate):
            k = i * sampling_rate + j
            if k >= len(reference_indexes):
                break
            reference_indexes[k] = initial_mapping[i]

    return reference_indexes




if __name__ == "__main__":
    loader = UADetracLoader()
    test_images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_images')
    test_labels, test_boxes = loader.load_labels(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_xml')
    test_vi = loader.get_video_start_indices()
    test_labels = test_labels['vehicle']


    ref_images = get_ref_frames_per_video(test_images, test_labels, test_vi)

    filtered_images, filtered_labels, filtered_boxes, filtered_vi = loader.filter_input3(test_images, test_labels,
                                                                                         test_boxes, test_vi)
    skip_rate = 15
    skipped_images = filtered_images[::skip_rate]
    skipped_labels = filtered_labels[::skip_rate]
    skipped_boxes = filtered_boxes[::skip_rate]
    skipped_vi = filtered_vi / skip_rate
    skipped_vi = np.ceil(skipped_vi)

    ## avg MSE is around 76, variance is 176
    frames_to_eval_indices = get_eval_indices(skipped_images, skipped_vi, ref_images)
    initial_mapping = get_rep_propagation_mapping(skipped_images, frames_to_eval_indices)
    mapping = get_skip_propagation_mapping(filtered_images, initial_mapping, skip_rate)
    mapping = mapping.astype(np.int)

    skipped_labels = np.array(skipped_labels)
    skipped_boxes = np.array(skipped_boxes)
    images_eval = skipped_images[frames_to_eval_indices]
    labels_eval = skipped_labels[frames_to_eval_indices]
    boxes_eval = skipped_boxes[frames_to_eval_indices]

    test_dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    test_dataset.set_images(images_eval)
    test_dataset.set_labels(labels_eval)
    test_dataset.set_boxes(boxes_eval)

    trained_model = '/nethome/jbang36/eva_jaeho/others/amdegroot/weights/finalists/ssd300_UAD_0408_90000.pth'
    args.trained_model = trained_model
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    logger.info(f"Loaded model {args.trained_model}")

    net = net.cuda()
    cudnn.benchmark = True

    import time

    st = time.perf_counter()
    output_dir = get_output_dir('ssd300_uad', set_type)
    box_list = detect_all_boxes(net, test_dataset, output_dir)
    et = time.perf_counter()

    write_voc_results_file(box_list, test_dataset)

    all_class_recs, nposes = group_annotation_by_class(test_dataset)

    image_count = len(images_eval)
    sampled_gt_labels, sampled_predicted_labels = do_python_eval_for_uniform_sampling(all_class_recs, nposes,
                                                                                      image_count, output_dir)

    ##TODO: propagate the label and compute precision
    dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    dataset.set_images(filtered_images)
    dataset.set_labels(filtered_labels)
    dataset.set_boxes(filtered_boxes)

    ## convert labels format
    all_gt_labels = convert_labels_to_binary(dataset)

    ## propagate the labels and compute precision
    sampled_propagated_predicted_labels = propagate_labels(sampled_predicted_labels, mapping)

    from sklearn.metrics import accuracy_score

    for key, value in all_gt_labels.items():
        score = accuracy_score(all_gt_labels[key], sampled_propagated_predicted_labels[key])
        print(f"key: {key}, score: {score}")

    logger.info(f"Total time taken for evaluating {len(images_eval)} is {et - st} (secs)")
