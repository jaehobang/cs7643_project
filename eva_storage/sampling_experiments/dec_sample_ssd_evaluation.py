"""

1. Evaluate the accuracy and speed of uniform sampling on SSD for UADetrac dataset



"""

from loaders.uadetrac_loader import UADetracLoader
from logger import Logger
from others.amdegroot.eval_uad2 import *  ## we import all the functions from here and perform our own evaluation
from others.amdegroot.data.uad import UAD_ROOT, UADAnnotationTransform, UADDetection
from others.amdegroot.data.uad import UAD_CLASSES as labelmap
from eva_storage.sampling_experiments.no_sample_ssd_evaluation import do_python_eval_for_uniform_sampling, \
    convert_labels_to_binary, propagate_labels

import others.ptdec_eva.examples.uadetrac as dec_uad

logger = Logger()


#################################################################################################
#################################################################################################


def gather_labels_by_frame(detpath, classname, class_recs, npos, image_count, ovthresh=0.5, use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)

       TODO: make it return labels, whether it be 0, 1, 2, 3 or
    """
    # assumes detections are in detpath.format(classname)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        predicted_labels = np.zeros(image_count)
        gt_labels = np.zeros(image_count)

        for d in range(nd):
            R = class_recs[int(image_ids[d])]
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                gt_labels[int(image_ids[d])] = 1

            if confidence[d] > ovthresh:
                ## we denote prediction as 1
                predicted_labels[int(image_ids[d])] = 1

    return gt_labels, predicted_labels


##################################################################################################
##################################################################################################


##################################################################################
##################################################################################

def load_original_data(test=True):
    if test:
        loader = UADetracLoader()

        test_images = loader.load_cached_images(name='uad_test_images.npy', vi_name='uad_test_vi.npy')
        test_labels = loader.load_cached_labels(name='uad_test_labels.npy')
        test_boxes = loader.load_cached_boxes(name='uad_test_boxes.npy')

        test_labels = test_labels['vehicle']

        test_images, test_labels, test_boxes = loader.filter_input3(test_images, test_labels, test_boxes)

        return test_images, test_labels, test_boxes

    else:
        logger.error("not implemented yet!!!")
        return None


def sample3(images, labels, boxes, sampling_rate=30):
    ## for uniform sampling, we will say all the frames until the next selected from is it's 'property'
    reference_indexes = []
    length = len(images[::sampling_rate])

    for i in range(length):
        for j in range(sampling_rate):
            if i * sampling_rate + j >= len(images):
                break
            reference_indexes.append(i)

    assert (len(reference_indexes) == len(images))
    return images[::sampling_rate], labels[::sampling_rate], boxes[::sampling_rate], reference_indexes


def sample_jnet(images, labels, boxes, sampling_rate=30):
    """
    TODO:

    :param images:
    :param labels:
    :param boxes:
    :param sampling_rate:
    :return:
    """
    ## we determine the number of clusters through sampling rate
    cluster_num = len(images) / sampling_rate



def get_rep_and_mapping(cluster_labels):
    label_set = set()
    indices_list = []
    for i, cluster_label in enumerate(cluster_labels):
        if cluster_label not in label_set:
            label_set.add(cluster_label)
            indices_list.append(i)

    rep_indices = indices_list

    mapping = np.zeros(len(cluster_labels))
    for i, value in enumerate(rep_indices):
        corresponding_cluster_number = cluster_labels[value]
        members_in_cluster_indices = cluster_labels == corresponding_cluster_number
        mapping[members_in_cluster_indices] = i

    assert(len(mapping) == len(cluster_labels))
    assert(len(rep_indices) == len(label_set))

    mapping = mapping.astype(np.int)
    rep_indices = np.array(rep_indices).astype(np.int)
    return rep_indices, mapping




#########
####### New run script
#######

if __name__ == "__main__":

    ## load the dataset -- note we have to have a 28x28 version along with the original
    loader = UADetracLoader()

    images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_images')
    labels, boxes = loader.load_labels(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_xml')
    labels = labels['vehicle']

    images, labels, boxes = loader.filter_input3(images, labels, boxes)

    ## need to load DEC model
    cluster_num = 600
    dec = dec_uad.load_dec('dec', cluster_num) ##model_name, cluster_num
    predicted_labels = dec_uad.predict_wrapper(images, labels, dec)

    ## need to pick rep and mapping
    rep_indices, mapping = get_rep_and_mapping(predicted_labels)

    images_eval = images[rep_indices]
    labels_eval = np.array(labels)[rep_indices]
    boxes_eval = np.array(boxes)[rep_indices]

    eval_dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    eval_dataset.set_images(images_eval)
    eval_dataset.set_labels(labels_eval)
    eval_dataset.set_boxes(boxes_eval)

    ### load the model
    trained_model = '/nethome/jbang36/eva_jaeho/others/amdegroot/weights/finalists/ssd300_UAD_0408_90000.pth'
    args.trained_model = trained_model
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    logger.info(f"Loaded model {args.trained_model}")

    net = net.cuda()
    cudnn.benchmark = True

    st = time.perf_counter()
    output_dir = get_output_dir('ssd300_uad', set_type)
    box_list = detect_all_boxes(net, eval_dataset, output_dir)
    et = time.perf_counter()

    write_voc_results_file(box_list, eval_dataset)

    all_class_recs, nposes = group_annotation_by_class(eval_dataset)

    image_count = len(images_eval)
    sampled_gt_labels, sampled_predicted_labels = do_python_eval_for_uniform_sampling(all_class_recs, nposes,
                                                                                      image_count, output_dir)

    ##TODO: propagate the label and compute precision
    dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    dataset.set_images(images)
    dataset.set_labels(labels)
    dataset.set_boxes(boxes)

    ## convert labels format
    all_gt_labels = convert_labels_to_binary(dataset)

    ## propagate the labels and compute precision
    sampled_propagated_predicted_labels = propagate_labels(sampled_predicted_labels, mapping)

    from sklearn.metrics import accuracy_score

    for key, value in all_gt_labels.items():
        score = accuracy_score(all_gt_labels[key], sampled_propagated_predicted_labels[key])
        print(f"key: {key}, score: {score}")

    logger.info(f"Total time taken for evaluating {len(images_eval)} is {et - st} (secs)")


"""
RESULTS:
key: car, score: 0.9912226040201542
key: bus, score: 0.5880855306496697
key: others, score: 0.6250467356276818
key: van, score: 0.6053376537824702
"""


