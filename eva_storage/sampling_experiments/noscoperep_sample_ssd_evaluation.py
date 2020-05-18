"""
We implement no scope functionality and evaluate on ssd

"""

"""

1. Use the representative frame method from noscope and evaluate on UAD



"""

from loaders.uadetrac_loader import UADetracLoader
from logger import Logger
from others.amdegroot.eval_uad2 import *  ## we import all the functions from here and perform our own evaluation
from others.amdegroot.data.uad import UAD_ROOT, UADAnnotationTransform, UADDetection
from others.amdegroot.data.uad import UAD_CLASSES as labelmap

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

def do_python_eval_for_uniform_sampling(all_class_recs, nposes, image_count, output_dir='output', use_07=True):
    """
    This function simply wraps the voc_eval function that does the main evaluation
    This function wraps and simply presents the AP values for each given class
    :param true_case_stats -- I have no idea what this is yet
    :param all_gt_boxes -- ground truth boxes that are organized in terms of image indexes
    :param all_difficult_case -- I have no idea what this is yet
    :param output_dir: directory to where detection files are located
    :param use_07: whether to use 07 evaluation format
    :return:
    """
    # cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    all_gt_labels = {}
    all_predicted_labels = {}
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)

        gt_labels, predicted_labels = gather_labels_by_frame(filename, cls, all_class_recs[cls], nposes[cls],
                                                             image_count, ovthresh=0.5, use_07_metric=use_07_metric)
        ## labels is [0,1 ....]? need to check this

        all_gt_labels[cls] = gt_labels
        all_predicted_labels[cls] = predicted_labels

    ##TODO: make sure what the keys are
    print(all_gt_labels.keys())
    print(all_predicted_labels.keys())
    return all_gt_labels, all_predicted_labels


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


def convert_labels_to_binary(dataset):
    """
    this generates format as follows:
    {'car', :[1,0,1,1,1,1,1,1,.....],
     'bus' : [0,0,0,1,1,1,1,1,1].....}
    :param dataset: dataset object
    :return: above
    """

    label_dict = {}
    for cls_index, cls in enumerate(labelmap):
        label_dict[cls] = np.zeros(len(dataset))

    for image_id in range(len(dataset)):
        labels = dataset.get_labels(image_id)  # ['car', 'car', 'bus', 'car', 'car'....]
        for label_id, label in enumerate(labels):
            label_dict[label][image_id] = 1

    return label_dict


def propagate_labels(sampled_predicted_labels: dict, mapping):
    ## we propagate the labels from sampling to all frames
    new_dict = {}
    for key, value in sampled_predicted_labels.items():
        print(f"{key}, type of key {type(key)}")
        new_dict[key] = np.zeros(len(mapping))
        for i in range(len(mapping)):
            new_dict[key][i] = sampled_predicted_labels[key][mapping[i]]

    return new_dict


def get_rep_indices(images, t_diff, delta_diff):
    """

    :param images: are already SAMPLED IMAGES -- around 3600 ish
    :return: representative frames, and mapping so that we can convert from rep frames to all frames
    """
    ## TODO: mapping's values should be the indices of the rep_frame it refers to
    ## we will be using a MSE evaluation method
    rep_frames_indices = [0]
    mapping = [0]
    curr_ref = 0

    for i in range(0, len(images) - 1, t_diff):
        mse_error = np.square(np.subtract(images[i], images[i+1])).mean()
        if mse_error > delta_diff:
            rep_frames_indices.append(i+1)
            curr_ref += 1
            mapping.append(curr_ref)
        else:
            mapping.append(curr_ref)

    assert(len(mapping) == len(images))
    rep_frames_indices = np.array(rep_frames_indices)
    return rep_frames_indices, mapping





def sample3(images, labels, boxes, sampling_rate = 30):
    ## for uniform sampling, we will say all the frames until the next selected from is it's 'property'
    reference_indexes = []
    length = len(images[::sampling_rate])

    for i in range(length):
        for j in range(sampling_rate):
            if i * sampling_rate + j >= len(images):
                break
            reference_indexes.append(i)

    assert(len(reference_indexes) == len(images))
    return images[::sampling_rate], labels[::sampling_rate], boxes[::sampling_rate], reference_indexes



def get_final_mapping(mapping, skip_rate, original_image_count):
    """
    We need this function because we apply the difference detector after skipping frames, so the mapping is from
    :param mapping:
    :param skip_rate:
    :param original_image_count:
    :return:
    """
    final_mapping = np.zeros(original_image_count)
    for i in range(original_image_count):
        if (i // skip_rate) >= len(mapping):
            print(i, skip_rate, len(mapping))
            print("--------------")
        else:
            final_mapping[i] = mapping[i // skip_rate]
    return final_mapping


#########
####### New run script
#######



if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    loader = UADetracLoader()
    skip_rate = 15
    ## we are not sure how many frames we will evaluate, but diff detection should return True or False as to whether we should evaluate


    t_diff = 1 ##TODO: we need to experiment with this as well but since we have skip_rate of 15, I think it will be okay
    delta_diff = 60  ##TODO: we need to modify this parameter (experiments in NoScope is probably using normalized images

    images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_images')
    labels, boxes = loader.load_labels(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_xml')
    labels = labels['vehicle']

    images, labels, boxes = loader.filter_input3(images, labels, boxes)

    ### we skip frames

    images_us = images[::skip_rate]
    labels_us = labels[::skip_rate]
    boxes_us = boxes[::skip_rate]
    # convert to np arrays to do index slicing
    labels_us = np.array(labels_us)
    boxes_us = np.array(boxes_us)

    image_count = len(images)
    assert(images_us.dtype == np.uint8)

    rep_indices, mapping = get_rep_indices(images_us, t_diff, delta_diff)
    rep_images = images_us[rep_indices]
    rep_labels = labels_us[rep_indices]
    rep_boxes = boxes_us[rep_indices]

    final_mapping = get_final_mapping(mapping, skip_rate, len(images))
    final_mapping = final_mapping.astype(np.int)

    ## now we have rep_frames, and we also have mappings

    test_dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    test_dataset.set_images(rep_images)
    test_dataset.set_labels(rep_labels)
    test_dataset.set_boxes(rep_boxes)


    trained_model = '/nethome/jbang36/eva_jaeho/others/amdegroot/weights/finalists/ssd300_UAD_0408_90000.pth'
    args.trained_model = trained_model
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    logger.info(f"Loaded model {args.trained_model}")

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    output_dir = get_output_dir('ssd300_uad', set_type)
    box_list = detect_all_boxes(net, test_dataset, output_dir)

    write_voc_results_file(box_list, test_dataset)

    all_class_recs, nposes = group_annotation_by_class(test_dataset)

    image_count = len(rep_images)
    sampled_gt_labels, sampled_predicted_labels = do_python_eval_for_uniform_sampling(all_class_recs, nposes,
                                                                                      image_count, output_dir)

    print('hello world')
    ## need to propagate the labels
    ## we also need to derive the gt labels for non sampled things....we just need to convert 'labels' to binary format I think
    ## before we do that let's examine the output of all_gt_labels, all_predicted_labels

    ##TODO: propagate the label and compute precision
    dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    dataset.set_images(images)
    dataset.set_labels(labels)
    dataset.set_boxes(boxes)

    ## convert labels format
    all_gt_labels = convert_labels_to_binary(dataset)

    ## propagate the labels and compute precision
    sampled_propagated_predicted_labels = propagate_labels(sampled_predicted_labels, final_mapping)

    from sklearn.metrics import accuracy_score

    for key, value in all_gt_labels.items():
        score = accuracy_score(all_gt_labels[key], sampled_propagated_predicted_labels[key])
        print(f"key: {key}, score: {score}")


"""
RESULTS:
car, type of key <class 'str'>
bus, type of key <class 'str'>
others, type of key <class 'str'>
van, type of key <class 'str'>
key: car, score: 0.9891395303291969
key: bus, score: 0.5857175921804618
key: others, score: 0.5655278010219524
key: van, score: 0.5822636067441737
"""
