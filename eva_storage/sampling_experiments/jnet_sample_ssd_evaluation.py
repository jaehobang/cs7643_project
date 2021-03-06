"""

1. Evaluate the accuracy and speed of uniform sampling on SSD for UADetrac dataset



"""

from loaders.uadetrac_loader import UADetracLoader
from logger import Logger
from others.amdegroot.eval_uad2 import *  ## we import all the functions from here and perform our own evaluation
from others.amdegroot.data.uad import UAD_ROOT, UADAnnotationTransform, UADDetection
from others.amdegroot.data.uad import UAD_CLASSES as labelmap
from sklearn.metrics import accuracy_score

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

def sample_jnet(images, labels, boxes, sampling_rate = 30):
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

### refactoring

def evaluate_with_ssd(images, labels, boxes, rep_images, rep_labels, rep_boxes, mapping):
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
    sampled_propagated_predicted_labels = propagate_labels(sampled_predicted_labels, mapping)

    for key, value in all_gt_labels.items():
        score = accuracy_score(all_gt_labels[key], sampled_propagated_predicted_labels[key])
        print(f"key: {key}, score: {score}")


    return



def evaluate_with_gt(images, labels, boxes, rep_images, rep_labels, rep_boxes, mapping):
    test_dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    test_dataset.set_images(rep_images)
    test_dataset.set_labels(rep_labels)
    test_dataset.set_boxes(rep_boxes)


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
    all_rep_labels = convert_labels_to_binary(test_dataset)

    ## propagate the labels and compute precision
    sampled_propagated_predicted_labels = propagate_labels(all_rep_labels, mapping)

    for key, value in all_gt_labels.items():
        score = accuracy_score(all_gt_labels[key], sampled_propagated_predicted_labels[key])
        print(f"key: {key}, score: {score}")

    return




#########
####### New run script
#######

if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    loader = UADetracLoader()
    skip_rate = 15
    sampling_rate = 6


    images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_images')
    labels, boxes = loader.load_labels(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_xml')
    labels = labels['vehicle']

    images, labels, boxes = loader.filter_input3(images, labels, boxes)
    #images = images[:20000]
    #labels = labels[:20000]
    #boxes = boxes[:20000]


    image_count = len(images)
    cluster_count = image_count // (skip_rate * sampling_rate)
    images = images.astype(np.uint8)


    from eva_storage.UNet import UNet
    network = UNet()
    images_compressed, _ = network.execute(images, load_dir = '/nethome/jbang36/eva_jaeho/data/models/unet_plain_testdata_0505-epoch60.pth')


    from eva_storage.clusterModule import ClusterModule
    cluster_module = ClusterModule()
    _, rep_indices, all_cluster_labels = cluster_module.run(images_compressed, number_of_clusters = cluster_count)
    ## we need to get rep labels, rep_boxes as well
    rep_images = images[rep_indices]
    rep_labels = np.array(labels)[rep_indices]
    rep_boxes = np.array(boxes)[rep_indices]

    mapping = cluster_module.get_mapping(rep_indices, all_cluster_labels)
    mapping = mapping.astype(np.int)

    ## now we have rep_frames, and we also have mappings
    ##evaluate_with_ssd(images, labels, boxes, rep_images, rep_labels, rep_boxes)
    evaluate_with_gt(images, labels, boxes, rep_images, rep_labels, rep_boxes, mapping)


##############################################
### Generate numbers for uniform sampling ####
##############################################
"""
if __name__ == "__main__":

    loader = UADetracLoader()
    sampling_rate = 15 ## total number of frames for

    images, labels, boxes = load_original_data()

    ## let's sample images
    images_us, labels_us, boxes_us, mapping = sample3(images, labels, boxes, sampling_rate=sampling_rate) ## basic frame skipping

    ##TODO: insert jnet sampling method here
    ## we need to train the network or have a saved version of that network, we will be using regular frames...
    



    test_dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    test_dataset.set_images(images_us)
    test_dataset.set_labels(labels_us)
    test_dataset.set_boxes(boxes_us)

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

    image_count = len(images_us)
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
    sampled_propagated_predicted_labels = propagate_labels(sampled_predicted_labels, mapping)

    from sklearn.metrics import accuracy_score

    for key, value in all_gt_labels.items():
        score = accuracy_score(all_gt_labels[key], sampled_propagated_predicted_labels[key])
        print(f"key: {key}, score: {score}")

"""
"""Results

skip: 15, sample: 4
key: car, score: 0.8558940303024908
key: bus, score: 0.47688144284010187
key: others, score: 0.7077821496608329
key: van, score: 0.5417237879893888


skip: 15, sample:6

key: car, score: 0.8136806309754838
key: bus, score: 0.47095269464276174
key: others, score: 0.706251001477736
key: van, score: 0.5176883223244966

"""


