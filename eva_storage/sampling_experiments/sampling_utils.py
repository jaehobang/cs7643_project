
from others.amdegroot.eval_uad2 import *  ## we import all the functions from here and perform our own evaluation
from others.amdegroot.data.uad import UAD_ROOT, UADAnnotationTransform, UADDetection
#from others.amdegroot.data.uad import UAD_CLASSES as labelmap
from sklearn.metrics import accuracy_score


def convert_labels_to_binary(dataset, labelmap):
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



def evaluate_with_gt(images, labels, boxes, rep_images, rep_labels, rep_boxes, mapping, labelmap):
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
    all_gt_labels = convert_labels_to_binary(dataset, labelmap)
    all_rep_labels = convert_labels_to_binary(test_dataset, labelmap)

    ## propagate the labels and compute precision
    sampled_propagated_predicted_labels = propagate_labels(all_rep_labels, mapping)

    for key, value in all_gt_labels.items():
        score = accuracy_score(all_gt_labels[key], sampled_propagated_predicted_labels[key])
        print(f"key: {key}, score: {score}")

    return





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


def create_dummy_boxes(labels):
    boxes = []
    for i in range(len(labels)):
        box_frame = []
        for j in labels[i]:
            box_frame.append((0,0,0,0))
        boxes.append(box_frame)
    return boxes
