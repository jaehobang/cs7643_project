"""

1. Evaluate the accuracy and speed of uniform sampling on SSD for UADetrac dataset



"""

from loaders.uadetrac_loader import UADetracLoader
from loaders.jackson_loader import JacksonLoader
from logger import Logger
from others.amdegroot.eval_uad2 import * ## we import all the functions from here and perform our own evaluation
from others.amdegroot.data.uad import UAD_ROOT, UADAnnotationTransform, UADDetection
from others.amdegroot.data.uad import UAD_CLASSES as labelmap
from eva_storage.sampling_experiments.sampling_utils import *


logger = Logger()




#################################################################################################
#################################################################################################


def gather_labels_by_frame(detpath, classname, class_recs, npos, image_count, ovthresh=0.5,  use_07_metric=True):
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
    #cachedir = os.path.join(devkit_path, 'annotations_cache')
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

        gt_labels, predicted_labels = gather_labels_by_frame(filename, cls, all_class_recs[cls], nposes[cls], image_count, ovthresh=0.5, use_07_metric=use_07_metric)
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

"""
def convert_labels_to_binary(dataset):
    
    this generates format as follows:
    {'car', :[1,0,1,1,1,1,1,1,.....],
     'bus' : [0,0,0,1,1,1,1,1,1].....}
    :param dataset: dataset object
    :return: above
    

    label_dict = {}
    for cls_index, cls in enumerate(labelmap):
        label_dict[cls] = np.zeros(len(dataset))

    for image_id in range(len(dataset)):
        labels = dataset.get_labels(image_id) #['car', 'car', 'bus', 'car', 'car'....]
        for label_id, label in enumerate(labels):
            label_dict[label][image_id] = 1

    return label_dict

def propagate_labels(sampled_predicted_labels:dict, mapping):
    ## we propagate the labels from sampling to all frames
    new_dict = {}
    for key, value in sampled_predicted_labels.items():
        print(f"{key}, type of key {type(key)}")
        new_dict[key] = np.zeros(len(mapping))
        for i in range(len(mapping)):
            new_dict[key][i] = sampled_predicted_labels[key][mapping[i]]

    return new_dict

"""

########### experiments with much more data loaded / evaluated

def create_dummy_boxes(labels):
    boxes = []
    for i in range(len(labels)):
        box_frame = []
        for j in labels[i]:
            box_frame.append((0,0,0,0))
        boxes.append(box_frame)
    return boxes




if __name__ == "__main__":
    total_eval_num = 100
    """
    loader = UADetracLoader()
    ## we assume the skip rate is 15, but in essence, we are performing uniform sampling once every 4 images

    images = loader.load_images(dir = '/nethome/jbang36/eva_jaeho/data/ua_detrac/test_images')
    labels, boxes = loader.load_labels(dir = '/nethome/jbang36/eva_jaeho/data/ua_detrac/test_xml')
    labels = labels['vehicle']
    images, labels, boxes = loader.filter_input3(images, labels, boxes)

    """
    loader = JacksonLoader()
    images = loader.load_images(image_size = 300)

    ## we want to filter out only the ones that we want to use
    from others.amdegroot.data.jackson import JACKSON_CLASSES
    labels = loader.load_labels(relevant_classes = JACKSON_CLASSES)

    images, labels = loader.filter_input(images, labels)
    boxes = create_dummy_boxes(labels)

    sampling_rate = int(len(images) / total_eval_num)

    ## let's sample images
    images_us, labels_us, boxes_us, mapping = sample3(images, labels, boxes, sampling_rate = sampling_rate)


    evaluate_with_gt(images, labels, boxes, images_us, labels_us, boxes_us, mapping, JACKSON_CLASSES)



    """
    sampling_rate 60
    testing data 937*60 instances
    
    key: car, score: 0.9735965958658999
    key: bus, score: 0.6489931810493706
    key: others, score: 0.6369220360710025
    key: van, score: 0.6027382626809337
    04-22-2020 20:30:24 [info:049]INFO : Total time taken for evaluating 937 is 21.98056793026626 (secs)
    
    """





##############################################
### Generate numbers for uniform sampling ####
##############################################






"""
if __name__ == "__main__":



    loader = UADetracLoader()
    sampling_rate = 30

    images, labels, boxes = load_original_data()


    ## let's sample images
    images_us, labels_us, boxes_us, mapping = sample3(images, labels, boxes, sampling_rate = sampling_rate)


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

    import time
    st = time.perf_counter()
    output_dir = get_output_dir('ssd300_uad', set_type)
    box_list = detect_all_boxes(net, test_dataset, output_dir)
    et = time.perf_counter()

    write_voc_results_file(box_list, test_dataset)

    all_class_recs, nposes = group_annotation_by_class(test_dataset)


    image_count = len(images_us)
    sampled_gt_labels, sampled_predicted_labels = do_python_eval_for_uniform_sampling(all_class_recs, nposes, image_count, output_dir)

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

    logger.info(f"Total time taken for evaluating {len(images_us)} is {et - st} (secs)")
"""



"""Results

sampling rate: 30

car, type of key <class 'str'>
bus, type of key <class 'str'>
others, type of key <class 'str'>
van, type of key <class 'str'>
key: car, score: 0.9268027633315221
key: bus, score: 0.7083753784056509
key: others, score: 0.967398897772258
key: van, score: 0.6687107040285648
04-20-2020 15:42:53 [info:049]INFO : Total time taken for evaluating 430 is 9.831716096028686 (secs)


sampling rate: 90 (30*3)

key: car, score: 0.9254055732360476
key: bus, score: 0.600481254366219
key: others, score: 0.9720561980905069
key: van, score: 0.6319180315143988

04-20-2020 15:42:01 [info:049]INFO : Total time taken for evaluating 144 is 4.675174464471638 (secs)

"""

#######################################################################
