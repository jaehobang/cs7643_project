"""
We implement no scope functionality and evaluate on ssd

"""

"""

1. Evaluate the accuracy and speed of uniform sampling on SSD for UADetrac dataset



"""

from loaders.uadetrac_loader import UADetracLoader
from loaders.jackson_loader import JacksonLoader
from logger import Logger
from others.amdegroot.eval_uad2 import *  ## we import all the functions from here and perform our own evaluation
from eva_storage.sampling_experiments.sampling_utils import create_dummy_boxes
from eva_storage.sampling_experiments.sampling_utils import evaluate_with_gt
from others.amdegroot.data.uad import UAD_CLASSES
from others.amdegroot.data.jackson import JACKSON_CLASSES

logger = Logger()


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

def set_frame_count(wanted_frame_count, images, initial_t_diff, initial_delta_diff):
    error_bound = 50
    rep_frame_indices, mapping = get_rep_indices(images, initial_t_diff, initial_delta_diff)
    curr_delta_diff = initial_delta_diff
    curr_t_diff = initial_t_diff
    up = False
    down = False
    change_rate = 100
    change_rate_down = 10
    print(f"Number of total frames {len(images)}")

    while True:
        if len(rep_frame_indices) > wanted_frame_count + error_bound:
            curr_delta_diff += change_rate
            rep_frame_indices, mapping = get_rep_indices(images, curr_t_diff, curr_delta_diff)
            print(f"UP: {curr_delta_diff} length: {len(rep_frame_indices)}")
            up = True
        elif len(rep_frame_indices) < wanted_frame_count - error_bound:
            curr_delta_diff -= change_rate_down
            rep_frame_indices, mapping = get_rep_indices(images, curr_t_diff, curr_delta_diff)
            print(f"DOWN: {curr_delta_diff} length: {len(rep_frame_indices)}")
            down = True
        else:
            break

        if up and down:
            ## we don't want the while loop to run forever, this parameter searching method is very rudimentary
            break



    return rep_frame_indices, mapping



if __name__ == "__main__":
    import os


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    skip_rate = 15
    t_diff = 1 ##TODO: we need to experiment with this as well but since we have skip_rate of 15, I think it will be okay
    delta_diff = 60  ##TODO: we need to modify this parameter (experiments in NoScope is probably using normalized images
    ideal_frame_count = 2000

    """
    loader = UADetracLoader()
    images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_images')
    labels, boxes = loader.load_labels(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_xml')
    labels = labels['vehicle']

    images, labels, boxes = loader.filter_input3(images, labels, boxes)
    """

    labelmap = JACKSON_CLASSES

    #jackson loader
    loader = JacksonLoader()
    images = loader.load_images(image_size=300)

    ## we want to filter out only the ones that we want to use
    from others.amdegroot.data.jackson import JACKSON_CLASSES

    labels = loader.load_labels(relevant_classes=JACKSON_CLASSES)

    images, labels = loader.filter_input(images, labels)
    boxes = create_dummy_boxes(labels)

    ### we skip frames

    images_us = images[::skip_rate]
    labels_us = labels[::skip_rate]
    boxes_us = boxes[::skip_rate]
    # convert to np arrays to do index slicing
    labels_us = np.array(labels_us)
    boxes_us = np.array(boxes_us)

    print(f"{len(images)}, {len(images_us)} {skip_rate}")

    rep_indices, mapping = set_frame_count(ideal_frame_count, images_us, t_diff, delta_diff)

    #rep_indices, mapping = get_rep_indices(images_us, t_diff, delta_diff)
    rep_images = images_us[rep_indices]
    rep_labels = labels_us[rep_indices]
    rep_boxes = boxes_us[rep_indices]

    final_mapping = get_final_mapping(mapping, skip_rate, len(images))
    final_mapping = final_mapping.astype(np.int)


    print(f"Number of frames evaluated: {len(rep_images)}")
    evaluate_with_gt(images, labels, boxes, rep_images, rep_labels, rep_boxes, final_mapping, labelmap) #UAD_CLASSES


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
