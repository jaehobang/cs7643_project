"""
This file implements the dataset loading methods for UA-detrac
If any problem occurs, please email jaeho.bang@gmail.com


@Jaeho Bang

"""

import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2

from loaders.abstract_loader import AbstractLoader
import time
from loaders.decompressionModule import DecompressionModule
import pandas as pd
import re

#####
## ssd related imports
#####
from others.amdegroot.data.seattle import SEATTLE_CLASSES as labelmap
from others.amdegroot.ssd import build_ssd
import torch
from logger import Logger
import torch.backends.cudnn as cudnn
from others.amdegroot.eval_uad2 import get_output_dir, detect_all_boxes, write_voc_results_file
from others.amdegroot.data.seattle import SEATTLEDetection, SEATTLE_CLASSES, SEATTLE_ROOT
from others.amdegroot.data import BaseTransform
### end ###



from logger import Logger, LoggingLevel



# Make this return a dictionary of label to data for the whole dataset

class SeattleLabelGenerator:

    def __init__(self):
        self.logger = Logger()

    def generate_annotations(self, images, model_dir = None, cuda = True):

        if model_dir is None:
            model_dir = '/nethome/jbang36/eva_jaeho/others/amdegroot/weights/finalists/ssd300_UAD_0408_90000.pth'
        self.logger.info(f"model directory is {model_dir}")

        num_classes = len(labelmap) + 1
        net = build_ssd('test', 300, num_classes)
        dataset_mean = (92, 111, 120) ## figured out with np.mean(images, axis = (0,1,2))

        dataset = SEATTLEDetection(transform = BaseTransform(300, dataset_mean))
        dataset.set_images(images)

        net.load_state_dict(torch.load(model_dir))
        net.eval()
        self.logger.info(f"Loaded model {model_dir}")

        if cuda:
            net = net.cuda()
            cudnn.benchmark = True

        output_dir = get_output_dir('ssd300_seattle', 'annotations')
        box_list = detect_all_boxes(net, dataset, output_dir)
        write_voc_results_file(box_list, dataset)
        self.logger.info(f"Wrote annotations to {output_dir}")

        return



class SeattleLoader(AbstractLoader):
    def __init__(self, image_width=300, image_height=300):
        self.image_width = image_width
        self.image_height = image_height
        self.decompressionModule = DecompressionModule()
        self.images = None
        self.logger = Logger()


    def load_video(self, dir: str):
        """
        This function is not needed for ua_detrac
        Should never be called
        :return: None
        """
        return None

    def load_boxes(self, dir: str):
        if self.boxes is None:
            self.logger.info("Running load_labels first to get the boxes...")
            self.load_labels(dir)
        return self.get_boxes()

    def debugMode(self, mode=False):
        if mode:
            self.logger.setLogLevel(LoggingLevel.DEBUG)
        else:
            self.logger.setLogLevel(LoggingLevel.INFO)



    def load_images(self, dir: str = None, image_size=None, frame_count_limit = 1000000):
        ## load directly from video
        st = time.perf_counter()
        if dir is None:
            dir = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_final.mp4'
        images = self.decompressionModule.convert2images(dir, frame_count_limit = frame_count_limit)
        self.logger.info(f"Loaded {len(images)} in {time.perf_counter() - st} seconds")
        return images

    def load_predicate_labels(self, dir):
        """
        This will be in numpy array format where index refers to frame_num.

        :param dir:
        :return:
        """

        dir_name = os.path.dirname(dir)
        file_name = os.path.basename(dir)
        assert(os.path.exists(dir)) ## dir must exist or we raise error
        value_arr = []
        with open(dir, 'r') as file:
            line = file.readline().strip()
            while line:
                line_arr = line.split(',') ##[frame_id, value]
                while len(value_arr) < int(line_arr[0]):
                    value_arr.append(0) ##if not labeled, we assume the label is zero
                value_arr.append(int(line_arr[1]))
                line = file.readline().strip()
        ##let's convert to numpy array
        value_arr = np.array(value_arr)
        return value_arr





    def load_labels(self, dir: str = None, relevant_classes = None):
        """
        Example of relevant_classes: ['car']
        Loads vehicle type, speed, color, and intersection of ua-detrac
        vehicle type, speed is given by the dataset
        color, intersection is derived from functions built-in
        :return: labels
        """

        ## we expect annotations to be annotations dir,
        # ex: /nethome/jbang36/eva_jaeho/data/seattle/seattle2_short_annotations

        if dir is None:
            dir = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_short_annotations'

        ##TODO: 1. check if directory exists
        if not os.path.isdir(dir):
            self.logger.error(f"{dir} is non-existent")
            raise ValueError

        ### to do evaluation easily, we need to load in uad format
        # [['car', 'bus', 'car'],['car', 'others', ...]]
        # [[(min_x, min_y, max_x, max_y), (min_x, min_y, max_x, max_y)....]]
        labels = []
        boxes = []
        confidence_threshold = 0.5

        txt_files = os.listdir(dir)
        print(f"txt files that are being loaded are {txt_files}")
        count_dict = {}

        for txt_file in txt_files:
            if 'readme' in txt_file:
                continue
            m = re.match(r'det_test_(\w+).txt', txt_file)
            category = m.group(1)
            ## we there are specific classes we want, then ignore rest
            if relevant_classes is not None and category not in relevant_classes:
                continue
            self.logger.info(f"Working on category: {category}")
            full_name = os.path.join(dir, txt_file)

            if category not in count_dict.keys():
                count_dict[category] = 0

            with open(full_name) as file:
                file_content = file.read().splitlines()
                for line in file_content:
                    parsed_content = line.split(" ")
                    ## need to convert to float -> int (except confidence)
                    frame_id, confidence, min_x, min_y, max_x, max_y = list(map(float, parsed_content))
                    frame_id, min_x, min_y, max_x, max_y = list(map(int, [frame_id, min_x, min_y, max_x, max_y]))
                    ## we should only load things that have a certain threshold

                    while len(labels) <= frame_id:
                        labels.append([])
                        boxes.append([])
                    if confidence >= confidence_threshold:
                        labels[frame_id].append(category)
                        boxes[frame_id].append((min_x, min_y, max_x, max_y))
                        count_dict[category] += 1


        ## TODO: we might need to clean empty arrays
        self.logger.info(f"label statistics info: {count_dict}")


        return labels, boxes


    def convert_labels(self, labels, frame_limit_count, relevant_classes = None):
        assert (frame_limit_count < len(labels))

        ###generate the output list of list object
        output = []
        for i in range(frame_limit_count):
            output.append([])

        i = 0
        while labels['frame'][i] < frame_limit_count:
            obj = labels['object_name'][i]
            if relevant_classes is None:
                output[labels['frame'][i]].append(obj)
            else:
                if obj in relevant_classes:
                    output[labels['frame'][i]].append(obj)
            i += 1

        return output


    ##### Updated 3/3/2020 -- Filters the image input because the loaded data can contain None

    def filter_input(self, images:np.array, labels:list):
        ## we must have labels, but might not have images or boxes
        if labels is None:
            raise ValueError

        length = len(labels)
        count = 0
        for i in range(length):
            if labels[i] != []:
                count += 1

        new_images = np.ndarray(shape = (count, images.shape[1], images.shape[2], images.shape[3]))
        new_labels = []

        index = 0

        for i, elem in enumerate(labels):
            if elem != []:
                new_images[index] = images[i]
                index += 1
                new_labels.append(elem)

        return new_images, new_labels

    def get_boxes(self):
        """
        This function must be run after load_labels
        :param dir:
        :return:
        """
        if self.boxes is None:
            self.logger.error("get_labels must be run before this function to obtain the boxes!")
            raise ValueError

        self.logger.debug(f"Total number of frames loaded: {len(self.boxes)}")
        return self.boxes

    def get_video_start_indices(self):
        """
        This function returns the starting indexes for each video bc uadetrac has numerous videos of different perspectives
        :return: python list with starting indexes saved
        """
        return self.video_start_indices

    def save_images(self, name, vi_name):
        # we need to save the image / video start indexes
        # convert list to np.array
        """
        save_dir = os.path.join(self.eva_dir, 'data', cache_path, name)
        save_dir_vi = os.path.join(self.eva_dir, 'data', args.cache_path, vi_name)
        if self.images is None:
            self.logger.error("No image loaded, call load_images() first")
        elif type(self.images) is np.ndarray:
            np.save(save_dir, self.images)
            np.save(save_dir_vi, np.array(self.video_start_indices))
            self.logger.info(f"saved images to {save_dir}")
            self.logger.info(f"saved video indices to {save_dir_vi}")
        else:
            self.logger.error("Image array type is not np.....cannot save")
        """
        pass

if __name__ == "__main__":
    import time

    st = time.time()
    loader = SeattleLoader()

    loader.load_labels(relevant_classes = ['car', 'van', 'others'])

    """
    images = loader.load_images()
    labels = loader.load_labels()
    boxes = loader.load_boxes()

    print("Time taken to load everything from disk", time.time() - st, "seconds")
    loader.save_boxes()
    loader.save_labels()
    loader.save_images()

    st = time.time()
    images_cached = loader.load_cached_images()
    labels_cached = loader.load_cached_labels()
    boxes_cached = loader.load_cached_boxes()
    print("Time taken to load everything from npy", time.time() - st, "seconds")

    assert (images.shape == images_cached.shape)
    assert (boxes.shape == boxes_cached.shape)

    for key, value in labels.items():
        assert(labels[key] == labels_cached[key])
    assert(labels.keys() == labels_cached.keys())

    """

