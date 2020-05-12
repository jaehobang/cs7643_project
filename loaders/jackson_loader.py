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


from logger import Logger, LoggingLevel



# Make this return a dictionary of label to data for the whole dataset

class JacksonLoader(AbstractLoader):
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

    def load_images(self, dir: str = None, image_size=None):
        """
        This function simply loads image of given image
        :return: image_array (numpy)
        """
        if image_size is not None:
            if type(image_size) == int:
                self.image_height = image_size
                self.image_width = image_size
            elif type(image_size) == list or type(image_size) == tuple:
                self.image_width, self.image_height = image_size

        if dir == None:
            dir = os.path.join('/nethome/jbang36/eva_jaeho/data/jackson_town_square', 'jackson_town_square.mp4')
            self.logger.info(f"Image directory to load not given.. using default: {dir}")

        st = time.perf_counter()
        self.images = self.decompressionModule.convert2images(dir)
        self.logger.info(f"Loaded {len(self.images)} in {time.perf_counter() - st} (secs)")

        n, width, height, channels = self.images.shape
        images = np.ndarray(shape = (n, self.image_width, self.image_height, channels))
        if self.image_width != width or self.image_height != height:
            for i in range(n):
                img = cv2.resize(self.images[i], (self.image_width, self.image_height))
                images[i] = img


        self.logger.info(f"Total time to load {n} images: {time.perf_counter() - st} (sec)")

        return self.images

    def load_labels(self, dir: str = None):
        """
        Loads vehicle type, speed, color, and intersection of ua-detrac
        vehicle type, speed is given by the dataset
        color, intersection is derived from functions built-in
        :return: labels
        """

        if dir == None:
            dir = os.path.join('/nethome/jbang36/eva_jaeho/data', 'jackson-town-square.csv')

        frame_count_limit = self.decompressionModule.get_frame_count()

        labels = pd.read_csv(dir)
        # Preview the first 5 lines of the loaded data

        converted_labels = self.convert_labels2(labels, frame_count_limit)
        return converted_labels


    def convert_labels2(self, labels, frame_limit_count):
        assert (frame_limit_count < len(labels))
        ###generate the output list of list object
        output = []
        for i in range(frame_limit_count):
            output.append([])

        i = 0
        while labels['frame'][i] < frame_limit_count:
            output[labels['frame'][i]].append(labels['object_name'][i])
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

    """
    def save_labels(self, name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        if self.labels is None:
            self.logger.error("No labels loaded, call load_labels() first")

        elif type(self.labels) is dict:
            np.save(save_dir, self.labels, allow_pickle=True)
            self.logger.info(f"saved labels to {save_dir}")

        else:
            self.logger.error(f"Expected labels type to be dict, got {type(self.labels)}")

    def save_boxes(self, name=args.cache_box_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        if self.images is None:
            self.logger.error("No labels loaded, call load_boxes() first")

        elif type(self.images) is np.ndarray:
            np.save(save_dir, self.boxes)
            self.logger.info(f"saved boxes to {save_dir}")

        else:
            self.logger.error("Labels type is not np....cannot save")

    def load_cached_images(self, name=args.cache_image_name, vi_name=args.cache_vi_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        save_dir_vi = os.path.join(self.eva_dir, 'data', args.cache_path, vi_name)
        self.images = np.load(save_dir)
        self.video_start_indices = np.load(save_dir_vi)
        return self.images

    def load_cached_boxes(self, name=args.cache_box_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        self.boxes = np.load(save_dir, allow_pickle=True)
        return self.boxes

    def load_cached_labels(self, name=args.cache_label_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        labels_pickeled = np.load(save_dir, allow_pickle=True)
        self.labels = labels_pickeled.item()
        return self.labels
    """

if __name__ == "__main__":
    import time

    st = time.time()
    loader = JacksonLoader()

    ## frame mismatch between xml and actual files in MVI_39811
    images = loader.load_cached_images()
    print(images.shape)
    video_start_indices = loader.get_video_start_indices()
    labels = loader.load_labels(dir='/nethome/jbang36/eva/data/ua_detrac/DETRAC-Train-Annotations-XML')

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

