"""
This file consists of the decompressionModule class that works with compressed videos to make them decompressed.
Some further optimizations could be possible (such as utilization of multiple threads, but at the moment everything is serial)

@Jaeho Bang
"""

import cv2
import numpy as np
import os
import time
from logger import Logger



class DecompressionModule:
    def __init__(self):
        self.image_matrix = None
        self.video_stats = {} #This will keep data of all the videos that have been parsed.. will not keep the image matrix only meta-data
        self.logger = Logger()
        self.curr_video = ''

    def reset(self):
        self.image_matrix = None


    def add_meta_data(self, path, frame_count, width, height, fps):
        self.video_stats[path] = {}
        self.video_stats[path]['fps'] = fps
        self.video_stats[path]['width'] = width
        self.video_stats[path]['height'] = height
        self.video_stats[path]['frame_count'] = frame_count


    def get_frame_count(self):
        return self.video_stats[self.curr_video]['frame_count']


    def convert2images(self, path, frame_count_limit = 60000):
        self.vid_ = cv2.VideoCapture(path)
        if (self.vid_.isOpened() == False):
            self.logger.error(f"Error opening video {path}")
            raise ValueError

        self.curr_video = path

        frame_count = min(self.vid_.get(cv2.CAP_PROP_FRAME_COUNT), frame_count_limit)
        width = self.vid_.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.vid_.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.vid_.get(cv2.CAP_PROP_FPS)
        channels = 3


        self.add_meta_data(path, frame_count, width, height, fps)

        assert (frame_count == int(frame_count))
        assert (width == int(width))
        assert (height == int(height))

        frame_count = int(frame_count)
        width = int(width)
        height = int(height)

        self.logger.info(f"meta data of the video {path} is {frame_count, height, width, channels}")
        self.image_matrix = np.ndarray(shape=(frame_count, height, width, channels), dtype = np.uint8)


        for i in range(frame_count):
            success, image = self.vid_.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not success:
                print("Image retrieval has failed")
            else:
                self.image_matrix[i, :, :, :] = image  # stored in rgb format

        return self.image_matrix


if __name__ == "__main__":
    eva_dir = os.path.abspath('../')
    data_dir = os.path.join(eva_dir, 'data', 'videos')
    dc = DecompressionModule()
    files = os.listdir(data_dir)

    full_name = os.path.join(data_dir, files[0])
    tic = time.time()
    print("--------------------------")
    print("Starting ", files[0])
    dc.convert2images(full_name)
    print("Finished conversion, total time taken is:", time.time() - tic, "seconds")
    print("Image matrix shape:", dc.image_matrix.shape)
