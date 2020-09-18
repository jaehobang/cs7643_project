"""
In this file, we generate videos of given frame numbers and measure the storage space they take up
we save the results in ~/storage/~
"""

import sys
import os
import time
import random
import numpy as np




sys.argv = ['']
sys.path.append('/nethome/jbang36/eva_jaeho')



from loaders.uadetrac_loader import UADetracLoader
from loaders.seattle_loader import SeattleLoader
from loaders.ssd_loader import SSDLoader
from loaders.pp_loader import PPLoader
from loaders.uadetrac_label_converter import UADetracConverter

from eva_storage.sampling_experiments.sampling_utils import sample3_middle
from eva_storage.sampling_experiments.sampling_utils import evaluate_with_gt5
from eva_storage.sampling_experiments.noscope_sample_ssd_evaluation import set_frame_count, get_rep_indices_noscope
from eva_storage.jvc.preprocessor import Preprocessor
from eva_storage.jvc.jvc import JVC

import sklearn.metrics as metrics
import cv2


def generate_images(images, number_of_images):
    ## save the images in a folder
    folder = os.path.join('/nethome/jbang36/eva_jaeho/data/seattle/', f'seattle2_{number_of_images}_images')
    os.makedirs(folder, exist_ok=True)
    for i in range(number_of_images):
        file_name = f'image_{i}.jpeg'
        full_name = os.path.join(folder, file_name)
        cv2.imwrite(full_name, images[i])
    return

def generate_video(images, number_of_images):
    images_to_process = images[:number_of_images]
    filename = os.path.join('/nethome/jbang36/eva_jaeho/data/seattle/', f'seattle2_{number_of_images}.mp4')
    from eva_storage.jvc.ffmpeg_commands import FfmpegCommands

    FfmpegCommands.write_video(images_to_process, filename)
    return

if __name__ == "__main__":
    ## we assume framerate to be 60
    default_framerate = 60
    video_length = np.array([10, 100, 1000]) ## seconds
    number_of_samples = video_length * default_framerate
    loader = SeattleLoader()
    load_dir = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_100000.mp4'
    images = loader.load_images(load_dir)

    for number in number_of_samples:
        #generate_images(images, number)
        generate_video(images, number)
    print('done!')

    """
    Since this file does not have a txt, we will save the information here
    600 images - 24MB
    6000 images - 235MB
    60000 images - 2.3GB
    600 video - 84KB
    6000 video - 752KB
    60000 video - 7.4MB
    
    
    """
