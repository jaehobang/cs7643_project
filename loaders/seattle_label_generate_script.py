"""
Script to generate annotations for SEATTLE

"""


import os
import sys
sys.argv=['']
## note ada-03 is '/nethome/jbang36/eva_jaeho'
##      ada-01 is '/nethome/jbang36/eva_storage'
sys.path.append('/nethome/jbang36/eva_jaeho')
sys.path.append('/nethome/jbang36/eva_storage')

import time


import numpy as np
#import utils.helpers as helpers
from loaders.uadetrac_loader import UADetracLoader
from eva_storage.preprocessingModule import PreprocessingModule
from eva_storage.UNet import UNet
from eva_storage.clusterModule import ClusterModule
from filters.minimum_filter import FilterMinimum

from loaders.decompressionModule import DecompressionModule
from miscellaneous_code import *
from loaders.seattle_loader import *



## ada-01: eva_storage -> eva_jaeho
video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_short.mp4'

st = time.perf_counter()
decompressionModule = DecompressionModule()
images = decompressionModule.convert2images(video_directory, frame_count_limit = 1000000)
print(f"finished loading {len(images)} in {time.perf_counter() - st}")

st = time.perf_counter()
label_generator = SeattleLabelGenerator()
label_generator.generate_annotations(images)
print(f"finished generating labels in {time.perf_counter() - st}")

