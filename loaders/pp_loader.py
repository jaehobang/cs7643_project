"""
This file is used to load PPs
Wrapper around filters and udf
Currently doesn't have optimization implemented, we assume very simple query execution

@Jaeho Bang
"""

from loaders.ssd_loader import SSDLoader
from filters.research_filter import FilterResearch
import numpy as np
import pp_constants
from logger import Logger


class PPLoader:

    def __init__(self, udf_load_dir = None, filter_load_dir = None):
        """
        We need to load the filters and udfs
        We do not utilize the pp optimizations at this point

        """
        ### currently udf loads by default but filter does not load by default.
        self.udf = SSDLoader(model_dir = udf_load_dir)
        filter_load_dir = '/nethome/jbang36/data/filters/default.txt'
        self.filter = FilterResearch(load_dir = filter_load_dir)
        ##
        self.logger = Logger()

    def train(self, images, labels, option):
        if option == pp_constants.ALL:
            self.filter.train(images, labels)
            self.udf.train(images, labels, cuda = True)
        elif option == pp_constants.FILTER:
            self.filter.train(images, labels)
        elif option == pp_constants.UDF:
            self.udf.train(images, labels, cuda = True)
        else: ## not a valid option
            self.logger.info(f"Option {option} is not valid")
            raise ValueError

        return

    def detect(self, images):
        labels_preliminary = self.filter.predict(images)
        images_filtered = images[labels_preliminary]
        ## we need the mapping
        labels_final, boxes_final = self.udf.predict(images_filtered) ##TODO: this is not straight forward because I think the returned label is like dictionary or something
        labels = np.zeros(len(images))
        labels[labels_preliminary] = labels_final
        labels[~labels_preliminary] = 0
        return labels





