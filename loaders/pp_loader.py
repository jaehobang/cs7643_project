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

    def __init__(self, dataset_name, predicate_name, udf_load_dir = None):
        """
        We need to load the filters and udfs
        We do not utilize the pp optimizations at this point
        """

        ### currently udf loads by default but filter does not load by default.
        self.logger = Logger()
        self.logger.info("init pploader")
        self.udf = SSDLoader(model_dir = udf_load_dir)
        self.logger.info("Loaded udf")
        self.filter = FilterResearch(dataset_name, predicate_name)
        self.logger.info("Loaded filter")

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



    def predict(self, images, query='car=1'):
        ## we need the function behind converting labels...?
        labels_preliminary = self.filter.predict(images)
        images_filtered = images[labels_preliminary == 1]
        ## we need the mapping
        box_dict = self.udf.predict(images_filtered, cuda = True) ##TODO: this is not straight forward because I think the returned label is like dictionary or something
        ## since box_dict predicts the binary classifications, we simply need to know which class we are interested and return the result
        ## TODO: We need to figure out how the system can determine which queries are relevant
        ## but for now, we will just manually code this
        labels_that_we_want_from_udf = None
        for cls_ind, cls_name in enumerate(box_dict):
            if cls_name in query:
                labels_that_we_want_from_udf = box_dict[cls_name]



        labels = np.zeros(len(images))
        label_ones = np.nonzero(labels_preliminary)[0]
        labels[label_ones] = labels_that_we_want_from_udf
        #labels[~labels_preliminary] = 0

        return labels




if __name__ == "__main__":
    ### we should train the filters for future use
    ### Train Filter Script

    from loaders.seattle_loader import SeattleLoader
    from loaders.uadetrac_label_converter import UADetracConverter

    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_train.mp4'

    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    annotation_save_dir = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_train_annotations/contains_others.txt'
    contains_car_label = loader.load_predicate_labels(annotation_save_dir)

    pp_loader = PPLoader(dataset_name = 'seattle', predicate_name = 'others=1')
    data_loader = SeattleLoader()
    converter = UADetracConverter()
    images = data_loader.load_images() ## load the default video -- seattle2.mov, how do we deal with the model drift problem for linear SVC

    labels, boxes = data_loader.load_labels(relevant_classes = ['car', 'others', 'van'])
    limit_labels = converter.convert2limit_queries2(labels, {'others': 1}, operator = 'or')
    #limit_labels = converter.convert2limit_queries2(labels, {'car': 1}, operator = 'or')
    ## training the filters
    ## we don't need to input all the images to train the models

    pp_loader.train(images, limit_labels, pp_constants.FILTER)











