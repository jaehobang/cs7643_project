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



    def predict(self, images):
        ## we need the function behind converting labels...?
        labels_preliminary = self.filter.predict(images)
        images_filtered = images[labels_preliminary]
        ## we need the mapping
        labels_final, boxes_final = self.udf.predict(images_filtered) ##TODO: this is not straight forward because I think the returned label is like dictionary or something
        ### what is the output of udf.predict?? -> [labels], [boxes]
        ## labels are in the format of [['car', 'bus', 'others'], ['car', 'car', 'others']...]

        labels = np.zeros(len(images))
        labels[labels_preliminary] = labels_final
        labels[~labels_preliminary] = 0

        return labels




if __name__ == "__main__":
    ### we should train the filters for future use
    from loaders.seattle_loader import SeattleLoader
    from loaders.uadetrac_label_converter import UADetracConverter

    pp_loader = PPLoader()
    data_loader = SeattleLoader()
    converter = UADetracConverter()
    images = data_loader.load_images() ## load the default video -- seattle2.mov, how do we deal with the model drift problem for linear SVC

    labels, boxes = data_loader.load_labels(relevant_classes = ['car', 'others', 'van'])
    limit_labels = converter.convert2limit_queries2(labels, {'car': 1}, operator = 'or')
    ## training the filters
    ## we don't need to input all the images to train the models

    pp_loader.train(images, limit_labels, pp_constants.FILTER)











