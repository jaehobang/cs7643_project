"""
We want to evaluate multiple pipelines and how they perform on an end-to-end basis
The scenarios we want to deal with are as follows:
UDF
Sample -> UDF
Filter -> UDF (PP)
Sample -> Filter -> UDF

"""

from loaders.seattle_loader import SeattleLoader
import time

if __name__ == "__main__":
    import time
    st = time.perf_counter()

    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2.mov'

    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    labels = loader.load_labels(video_directory, relevant_classes = ['car', 'others', 'van'])

    #### example_query = 'select * from Seattle where car == 1'

    ## we need to invoke the ssd method for evaluation and return the labels to all these frames



