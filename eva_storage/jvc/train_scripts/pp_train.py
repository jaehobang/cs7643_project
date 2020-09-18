
import sys

sys.argv = ['']
sys.path.append('/nethome/jbang36/eva_jaeho')

from loaders.seattle_loader import SeattleLoader
from loaders.pp_loader import PPLoader
import pp_constants


import numpy as np
import os





if __name__ == "__main__":
    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_train.mp4'

    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    label_directory = os.path.join(os.path.dirname(video_directory), 'seattle2_train_annotations')
    labels = {}
    labels['car'] = np.array(loader.load_predicate_labels(os.path.join(label_directory, 'det_test_car.txt')))
    labels['others'] = np.array(loader.load_predicate_labels(os.path.join(label_directory, 'det_test_others.txt')))
    labels['van'] = np.array(loader.load_predicate_labels(os.path.join(label_directory, 'det_test_van.txt')))

    dataset_name = 'seattle2'
    label_predicate_pairs = [('car', 'car=1'), ('others', 'others=1'), ('van', 'van=1')]
    for label, predicate in label_predicate_pairs:
        print(f"Working on {label}")
        model = PPLoader(dataset_name, predicate)
        model.train(images, labels[label], pp_constants.FILTER)