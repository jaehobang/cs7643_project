"""
Train jnet on original train images -- 4/20/2020 is used for jnet sampling evaluation
"""
import sys

home_dir = '/home/jbang36/eva_jaeho'
sys.path.append(home_dir)


def train_conservative(train_images, preprocess, name):
    segmented_images = preprocess.run(train_images, None, load=True)
    model = UNet()
    model.train(train_images, segmented_images, save_name=name)
    return


if __name__ == "__main__":
    # %%
    import os
    import time

    from loaders.uadetrac_loader import UADetracLoader
    from eva_storage.preprocessingModule import PreprocessingModule
    from eva_storage.UNet import UNet
    from eva_storage.postprocessingModule import PostprocessingModule
    from logger import Logger

    loader = UADetracLoader()
    preprocess = PreprocessingModule()
    bare_model = UNet()
    postprocess = PostprocessingModule()
    logger = Logger()

    st = time.time()
    # 1. Load the images (cached images is fine)
    #train_images = loader.load_cached_images(name = 'uad_train_images.npy', vi_name = 'uad_train_vi.npy')
    train_images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/test_images')

    logger.info(f"Done loading images in {time.time() - st} (sec)")


    directory_begin = '/nethome/jbang36/eva_jaeho/data/models/plain'

    model_name = 'unet_plain_testdata_0505'



    ## train the models
    import numpy as np

    #train_images = train_images[::15]

    level_model = UNet()
    ## TODO: because the UNet_final takes in 1 channel as output, we are creating the network to learn a black and white image of the original image
    st = time.perf_counter()
    level_model.train(train_images, None, model_name)
    logger.info(f"Total time to train the network is {time.perf_counter() - st} (sec)")
