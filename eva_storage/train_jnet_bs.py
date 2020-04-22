"""
Train jnet on bs'ed images


"""



import sys
home_dir = '/home/jbang36/eva_jaeho'
sys.path.append(home_dir)





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
    logger = Logger()



    st = time.time()
    # 1. Load the images (cached images is fine)
    images = loader.load_cached_images()
    labels = loader.load_cached_labels()
    video_start_indices = loader.get_video_start_indices()
    print(f"Done loading images in {time.time() - st} (sec)")



    directory_begin = '/nethome/jbang36/eva_jaeho/data/models/'
    """
    model_names = ['history20_dist2thresh300',
                   'history20_dist2thresh300_bloat_lvl2',
                   'history20_dist2thresh300_bloat_lvl3',
                   'history20_dist2thresh300_bloat_lvl4']

    model_names = ['history20_dist2thresh300_bloat_lvl2',
                   'history20_dist2thresh300_bloat_lvl3',
                   'history20_dist2thresh300_bloat_lvl4']
    
    
    """
    hist4 = 200
    dist4 = 300
    model_names = ['history200_dist2thresh300']

    logger.info(f"Generating segmented images with history value: {hist4}, distance value: {dist4}")
    segmented_images = preprocess.run(images, video_start_indices, history = hist4, dist2Threshold = dist4)
    model = UNet()
    logger.info(f"Training the network ....")
    model.train(images, segmented_images, save_name=model_names[0])


