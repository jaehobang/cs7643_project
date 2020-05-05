"""
In this file, we will compare the memory consumption between compressed videos, compressed images, numpy array
Let's just take, MVI_40131 in UAD for example

UAD: 25 fps
960x540 pixels
"""



"""
TODOs:
write a compression script, compressed images -> videos"""

import os
import sys
sys.argv=['']
sys.path.append('/nethome/jbang36/eva_jaeho')
import cv2
import numpy as np
import time

from loaders.uadetrac_loader import UADetracLoader



def generate_video(images, save_dir, fps, frame_width, frame_height, code = 'XVID'):
    fourcc = cv2.VideoWriter_fourcc(*code)
    out = cv2.VideoWriter(save_dir, fourcc, fps, (frame_width, frame_height))

    for i,frame in enumerate(images):
        out.write(frame)


    out.release()

    return


if __name__ == "__main__":
    video_name = 'MVI_40131.mp4'
    loader = UADetracLoader()
    tic = time.perf_counter()
    images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/40131_images', image_size=[960, 540])

    print(f"loaded {len(images)} images in {time.perf_counter() - tic} (sec)!!!")

    s = '/nethome/jbang36/eva_jaeho/data/npy_files'
    save_dir = os.path.join(s, video_name)
    fps = 25
    frame_width = 960
    frame_height = 540

    generate_video(images, save_dir, fps, frame_width, frame_height, code='DIVX')
    np_save_dir = os.path.join(s, 'MVI_40131.npy')
    np.save(np_save_dir, images)



