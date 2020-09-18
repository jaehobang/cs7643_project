import sys
import os
import numpy as np




sys.argv = ['']
sys.path.append('/nethome/jbang36/eva_jaeho')



from loaders.seattle_loader import SeattleLoader



"""
In this file we compare various loading methods

libjpeg-turbo
cv2
loading video using cv2
loading video using ffmpeg
"""

def load_images_turbo():
    from turbojpeg import TurboJPEG
    base_directory = '/nethome/jbang36/eva_jaeho/data/seattle'
    image_directory = os.path.join(base_directory, 'seattle2_100000_images')
    image_files = os.listdir(image_directory)
    image_list = []
    turbo_directory = '/nethome/jbang36/libjpeg_turbo/libturbojpeg.so'
    jpeg = TurboJPEG(turbo_directory)
    for image_file in image_files:
        full_path = os.path.join(image_directory, image_file)
        in_file = open(full_path, 'rb')
        image_list.append( jpeg.decode(in_file.read()) )
        in_file.close()
    image_arr = np.stack(image_list, axis = 0)
    return image_arr

def load_images_cv2():
    import cv2
    base_directory = '/nethome/jbang36/eva_jaeho/data/seattle'
    image_directory = os.path.join(base_directory, 'seattle2_100000_images')
    image_files = os.listdir(image_directory)
    image_list = []
    for image_file in image_files:
        full_path = os.path.join(image_directory, image_file)
        image_list.append( cv2.imread(full_path) )
    image_arr = np.stack(image_list, axis=0)

    return image_arr

def load_video_cv2():
    base_directory = '/nethome/jbang36/eva_jaeho/data/seattle'
    video_directory = os.path.join(base_directory, 'seattle2_100000.mp4')
    loader = SeattleLoader()
    images = loader.load_images(video_directory)
    return images

def load_video_ffmpeg():
    import subprocess
    base_directory = '/nethome/jbang36/eva_jaeho/data/seattle'
    video_directory = os.path.join(base_directory, 'seattle2_100000.mp4')
    command = f"ffmpeg -i {video_directory} -vsync 0 -f rawvideo -pix_fmt rgb24 pipe:"
    args = command.split(" ")
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    communicate_kwargs = {}
    out, err = p.communicate(**communicate_kwargs)
    if p.returncode != 0:
        print(f"Return code is {p.returncode}")
        raise ValueError
    ##### time it takes to fetch the data
    height = 240
    width = 360
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])


    return images

if __name__ == "__main__":
    import time

    save_directory = os.path.join('/nethome/jbang36/eva_jaeho/data/benchmark_results/speed', 'data_loading_time.txt')
    file_descriptor = open(save_directory, 'a+')

    st = time.perf_counter()
    images = load_images_cv2()
    file_descriptor.write(f"method: load_images_cv2 took {time.perf_counter() - st} (seconds) shape is {images.shape}\n")

    st = time.perf_counter()
    images = load_images_turbo()
    file_descriptor.write(f"method: load_images_turbo took {time.perf_counter() - st} (seconds) shape is {images.shape}\n")

    st = time.perf_counter()
    images = load_video_cv2()
    file_descriptor.write(f"method: load_video_cv2 took {time.perf_counter() - st} (seconds) shape is {images.shape}\n")

    st = time.perf_counter()
    images = load_video_ffmpeg()
    file_descriptor.write(f"method: load_video_ffmpeg took {time.perf_counter() - st} (seconds) shape is {images.shape}\n")
