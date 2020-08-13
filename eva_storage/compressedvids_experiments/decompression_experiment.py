"""
In this file, we will experiment decompressing all frames vs i frames
We will try profiling the functions to see where the bulk of the time is taking place

"""

from loaders.decompressionModule import DecompressionModule
import numpy as np
import ffmpeg
import time



def cv_decompression(directory):
    st = time.perf_counter()
    decompression_module = DecompressionModule()
    frame_count_limit = 1000000

    images_all = decompression_module.convert2images(directory, frame_count_limit=frame_count_limit)
    print(f">>>>>>>decompressed video from decompression module is {images_all.shape} total time: {time.perf_counter() - st} seconds")



def ffmpeg_decompression(directory):
    st = time.perf_counter()
    probe = ffmpeg.probe(directory)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    print(f"total time to probe the video data {time.perf_counter() - st} seconds")
    st2 = time.perf_counter()
    out, _ = (
        ffmpeg
            .input(dir)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
    )
    print(f"total time to extract the video {time.perf_counter() - st2} seconds")


    st3 = time.perf_counter()
    video = (
        np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
    )
    print(f"total time to convert to numpy array is {time.perf_counter() - st3} seconds")
    print(f">>>>>>>>>decompressed video from ffmpeg is {video.shape} total time: {time.perf_counter() - st} seconds")


if __name__ == "__main__":


    dir = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_final.mp4'
    ### I also want to try out how fast ffmpeg-python is
    cv_decompression(dir)
    ffmpeg_decompression(dir)




