"""
In this file, we will define the compressor of new videos.

Current plan:
1. we will try using force_key_frames to do new encoding.
2. We assume the other module gives a meta-data format that we use to encode / decode

"""

from loaders.seattle_loader import SeattleLoader
from eva_storage.jvc.ffmpeg_commands import FfmpegCommands
from eva_storage.jvc.preprocessor import Preprocessor
import os



class Compressor:

    def __init__(self):
        pass

    def indices2timestamps(self, rep_indices, timestamps):
        """
        Converts rep indices to timestamps -- we will use the frame_info to do wso

        :param rep_indices:
        :param timestamps:
        :return:
        """
        frame_info = FfmpegCommands.get_frameinfo(video_directory)  ## this gives pict type for every frame in video
        ## now we can convert from video indices to timstamps



    def run(self, images, rep_indices, save_directory):
        """

        :param images: images we are trying to form into the compressed format
        :param rep_indices: the frames that we be hardcoded into an i frame
        :param save_directory:
        :return:
        """
        ## once we have the images, rep_indices, and where to save the video, we can move onto creating the command to generate the new video
        ### move as pipes,
        ## we need to create the timestamps list -- conversion from rep_indices is necessary



        FfmpegCommands.force_keyframes(images, timestamps_list, save_directory, framerate=60)




    def get_iframes(self, frameinfo):

        i_frame_list = []

        for i, frame in enumerate(frameinfo['frames']):
            if frame['pict_type'] == 'I':
                i_frame_list.append(i)

        return i_frame_list



if __name__ == "__main__":
    """
    1. load the video,
    2. generate the metadata
    3. transform that metadata into and force key frames
    """

    loader = SeattleLoader()
    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_short.mp4'
    images, meta_data = loader.load_images(video_directory)

    ## preprocessing the video
    preprocessor = Preprocessor()
    video_filename = os.path.basename(video_directory)
    ###TODO: eliminate the extension
    video_filename = video_filename.split('.')[0]

    rep_indices = preprocessor.run(images, video_filename)
    rep_metadata = preprocessor.get_tree()

    frame_info = FfmpegCommands.get_frameinfo(video_directory)

    #### we now have rep_indices and frame_info
    #### as a starter,








