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



class Encoder:

    def __init__(self):
        pass

    def indices2timestamps(self, rep_indices, frame_info):
        """
        Converts rep indices to timestamps -- we will use the frame_info to do wso

        :param rep_indices:
        :param timestamps:
        :return:
        """
        ## now we can convert from video indices to timstamps

        ##TODO: convert rep_indices to timestamps_list and return it
        ## now we need to realize that we have the frame_info and we have rep_indices -- let's try this on jupyter first

        ### so now we need a helper function to convert the indices to timestamps
        timestamps_list = []

        for i, val in enumerate(rep_indices):
            timestamps_list.append(frame_info['frames'][val]['pkt_pts_time'])

        return timestamps_list



    def run(self, images, rep_indices, frame_info, save_directory):
        """

        :param images: images we are trying to form into the compressed format
        :param rep_indices: the frames that we be hardcoded into an i frame
        :param save_directory:
        :return:
        """
        ## once we have the images, rep_indices, and where to save the video, we can move onto creating the command to generate the new video
        ### move as pipes,
        ## we need to create the timestamps list -- conversion from rep_indices is necessary
        ## we need to convert from rep_indices to timestamps list
        timestamps_list = self.indices2timestamps(rep_indices, frame_info) ## I need access to the preprocessor
        FfmpegCommands.force_keyframes(images, timestamps_list, save_directory, framerate=60)

        return ##DONE!!




    def get_iframes(self, frameinfo):

        i_frame_list = []

        for i, frame in enumerate(frameinfo['frames']):
            if frame['pict_type'] == 'I':
                i_frame_list.append(i)

        return i_frame_list



if __name__ == "__main__":
    """
    Encoding pipeline from a regular seattle video
    """

    loader = SeattleLoader()
    video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_15000.mp4'
    images, meta_data = loader.load_images(video_directory)
    frame_info = FfmpegCommands.get_frameinfo(video_directory)

    ## preprocessing the video
    preprocessor = Preprocessor()
    video_filename = os.path.basename(video_directory)
    video_filename = video_filename.split('.')[0]

    rep_indices = preprocessor.run(images, video_filename)

    new_video_directory = os.path.join( os.path.dirname(video_directory), 'seattle2_15000_jvc.mp4' )
    encoder = Encoder()
    encoder.run(images, rep_indices, frame_info, new_video_directory)







