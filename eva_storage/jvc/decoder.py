
from loaders.seattle_loader import SeattleLoader
from loaders.decompressionModule import DecompressionModule
from eva_storage.jvc.ffmpeg_commands import FfmpegCommands
import os
import numpy as np


class Decoder:

    def __init__(self):
        self.decompressionModule = DecompressionModule()
        self.frame_limit_count = 1000000
        self.loader = SeattleLoader()
        self.hierarchy = None

    def reset(self):
        self.hierarchy = None



    def run(self, video_directory, hierarchy_directory, number_of_samples = None):
        """
        Decompress the video
        :param path_to_video: str
        :param number_of_samples: number of samples we want from the video, if None, we decompress the whole thing
        :return:
        """
        print(f"decompressing from directory: {video_directory}")
        print(f"Loading hierarchy from directory: {hierarchy_directory}")



        if number_of_samples:
            ##TODO: we need the path to hierarchy

            rep_indices = self.interpret_metadata(hierarchy_directory, number_of_samples)
            video_iframe_indices = FfmpegCommands.get_iframe_indices(video_directory)
            ## now we have a the iframe indices and the rep indices....
            video_iframes = FfmpegCommands.get_iframes(video_directory)
            print(f"video_iframes shape is {video_iframes.shape}")
            ## now that we have the iframes, we need to know which i frames to extract...
            ## we use rep_indices and video_iframe_indices to determine this
            print(f"rep indices shape is {rep_indices.shape}")
            print(f"video iframe indices shape is {video_iframe_indices.shape}")
            projected_rep_indices = self.translate_rep_indices(rep_indices, video_iframe_indices)
            print(f"projected_rep_indices shape is {projected_rep_indices.shape}")
            return video_iframes[projected_rep_indices]
        else:
            ## if the number of samples is not included as argument just get decompress the whole video
            images = self.decompressionModule.convert2images(dir, frame_count_limit=self.frame_limit_count)

            return images



    def translate_rep_indices(self, rep_indices, video_iframe_indices):
        """
        video_iframe_indices are in the same order as video_iframes but have the actual frameid as values
        rep_indices are the in frameid

        hence, we need to project rep_indices values to video_iframe_indices axis

        :param video_iframes:
        :param rep_indices:
        :param video_iframe_indices:
        :return:
        """

        projected_rep_indices = []
        for value in rep_indices:
            result = np.where(video_iframe_indices == value)[0][0]
            projected_rep_indices.append(result)
        #print(f"projected_rep_indices len:{len(projected_rep_indices)}, element is like: {projected_rep_indices[0]}")
        projected_rep_indices = np.array(projected_rep_indices)

        assert(len(projected_rep_indices) == len(rep_indices))
        return projected_rep_indices




    def interpret_metadata(self, hierarchy_directory, number_of_samples):
        """

        :param path_to_video: path to the video we are looking into
        :param number_of_samples: number of samples the user requests to decoding the video
        :return:
        STEPS:
        1. need to find a way to convert path_to_video to path_to_metadata

        """
        print(f'loading hierarchy from {hierarchy_directory}')
        self.hierarchy = np.load(hierarchy_directory, allow_pickle = True)
        ## now that we have hierarchy, we can select the frames that we want
        rep_indices = self.hierarchy[:number_of_samples]
        ## we have already checked that we don't need to reorder the array, and this holds true even when we are doing evaluation as well


        return rep_indices



