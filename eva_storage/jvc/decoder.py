
from loaders.seattle_loader import SeattleLoader
from loaders.decompressionModule import DecompressionModule

class Decompressor:


    def __init__(self):
        self.decompressionModule = DecompressionModule()
        self.frame_limit_count = 1000000

        self.loader = SeattleLoader()



    def decompress(self, path_to_video, number_of_samples = None):
        """
        Decompress the video
        :param path_to_video: str
        :param number_of_samples: number of samples we want from the video, if None, we decompress the whole thing
        :return:
        """

        if number_of_samples:
            self.rep_indices = self.interpret_metadata(path_to_video, number_of_samples)
            time_frames = self.convert_indices_to_time()
            images = self.decompressionModule.decompress_sampled(dir, time_frames, frame_count_limit=self.frame_count_limit)
            assert(len(images) == number_of_samples)
        else:
            images = self.decompressionModule.convert2images(dir, frame_count_limit=self.frame_limit_count)

        return images


    def interpret_metadata(self, path_to_video, number_of_samples):
        """
        TODO: will do this on preprocessor_prototyping.ipynb
        :param path_to_video:
        :param number_of_samples:
        :return:
        STEPS:
        1. need to find a way to convert path_to_video to path_to_metadata

        """

        return


    def convert_indices_to_time(self):
        """
        We think of a method to convert indices to timeframe
        :return:
        """
        pass
