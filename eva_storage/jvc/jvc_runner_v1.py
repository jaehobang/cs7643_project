"""
This is the runner of the entire eva.jvc system.

Version 1,
the steps for the entire pipeline are as follows:
1. preprocessor -- get rep indices, save the metadata
2. nothing needs to be done with encoder; we leave the original format as is
3. during decoding:
    a. get i-frame indices
    b. get metadata -- convert to rep indices with user's given number of samples input
    c. when selecting rep_frame constraint, it has be an i-frame => we calculate with least dist diff
    d. send to ffmpeg (or custom code) frames you want decoded




@Jaeho Bang
"""

import os

from eva_storage.jvc.preprocessor import *
from eva_storage.jvc.encoder import *
from eva_storage.jvc.decoder import *

from loaders.seattle_loader import SeattleLoader

from eva_storage.temporalClusterModule import TemporalClusterModule
from eva_storage.featureExtractionMethods import *
from eva_storage.samplingMethods import *
from eva_storage.sampling_experiments.sampling_utils import *
from loaders.uadetrac_label_converter import UADetracConverter


class JVCRunner_POC:
    """
    POC stands for proof of concept
    """

    def __init__(self):
        """
        Steps:
        1. Load the video / images
        """
        self.video_directory = '/nethome/jbang36/eva_jaeho/data/seattle/seattle2_short.mp4'
        self.label_directory = os.path.join(os.path.dirname(self.video_directory), 'seattle2_short_annotations')


        loader = SeattleLoader()
        self.images = loader.load_images(self.video_directory)
        self.labels, self.boxes = loader.load_labels(self.label_directory, relevant_classes = ['car', 'others', 'van'])
        self.limit_labels = UADetracConverter.convert2limit_queries2(self.labels, {'van': 1}, operator = 'or')
        self.limit_labels = np.array(self.limit_labels)


        self.number_of_clusters = 100 ## let's use 1000 samples to determine its effectiveness

        print(f">>>>>>>>>>>>Starting jvc runner 1 ")
        self.jvc_runner1()

        ###### lower bound
        print(f">>>>>>>>>>>>Lower bound uniform sampling ")
        self.baseline()

        print(f">>>>>>>>>>>>Upper bound jvc runner 2 ")
        self.jvc_baseline()

        print(f">>>>>>>>>>>>DONE")



    def baseline(self):
        cluster_count = self.number_of_clusters
        total_eval_num = cluster_count
        images = self.images
        limit_labels = self.limit_labels
        boxes = self.boxes

        sampling_rate = int(len(images) / total_eval_num)

        rep_images, rep_labels, rep_boxes, mapping = sample3_middle(images, limit_labels, boxes,
                                                                    sampling_rate=sampling_rate)

        evaluate_with_gt5(limit_labels, rep_labels, mapping)



    def jvc_baseline(self):
        feature_extraction_method = DownSampleSkippingCVMethod()
        sampling_method = MiddleEncounterMethod()
        cluster_module = TemporalClusterModule(downsample_method=feature_extraction_method,
                                               sampling_method=sampling_method)
        _, rep_indices, all_cluster_labels = cluster_module.run(self.images, number_of_clusters=self.number_of_clusters)
        rep_labels = self.limit_labels[rep_indices]
        mapping = cluster_module.get_mapping(rep_indices, all_cluster_labels)

        evaluate_with_gt5(self.limit_labels, rep_labels, mapping)


    def jvc_runner1(self):
        import json
        import subprocess
        args = ['ffprobe', '-select_streams', 'v', '-show_frames', '-show_entries', 'frame=pict_type', '-of', 'json',
                self.video_directory]

        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            print('Communication Error.. Raising Value Error')
            raise ValueError
        self.final_output = json.loads(out.decode('utf-8'))

        i_frame_list = []

        for i, frame in enumerate(self.final_output['frames']):
            if frame['pict_type'] == 'I':
                i_frame_list.append(i)

        ## now we have all the i frame indices
        ### TODO: upload the cluster module when doing the selection, we have to select based on i frame constraint
        feature_extraction_method = DownSampleSkippingCVMethod()
        sampling_method = IFrameConstraintMethod(i_frame_list)
        cluster_module = TemporalClusterModule(downsample_method = feature_extraction_method, sampling_method = sampling_method)
        _, rep_indices, all_cluster_labels = cluster_module.run(self.images, number_of_clusters=self.number_of_clusters)
        ####TODO: Still need to DEBUG this -- gives error that rep_indices need to be indices, but is list???

        rep_labels = self.limit_labels[rep_indices]
        mapping = cluster_module.get_mapping(rep_indices, all_cluster_labels)


        evaluate_with_gt5(self.limit_labels, rep_labels, mapping)



class JVCRunner_v1:

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.compressor = Compressor()
        self.decompressor = Decompressor()
        self.video_loader = SeattleLoader()


    def encode(self, path_to_video):
        video_filename = os.path.basename(path_to_video)
        ###TODO: eliminate the extension
        video_filename = video_filename.split('.')[0]
        images, metadata = self.video_loader.load_images(path_to_video) ## we might need metadata such as fps, frame_width, frame_height, fourcc from here
        rep_indices = self.preprocessor.run(images, video_filename)
        self.compressor.run(images, rep_indices, metadata)


    def decode(self, path_to_video, number_of_samples = None):
        images = self.decompressor.run(path_to_video, number_of_samples)

        return images


"""
if __name__ == "__main__":
    timer = Timer() ##TODO: use the timer to run the pipeline
    preprocessor = Preprocessor()
    compressor = Compressor()
    decompressor = Decompressor()

    video_loader = SeattleLoader()
    images = video_loader.load_images()
    meta_data = preprocessor.run(images)
    save_directory = compressor.run(images, meta_data)

    number_of_frames = 100 ## we can change this to whatever number we want
    images_jvc = decompressor.run(save_directory, number_of_frames)
"""

if __name__ == "__main__":
    print("Starting POC, comparing vs middle frame encounter method against gt")
    runner = JVCRunner_POC()
