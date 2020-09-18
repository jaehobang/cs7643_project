

"""
In this file, we implement a wrapper around the whole process
"""


from eva_storage.jvc.encoder import Encoder
from eva_storage.jvc.decoder import Decoder
from eva_storage.jvc.preprocessor import Preprocessor
from loaders.seattle_loader import SeattleLoader



import os


"""
Notes:
Preprocessor: self.hierarchy_save_dir = os.path.join('/nethome/jbang36/eva_jaeho/data/frame_hierarchy', video_type,
                                               video_name + '.npy')
                                               
                                               
Decoder: self.video_base_path = '/nethome/jbang36/eva_jaeho/data/'
        self.hierarchy_base_path = '/nethome/jbang36/eva_jaeho/data/frame_hierarchy'
        
                                             


"""

class JVC:


    def __init__(self, loader = None):
        self.preprocessor = Preprocessor()
        ### TODO: we have to keep modifying the video_type, video_name variables.... or we can just manage all that here??
        self.encoder = Encoder()
        self.decoder = Decoder() ## if user doesn't supply a loader, we load the default loader
        self.base_directory = '/nethome/jbang36/eva_jaeho/data'
        self.images = None
        self.directories = {}
        if loader is None:
            self.loader = SeattleLoader()

    def preprocess_default(self, images, video_type, video_name, **kwargs):
        """
        Function used when images are already given
        :param images:
        :return:
        """
        hierarchy_save_dir = os.path.join(self.base_directory, 'frame_hierarchy', video_type, video_name + '.npy')
        proposed_cluster_count = len(images) // 100 if len(images) // 100 > 0 else len(images)
        cluster_count = kwargs.get('cluster_count', proposed_cluster_count)
        stopping_point = kwargs.get('stopping_point', proposed_cluster_count)
        self.hierarchy = self.preprocessor.run_final(images, hierarchy_save_dir, cluster_count=cluster_count,
                                                     stopping_point=stopping_point)
        hierarchy = self.hierarchy
        self.directories['hierarchy'] = hierarchy_save_dir

        return sorted(hierarchy[:cluster_count])

    def preprocess(self, video_type, video_name, **kwargs):
        extension = kwargs.get('extension', '.mp4')
        ### this is just the name of the video
        self.original_video_directory = os.path.join(self.base_directory,  video_type, video_name + extension)
        video_directory = self.original_video_directory
        hierarchy_save_dir = os.path.join(self.base_directory, 'frame_hierarchy', video_type, video_name + '.npy')

        self.images = self.loader.load_images(video_directory)
        images = self.images
        proposed_cluster_count = len(images) // 100 if len(images) // 100 > 0 else len(images)
        cluster_count = kwargs.get('cluster_count', proposed_cluster_count)
        stopping_point = kwargs.get('stopping_point', proposed_cluster_count)
        self.hierarchy = self.preprocessor.run_final(images, hierarchy_save_dir, cluster_count = cluster_count, stopping_point = stopping_point)
        hierarchy = self.hierarchy


        ##update the directories
        self.directories['hierarchy'] = hierarchy_save_dir
        self.directories['video_dir'] = video_directory

        return sorted(hierarchy[:cluster_count]) ## we want to sort the examples chosen for evaluation


    def decode(self, video_type, jvc_video_name, hierarchy_name, **kwargs):
        sample_count = kwargs.get('sample_count', 100) ## TODO: make sure the decoder takes care of edge cases
        video_directory = os.path.join( self.base_directory, video_type, jvc_video_name + '.mp4')
        hierarchy_directory = os.path.join( self.base_directory, 'frame_hierarchy', video_type, hierarchy_name + '.npy')
        iframe_indices_directory = os.path.join( self.base_directory, 'iframe_indices', video_type, jvc_video_name + '.npy')
        extracted_images = self.decoder.run(video_directory, hierarchy_directory, iframe_indices_directory, number_of_samples = sample_count)


        return extracted_images


    def encode(self, video_type, jvc_video_name, **kwargs):
        save_directory = os.path.join( self.base_directory, video_type, jvc_video_name + '.mp4')
        iframe_indices_save_directory = os.path.join( self.base_directory, 'iframe_indices', video_type, jvc_video_name + '.npy')
        self.encoder.run(self.images, self.hierarchy, self.original_video_directory, save_directory, iframe_indices_save_directory)
        self.jvc_video_directory = save_directory

        self.directories['jvc_video_dir'] = self.jvc_video_directory
        self.directories['iframe_indices_dir'] = iframe_indices_save_directory
        return




if __name__ == "__main__":
    jvc = JVC()
    jvc.preprocess()
    jvc.encode()
    jvc.decode()
