"""
Defines the model used for generating index and compression

@Jaeho Bang
"""

import time
import numpy as np
from eva_storage.models.sdae_wrapper import SDAE_wrapper
from logger import Logger, LoggingLevel

import torch
import torch.utils.data
import argparse


parser = argparse.ArgumentParser(description='Define arguments for loader')
parser.add_argument('--learning_rate', type=int, default=0.0001, help='Learning rate for UNet')
parser.add_argument('--total_epochs', type=int, default=800, help='Number of epoch for training')
parser.add_argument('--l2_reg', type=int, default=1e-6, help='Regularization constaint for training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size used for training')
parser.add_argument('--compressed_size', type=int, default=100,
                    help='Number of features the compressed image format has')
parser.add_argument('--checkpoint_name', type=str, default='unet_uadetrac',
                    help='name of the file that will be used to save checkpoints')
args = parser.parse_args()




### TODO: Need to fix the train / execute methods!!!


class Network:

    def __init__(self, type=0):
        self.model = None
        self.dataset = None
        self.data_dimensions = None
        self.logger = Logger()
        self.network_type = type  ## for now, type will denoting whether we are using a compressed or full network

    def debugMode(self, mode=False):
        if mode:
            self.logger.setLogLevel(LoggingLevel.DEBUG)
        else:
            self.logger.setLogLevel(LoggingLevel.INFO)


    def _parse_dir(self, directory_string):
        """
        This function is called by other methods in UNET to parse the directory string to extract model name and epoch
        We will assume the format of the string is /dir/name/whatever/{model_name}-{epoch}.pth
        :param directory_string: string of interest
        :return:
        """

        tmp = directory_string.split('/')
        tmp = tmp[-1]
        model_name, epoch_w_pth = tmp.split('-')
        epoch = int(epoch_w_pth.split('.')[0])
        assert (type(epoch) == int)
        assert (type(model_name) == str)
        return model_name, epoch



    def train(self, train_images: np.ndarray, target_images: np.ndarray, save_name, load_dir = None, model_type = 'sdae', cuda=True):
        """
        Trains the network with given images
        :param images: original images
        :param segmented_images: tmp_data
        :return: None
        """
        if cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        ### all types of dataset here
        if model_type == 'sdae':
            self.model = SDAE_wrapper()

        assert(self.model is not None)

        ### we could be training the network from a starting point
        if load_dir is not None:
            self.model.load(load_dir)

        ## delegate the train step to the specific model
        st = time.perf_counter()

        ## give the model the training dataset
        self.model.train(train_images, target_images, save_name, args.total_epochs, lr = args.learning_rate, batch_size = args.batch_size, weight_decay = args.l2_reg, cuda = cuda)
        self.logger.info(f"Trained the model {self.model} in {time.perf_counter() - st} (sec)")

        return



    def execute(self, test_images: np.ndarray = None, model_type = 'dae', load_dir=None, cuda = True):
        """
        We will overload this function to take in no parameters when we are just executing on the given image..
        :return: compressed, segmented images that are output of the network
        """
        st = time.perf_counter()
        if self.model is None:
            if model_type == 'dae':
                self.model = SDAE_wrapper()
            else:
                raise ValueError
            if load_dir is None:
                raise ValueError
            else:
                self.model.load(load_dir, execute = True)


        assert (self.model is not None)

        self.logger.info("Images are given, creating dataset object and executing...    ")

        return self.model.execute(test_images, cuda)





if __name__ == "__main__":
    network = Network()

    from loaders.uadetrac_loader import UADetracLoader
    loader = UADetracLoader()
    images = loader.load_images()


