"""
Defines the model used for generating index and compression

@Jaeho Bang
"""

import time
import os
import numpy as np
from eva_storage.models.UNet_final import UNet_final
from eva_storage.models.UNet_compressed import UNet_compressed
from logger import Logger, LoggingLevel


import torch
import torch.utils.data
import torch.nn as nn
import argparse
import config
import cv2
import torch.utils.data as data


parser = argparse.ArgumentParser(description='Define arguments for loader')
parser.add_argument('--learning_rate', type = int, default=0.0001, help='Learning rate for UNet')
parser.add_argument('--total_epochs', type = int, default=60, help='Number of epoch for training')
parser.add_argument('--l2_reg', type = int, default=1e-6, help='Regularization constaint for training')
parser.add_argument('--batch_size', type = int, default = 64, help='Batch size used for training')
parser.add_argument('--compressed_size', type = int, default = 100, help='Number of features the compressed image format has')
parser.add_argument('--checkpoint_name', type = str, default = 'unet_uadetrac', help='name of the file that will be used to save checkpoints')
args = parser.parse_args()





class UnetTransform:
    """
        BaseTransofrm will perform the following operations:
        1. Avg -- convert the image to grayscale
        2. Normalize -- arrange all pixel values to be between 0 and 1
        3. Resize -- resize the images to fit the network specifications
    """
    def __init__(self, size):
        self.size = size

    def transform(self, image, mean, std):
        size = self.size
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x -= mean
        x /= std
        x = x.astype(np.float32)
        y = np.copy(x)
        y = np.mean(y, axis=2)

        ## we need a transform for the output as well
        ## make the base_transform return 2 different objects

        return x, y

    def __call__(self, image, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype = np.float32)
        return self.transform(image, self.mean, self.std)




class UADUNet(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, transform=None, dataset_name='UA-DETRAC', eval = False):

        self.transform = transform
        self.name = dataset_name

        self.root = ""
        self.image_width = -1
        self.image_height = -1
        self.logger = Logger()
        self.eval = eval
        self.X_train = None
        self.y_train = None


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        image_input, image_output = self.pull_item(index)
        return image_input, image_output


    def set_images(self, images):
        self.X_train = images
        self.image_width = self.X_train.shape[1]
        self.image_height = self.X_train.shape[2]
        self.mean = np.mean(self.X_train, axis = (0,1,2))
        self.std = np.std(self.X_train, axis = (0,1,2))


    def set_target_images(self, target_images):
        self.y_train = target_images

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img = self.X_train[index]
        if self.transform is not None:
            ## we expect this code to run only if target_trans
            img, target = self.transform(img, self.mean, self.std)

        if self.y_train is not None:
            target = self.y_train[index]

        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(target)


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        return self.X_train[index] ## note this images is BRG

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __len__(self):
        return len(self.X_train)

### TODO: Need to fix the train / execute methods!!!


class UNet:

    def __init__(self, type = 0):
        self.model = None
        self.dataset = None
        self.data_dimensions = None
        self.logger = Logger()
        self.network_type = type ## for now, type will denoting whether we are using a compressed or full network



    def debugMode(self, mode = False):
        if mode:
            self.logger.setLogLevel(LoggingLevel.DEBUG)
        else:
            self.logger.setLogLevel(LoggingLevel.INFO)

    """
    def createDataExecute(self, images:np.ndarray, batch_size = args.batch_size):
        assert(images.dtype == np.uint8)
        images = images.astype(np.float32)
        images /= 255.0
        images = np.transpose(images, (0, 3, 1, 2))

        return torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4)


    def createData(self, images:np.ndarray, segmented_images:np.ndarray, batch_size = args.batch_size):
        

        # we assume the data is not normalized...
        assert(images.dtype == np.uint8)
        assert(segmented_images.dtype == np.uint8)

        images_normalized = images.astype(np.float32)
        images_normalized = (images_normalized / 255.0)
        segmented_normalized = segmented_images.astype(np.float32)
        segmented_normalized = (segmented_normalized / 255.0)
        segmented_normalized = np.expand_dims(segmented_normalized, axis = 3)

        data = np.concatenate((images_normalized, segmented_normalized), axis = 3)
        data = np.transpose(data, (0,3,1,2))

        return torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = False, num_workers = 4)
    """

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
        assert(type(epoch) == int)
        assert(type(model_name) == str)
        return model_name, epoch


    def createDataset(self, images, target_images = None):

        dataset = UADUNet(transform=UnetTransform(300))
        dataset.set_images(images)
        dataset.set_target_images(target_images)
        return dataset



    def train(self, images:np.ndarray, segmented_images:np.ndarray, save_name, load_dir = None, cuda = True):
        """
        Trains the network with given images
        :param images: original images
        :param segmented_images: tmp_data
        :return: None
        """
        if cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')


        ## Note, now segmented_images can be None

        self.dataset = self.createDataset(images, segmented_images)
        dataset_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        epoch = 0

        if load_dir is not None:
            self.logger.info(f"Loading from {load_dir}")
            self._load(load_dir)
            model_name, epoch = self._parse_dir(load_dir)

        if self.model is None:
            ## load_dir might not have been specified, or load_dir is incorrect
            self.logger.info(f"New model instance created on device {config.train_device}")
            if self.network_type == 0:
                self.model = UNet_final(args.compressed_size).to(device = config.train_device, dtype = None, non_blocking = False)
            elif self.network_type == 1:
                self.model = UNet_compressed(args.compressed_size).to(device = config.train_device, dtype = None, non_blocking = False)


        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        st = time.perf_counter()


        self.logger.info("Training the network....")
        for ep in range(epoch, args.total_epochs):
            for i, data_pack in enumerate(dataset_loader):
                images_input, images_output = data_pack
                images_input = images_input.to(config.train_device)
                images_output = images_output.to(config.train_device)

                compressed, final = self.model(images_input)

                optimizer.zero_grad()
                loss = distance(final, images_output)
                loss.backward()
                optimizer.step()


            self.logger.info('epoch [{}/{}], loss:{:.4f}, time elapsed:{:.4f} (sec)'.format(ep, args.total_epochs,
                                                                                       loss.data,
                                                                                       time.perf_counter() - st))
            st = time.perf_counter()


        self._save(save_name, args.total_epochs)
        self.logger.info(f"Finished training the network and save as {save_name+'-epoch'+str(args.total_epochs)+'.pth'}")
        return None


    def _save(self, save_name, epoch = 0):
        """
        Save the model
        We will save this in the
        :return: None
        """
        eva_dir = config.eva_dir
        dir = os.path.join(eva_dir, 'data', 'models', '{}-epoch{}.pth'.format(save_name, epoch))
        print("Saving the trained model as....", dir)

        torch.save(self.model.state_dict(), dir)


    def _load(self, load_dir, execute = False):
        """
        Load the model

        :return:
        """

        if os.path.exists(load_dir): ## os.path.exists works on folders and files

            if execute:
                if self.network_type == 0:
                    self.model = UNet_final(args.compressed_size).to(config.eval_device, dtype=None, non_blocking=False)
                elif self.network_type == 1:
                    self.model = UNet_compressed(args.compressed_size).to(config.eval_device, dtype=None,
                                                                     non_blocking=False)
            else:
                if self.network_type == 0:
                    self.model = UNet_final(args.compressed_size).to(config.eval_device, dtype=None, non_blocking=False)
                if self.network_type == 0:
                    self.model = UNet_final(args.compressed_size).to(config.eval_device, dtype=None, non_blocking=False)
                self.model = UNet_final(args.compressed_size).to(config.train_device, dtype=None, non_blocking=False)

            self.model.load_state_dict(torch.load(load_dir))
            self.logger.info("Model load success!")

        else:
            self.logger.error("Checkpoint does not exist returning")


    def execute(self, images:np.ndarray = None, load_dir = None):
        """
        We will overload this function to take in no parameters when we are just executing on the given image..
        :return: compressed, segmented images that are output of the network
        """
        st = time.perf_counter()
        if load_dir is not None:
            self.logger.info(f"Loading from {load_dir}")
            self._load(load_dir, execute=True)

        if self.model is None:
            self.logger.error("There is no model and loading directory is not supplied. Value Error will be raised")
            raise ValueError

        assert(self.model is not None)
        #self.logger.debug(f"Model on gpu device {self.model.get_device()}, running execution on gpu device {config.eval_device}")
        seg_data = np.ndarray(shape=(images.shape[0], images.shape[1], images.shape[2]), dtype=np.uint8)
        compressed_data = np.ndarray(shape=(images.shape[0], args.compressed_size), dtype=np.uint8)
        self.logger.debug(f"Seg data projected shape {seg_data.shape}")
        self.logger.debug(f"Compressed data projected shape {compressed_data.shape}")

        if images is None:
            self.logger.info("Images are not given, assuming we already have dataset object...")

            for i, data_pack in enumerate(self.dataset):
                images_input, _ = data_pack
                images_input = images_input.to(config.eval_device)
                compressed, final = self.model(images_input)
                final_cpu = self._convertSegmented(final)
                compressed_cpu = self._convertCompressed(compressed)
                seg_data[i * args.batch_size:(i + 1) * args.batch_size] = final_cpu
                compressed_data[i*args.batch_size:(i + 1) * args.batch_size] = compressed_cpu
        else:
            self.logger.info("Images are given, creating dataset object and executing...    ")
            dataset = self.createDataset(images)
            dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                         num_workers=4, pin_memory=True)

            for i, data_pack in enumerate(dataset_loader):
                images_input, _ = data_pack
                images_input = images_input.to(config.eval_device)
                ## TODO: need to make some fixes here, do we need to supply a 4D input?
                compressed, final = self.model(images_input)
                final_cpu = self._convertSegmented(final) ## TODO: GPU -> CPU
                compressed_cpu = self._convertCompressed(compressed)
                seg_data[i * args.batch_size:(i + 1) * args.batch_size] = final_cpu
                compressed_data[i * args.batch_size:(i + 1) * args.batch_size] = compressed_cpu

        self.logger.info(f"Processed {len(images)} in {time.perf_counter() - st} (sec)")
        assert(compressed_data.dtype == np.uint8)
        assert(seg_data.dtype == np.uint8)
        return compressed_data, seg_data


    def _convertCompressed(self, compressed_image):
        """
        Converts the compressed images from the network to something that can be saved in an np.array
        :param compressed_image: Compressed image from the network
        :return: np compressed image
        """
        compressed_cpu = compressed_image.cpu()
        compressed_cpu = compressed_cpu.detach().numpy()
        compressed_cpu *= 255
        compressed_cpu = compressed_cpu.astype(np.uint8)
        return compressed_cpu



    def _convertSegmented(self, segmented_image):
        """
        Converts the segmented images from the network to something that can be saved in an nparray
        :param segmented_image: Segmented image output from the network
        :return: np segmented image
        """
        segmented_image = segmented_image.cpu()
        recon_p = segmented_image.permute(0, 2, 3, 1)
        recon_imgs = recon_p.detach().numpy()
        recon_imgs *= 255
        recon_imgs = recon_imgs.astype(np.uint8)
        recon_imgs = recon_imgs.squeeze()
        return recon_imgs





