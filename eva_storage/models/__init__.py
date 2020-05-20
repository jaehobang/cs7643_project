import numpy as np
import cv2
from logger import Logger
from torch.utils import data
import torch


class JNETTransform:
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
        self.std = np.array(std, dtype=np.float32)
        return self.transform(image, self.mean, self.std)


class JNETDataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, transform=None, dataset_name='UA-DETRAC', eval=False):

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


    def get_image_shape(self):
        return self.X_train.shape

    def set_images(self, images):
        self.X_train = images
        self.image_width = self.X_train.shape[1]
        self.image_height = self.X_train.shape[2]
        self.mean = np.mean(self.X_train, axis=(0, 1, 2))
        self.std = np.std(self.X_train, axis=(0, 1, 2))

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
        return self.X_train[index]  ## note this images is BRG

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __len__(self):
        return len(self.X_train)



class NetworkUtils:

    @staticmethod
    def convertCompressed(compressed_image):
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


    @staticmethod
    def convertSegmented(segmented_image):
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
