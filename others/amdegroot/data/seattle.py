"""UAD Dataset Classes

Original author: Jaeho Bang


"""
from .config import HOME
import os.path as osp
import torch
import torch.utils.data as data

from logger import Logger

SEATTLE_CLASSES = (  # always index 0
    'car', 'bus', 'others', 'van')

# note: if you used our download scripts, this should be right
SEATTLE_ROOT = osp.join(HOME, "eva_jaeho", "data", "seattle")


class SEATTLEDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, transform=None,
                 dataset_name='SEATTLE'):

        self.transform = transform
        self.name = dataset_name

        self.root = SEATTLE_ROOT
        self.image_width = -1
        self.image_height = -1
        self.logger = Logger()

        self.y_train = None
        self.X_train = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def set_images(self, images):
        self.X_train = images
        self.image_width = self.X_train.shape[1]
        self.image_height = self.X_train.shape[2]


    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """

        img = self.X_train[index]
        height, width, _ = img.shape

        if self.transform is not None:
            ## we expect this code to run only if target_trans
            img, _, _ = self.transform(img)
            # to rgb -- we have to do this bc cv2 loads BRG instead of RGB
            img = img[:, :, (2, 1, 0)]

        return torch.from_numpy(img).permute(2, 0, 1), None, height, width

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


    def __len__(self):
        return len(self.X_train)












