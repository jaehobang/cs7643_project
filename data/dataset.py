"""
In this file we define a general datset used for evaluation
"""

"""UAD Dataset Classes

Original author: Jaeho Bang


"""
import os
import torch
import torch.utils.data as data

from logger import Logger

HOME = os.path.expanduser("~")


class EvaluationDataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        transform (callable, optional): A function/transform that augments the
                                        raw images
    """

    def __init__(self, images, transform=None):

        self.transform = transform

        self.images = images
        self.logger = Logger()


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


    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """

        img = self.images[index]
        height, width, _ = img.shape

        if self.transform is not None:
            ## we expect this code to run only if target_trans
            img, _, _ = self.transform(img)
            # to rgb -- we have to do this bc cv2 loads BRG instead of RGB
            img = img[:, :, (2, 1, 0)]

        return torch.from_numpy(img).permute(2, 0, 1), None, height, width

    def __len__(self):
        return len(self.X_train)



class TrainDataset(data.Dataset):
    ### Here we define a general purpose dataset used for training, hence we need to have transform / target_transform
    pass










