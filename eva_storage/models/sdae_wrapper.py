"""
In this file, I will implement a wrapper for the Stacked Denoising Autoencoder Implemented by https://github.com/vlukiyanov/pt-sdae

"""

import torch
import torch.nn as nn
from others.ptdec_eva.ptsdae.ptsdae.sdae import StackedDenoisingAutoEncoder
from eva_storage.models.network_models import NetworkTemplate
import others.ptdec_eva.ptsdae.ptsdae.model as ae
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import numpy as np
from eva_storage.models import JNETTransform, JNETDataset, NetworkUtils
import config
from torch.utils.data import Dataset
from torchvision import transforms
import cv2



class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = cv2.resize(img, self._size)
        return resize_image

class GrayScale:

    def __call__(self, img:np.ndarray):
        assert(img.ndim == 3)
        mean_im = np.mean(img, axis = 2)
        mean_im = mean_im.astype(np.uint8)
        #mean_im = img
        return mean_im



class CachedUAD(Dataset):
    def __init__(self, input_size):
        assert(type(input_size) is tuple or type(input_size) is list)

        self.img_transform = transforms.Compose([
            Resize(input_size),
            GrayScale(),
            transforms.Lambda(self._transformation)
        ])

        self._cache = dict()

    def set_images(self, images):
        self.X_train = images
        self.image_width = images.shape[1]
        self.image_height = images.shape[2]

    def set_labels(self, labels):
        self.y_train = labels


    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """

        img = self.X_train[index]
        labels = self.y_train[index]

        img = self.img_transform(img)

        return img, labels

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt = self.pull_item(index)
        ### first of all, this is bogus because
        return im

    @staticmethod
    def _transformation(img):
        #print(img.shape)
        return torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float() * 0.02





class SDAE_wrapper(NetworkTemplate):
    def __init__(self):
        super().__init__()
        #super(NetworkTemplate, self).__init__()
        self.input_width = 100
        self.input_height = 100
        self.compressed_size = 100
        self.model = StackedDenoisingAutoEncoder([self.input_width * self.input_height,
                                                  500, 500, 2000, self.compressed_size],
                                                final_activation=None)




    def train(self, train_images, target_images, save_name, total_epochs, cuda = True, lr = 0.0001, weight_decay = 1e-6, batch_size = 256):

        train_dataset = CachedUAD((self.input_width, self.input_height))
        train_dataset.set_images(train_images)
        fake_labels = np.zeros(len(train_images))
        train_dataset.set_labels(fake_labels)

        self.dataset = train_dataset

        if cuda:
            self.model.cuda()

        pretrain_epochs = total_epochs // 8 * 3
        batch_size = 256

        print('Pretraining stage.')
        ae.pretrain(
            self.dataset,
            self.model,
            cuda=cuda,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
            scheduler=lambda x: StepLR(x, 100, gamma=0.1),
            corruption=0.2, epoch_callback=None
        )

        print('Training stage.')
        finetune_epochs = total_epochs // 8 * 5
        ae_optimizer = SGD(params=self.model.parameters(), lr=0.1, momentum=0.9)
        ae.train(
            self.dataset,
            self.model,
            cuda=cuda,
            epochs=finetune_epochs,
            batch_size=batch_size,
            optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
            corruption=0.2
        )
        self.save(save_name, total_epochs)
        return



    def execute(self, test_images, image_shape):

        batch_size = 256
        test_dataset = CachedUAD()
        test_dataset.set_images(test_images)
        fake_labels = np.zeros(len(test_images))
        test_dataset.set_labels(fake_labels)
        dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=4, pin_memory=True)

        seg_data = np.ndarray(shape=(image_shape[0], image_shape[1], image_shape[2]), dtype=np.uint8)
        compressed_data = np.ndarray(shape=(image_shape[0], self.compressed_size), dtype=np.uint8)
        self.logger.debug(f"Seg data projected shape {seg_data.shape}")
        self.logger.debug(f"Compressed data projected shape {compressed_data.shape}")

        self.model.eval()

        for i, data_pack in enumerate(dataset_loader):
            images_input, _ = data_pack
            images_input = images_input.to(config.eval_device)
            ## TODO: need to make some fixes here, do we need to supply a 4D input?
            compressed, final = self.model(images_input)
            final_cpu = NetworkUtils.convertSegmented(final)  ## TODO: GPU -> CPU
            compressed_cpu = NetworkUtils.convertCompressed(compressed)
            seg_data[i * batch_size:(i + 1) * batch_size] = final_cpu
            compressed_data[i * batch_size:(i + 1) * batch_size] = compressed_cpu


        self.logger.info(f"Processed {len(images)} in {time.perf_counter() - st} (sec)")
        assert (compressed_data.dtype == np.uint8)
        assert (seg_data.dtype == np.uint8)
        return compressed_data, seg_data
