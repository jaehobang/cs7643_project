import numpy as np
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import uuid

from others.ptdec_eva.ptdec.dec import DEC
from others.ptdec_eva.ptdec.model import train, predict
from others.ptdec_eva.ptsdae.ptsdae.sdae import StackedDenoisingAutoEncoder
import others.ptdec_eva.ptsdae.ptsdae.model as ae
from others.ptdec_eva.ptdec.utils import cluster_accuracy


class CachedMNIST(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        img_transform = transforms.Compose([
            transforms.Lambda(self._transformation)
        ])
        self.ds = MNIST(
            './data',
            download=True,
            train=train,
            transform=img_transform
        )
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float() * 0.02

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            ##TODO: There is a bug here!! the Code runs well if we don't do it on CUDA
            ## Need to move the data to cuda,
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                #self._cache[index][1] = self._cache[index][1].cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)






def main(
    cuda,
    batch_size,
    pretrain_epochs,
    finetune_epochs,
    testing_mode
):
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        pass

    ds_train = CachedMNIST(train=True, cuda=cuda, testing_mode=testing_mode)  # training dataset
    ds_val = CachedMNIST(train=False, cuda=cuda, testing_mode=testing_mode)  # evaluation dataset
    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10],
        final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print('Pretraining stage.')
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2, epoch_callback = None
    )
    print('Training stage.')
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback
    )
    print('DEC stage.')
    model = DEC(
        cluster_number=10,
        hidden_dimension=10,
        encoder=autoencoder.encoder
    )
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda
    )
    predicted, actual = predict(ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda)
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print('Final DEC accuracy: %s' % accuracy)


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    batch_size = 256
    cuda = True
    pretrain_epochs = 300
    finetune_epochs = 300
    testing_mode = False
    main(cuda,batch_size, pretrain_epochs, finetune_epochs, testing_mode)
