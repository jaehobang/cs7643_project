from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision import transforms

from others.ptdec_eva.ptdec.dec import DEC
from others.ptdec_eva.ptdec.model import train, predict
from others.ptdec_eva.ptsdae.ptsdae.sdae import StackedDenoisingAutoEncoder

import others.ptdec_eva.ptsdae.ptsdae.model as ae
import numpy as np
import torch


from loaders.uadetrac_loader import UADetracLoader




class CachedUAD(Dataset):
    def __init__(self):

        self.img_transform = transforms.Compose([
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
        return torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float() * 0.02




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


    CLUSTER_NUM = -1 ## TODO: we need to set this number depending on size of test dataset


    loader = UADetracLoader()
    images = loader.load_cached_images(name = 'uad_test_images.npy', vi_name = 'uad_test_vi.npy')
    labels = loader.load_cached_labels(name = 'uad_test_labels.npy')
    labels = labels['vehicle']
    boxes = loader.load_cached_boxes(name = 'uad_test_boxes.npy')
    images, labels, boxes = loader.filter_input3(images, labels, boxes)

    ## we need to skip some frames, and use some frames that have been skipped for validation
    skip_rate = 15
    val_skip_rate = skip_rate * 3 + 1
    images_train = images[::skip_rate]
    labels_train = labels[::skip_rate]
    boxes_train = boxes[::skip_rate]

    images_test = images[::val_skip_rate] ## just a random number that makes the validation set small enough
    labels_test = labels[::val_skip_rate]
    boxes_test = boxes[::val_skip_rate]

    ## we need to resize the images.... try 28 x 28
    images_train = images_train[:,::11,::11,:]
    images_train = np.mean(images_train, axis = 3)
    images_train = images_train.astype(np.uint8)
    print(images_train.shape)
    assert(images_train.shape[1] == 28)
    assert(images_train.shape[2] == 28)

    images_test = images_test[:,::11,::11,:]
    images_test = np.mean(images_test, axis = 3)
    images_test = images_test.astype(np.uint8)
    assert(images_test.shape[1] == 28)
    assert(images_test.shape[2] == 28)

    train_dataset = CachedUAD()
    train_dataset.set_images(images_train)
    train_dataset.set_labels(labels_train)

    test_dataset = CachedUAD()
    test_dataset.set_images(images_test)
    test_dataset.set_labels(labels_test)


    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10],
        final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print('Pretraining stage.')
    ae.pretrain(
        train_dataset,
        autoencoder,
        cuda=cuda,
        validation=test_dataset,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2, epoch_callback = None
    )
    print('Training stage.')
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        train_dataset,
        autoencoder,
        cuda=cuda,
        validation=test_dataset,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback
    )
    print('DEC stage.')
    model = DEC(
        cluster_number=CLUSTER_NUM,
        hidden_dimension=10,
        encoder=autoencoder.encoder
    )
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=train_dataset,
        model=model,
        epochs=100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda
    )

    save_model('autoencoder', autoencoder)
    save_model('dec', model)


    predicted = predict(test_dataset, model, 1024, silent=True, return_actual=False, cuda=cuda)
    #actual = actual.cpu().numpy()
    #predicted = predicted.cpu().numpy()
    #reassignment, accuracy = cluster_accuracy(actual, predicted)
    #print('Final DEC accuracy: %s' % accuracy)



def save_model(model_name, model):

    save_dir = "/nethome/jbang36/eva_jaeho/others/ptdec_eva/models"
    dir_ = os.path.join(save_dir, model_name + ".pth")

    print("Saving the trained model as....", dir_)

    torch.save(model.state_dict(), dir_)

    return


def load_dec(model_name, cluster_num):
    save_dir = "/nethome/jbang36/eva_jaeho/others/ptdec_eva/models"
    dir_ = os.path.join(save_dir, model_name + ".pth")
    auto_dir = os.path.join(save_dir, 'autoencoder.pth')

    print(f"Loading {dir_}")

    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10],
        final_activation=None
    )

    autoencoder.load_state_dict(torch.load(auto_dir))



    model = DEC(cluster_number=cluster_num,
                     hidden_dimension=10,
                     encoder=autoencoder.encoder)

    model.load_state_dict(torch.load(dir_))

    return model



if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    batch_size = 256
    cuda = True
    pretrain_epochs = 300
    finetune_epochs = 300
    testing_mode = False
    main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode)

    """
    Now, it runs, so we need to start making the APIs for experiments
    
    ###TODOS:
    1. We need to save the model parameters
    2. We need to make an API for evaluation of DEC -- clustering the frames for sampling
    3. We need to evaluate using SSD
    """
