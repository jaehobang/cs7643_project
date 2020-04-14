import sys

home_dir = '/home/jbang36/eva_jaeho'
sys.path.append(home_dir)

## Need to load the modules
from loaders.uadetrac_loader import UADetracLoader
from logger import Logger

import numpy as np
import itertools
import time
import torch
import eva_storage.baselines.indexing.external.ssd.custom_code.util_ssd_uad as util_custom
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


from eva_storage.UNet import UNet
from eva_storage.baselines.indexing.external.ssd.vision.utils.misc import freeze_net_layers
from eva_storage.baselines.indexing.external.ssd.vision.ssd.ssd import MatchPriorModified
from eva_storage.baselines.indexing.external.ssd.vision.ssd.vgg_ssd import create_vgg_ssd
from eva_storage.baselines.indexing.external.ssd.vision.nn.multibox_loss import MultiboxLoss
from eva_storage.baselines.indexing.external.ssd.vision.ssd.config import vgg_ssd_config
from eva_storage.baselines.indexing.external.ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

#from eva_storage.baselines.indexing.external.ssd.custom_code.util_ssd_uad import UADataset_lite, filter_input, overlap


## Because we need to do debugging mode, we will put everything on the cpu for now
# DEVICE = torch.device('cpu')


def train(loader, net, criterion, optimizer, device, logging, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0



def normalize_boxes(all_boxes, image_width, image_height):
    """
    We normalize the box parameters -- we also have to convert to center coordinate format, but this the network will do for us
    :param all_boxes:
    :param image_width:
    :param image_height:
    :return:
    """
    new_boxes = []
    for boxes_per_frame in all_boxes:
        new_boxes_per_frame = []
        for i, box in enumerate(boxes_per_frame):
            left, top, right, bottom = box
            new_boxes_per_frame.append(
                (left / image_width, top / image_height, right / image_width, bottom / image_height))
        new_boxes.append(new_boxes_per_frame)

    assert (len(new_boxes) == len(all_boxes))
    for i, boxes_per_frame in enumerate(all_boxes):
        assert (len(boxes_per_frame) == len(new_boxes[i]))

    return new_boxes




def load_jnet_results(images, model_directory, segmented_images_directory = None):
    """
    we need to load the segmented images instead of the original ones
    :return:
    """

    ## need to check if segmented images exist
    if segmented_images_directory is None:
        ## load the model
        network = UNet()

        ## execute the model on the images of interest
        _, level4_outputs = network.execute(images, load_dir=model_directory)

        ## we should overlap the images
        level4_overlapped_outputs = util_custom.overlap(images, level4_outputs)
        return level4_overlapped_outputs

    else:
        return np.load(segmented_images_directory)



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" ## we want to run everything on gpu 1
    DEVICE = train_device = torch.device('cuda') ## will this execute everything on gpu 1?

    loader = UADetracLoader()
    logger = Logger()
    images_train = loader.load_cached_images()
    labels_train = loader.load_cached_labels()
    boxes_train = loader.load_cached_boxes()

    ### we need to take out frames that don't have labels / boxes
    labels_train = labels_train['vehicle']
    images_train, labels_train, boxes_train = util_custom.filter_input(images_train, labels_train, boxes_train)
    image_width, image_height = images_train.shape[1], images_train.shape[2]


    images_train = images_train.astype(np.uint8)
    ## before we divide the dataset, we want to load the segmented images..or process them
    segmented_images_directory = '/nethome/jbang36/eva_jaeho/data/npy_files/jnet_output_train_images.npy'
    images_train = load_jnet_results(images_train,
                                     model_directory= '/nethome/jbang36/eva_jaeho/data/models/bloating_only/history20_dist2thresh300_bloat_lvl4-epoch60.pth',
                                     segmented_images_directory = segmented_images_directory)

    if segmented_images_directory is None:
        ## we should try saving the results for next time

        np.save('/nethome/jbang36/eva_jaeho/data/npy_files/jnet_output_train_images.npy', images_train)



    ## we need to normalize the boxes

    logger.info("Finished loading the UADetrac Dataset")

    ## we need to create a UA Dataset_lite..
    ## divide the set into train and validation
    val_division = int(0.8 * len(images_train))
    images_val = images_train[val_division:]
    images_train = images_train[:val_division]

    labels_val = labels_train[val_division:]
    labels_train = labels_train[:val_division]

    ## uadetrac_lite does that for us

    boxes_val = boxes_train[val_division:]
    boxes_train = boxes_train[:val_division]
    logger.info("Finished diving the loaded dataset into train and val")

    ## initialize the training variables
    base_net = os.path.join(home_dir, 'eva_storage/baselines/indexing/external/ssd/', "models/vgg16_reducedfc.pth")
    batch_size = 24
    num_workers = 4
    num_epochs = 100
    checkpoint_folder = os.path.join(home_dir, 'eva_storage/baselines/indexing/external/ssd', 'models/')
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    validation_epochs = 5
    debug_steps = 100

    ## create the net and transforms
    create_net = create_vgg_ssd
    config = vgg_ssd_config

    train_transform = TrainAugmentation(config.image_size, config.image_mean,
                                        config.image_std)

    ## Very important, we train on the boxes instead of the locations because locations -> boxes is near impossible
    target_transform = MatchPriorModified(config.priors, config.center_variance,
                                          config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean,
                                   config.image_std)

    ## Load the model
    num_classes = 5  ## is this right?? there is the background.....
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1
    base_net_lr = lr
    extra_layers_lr = lr

    print("Base net is frozen..")
    freeze_net_layers(net.base_net)
    params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                             net.regression_headers.parameters(), net.classification_headers.parameters())
    params = [
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]

    net.init_from_base_net(base_net)
    """
    net.init_from_pretrained_ssd(args.pretrained_ssd)
    """

    net.to(DEVICE)
    logger.info("Finished loading the dataset")

    ## Load the rest (optimizer, loss function, etc)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9,
                                weight_decay=5e-4)

    milestones = [int(v.strip()) for v in "80,100".split(",")]
    scheduler = MultiStepLR(optimizer, milestones=milestones,
                            gamma=0.1, last_epoch=last_epoch)

    logger.info("Finished creating external modules")

    dataset_train = util_custom.UADataset_lite(transform=train_transform, target_transform=target_transform)
    dataset_train.set_images(images_train)
    dataset_train.set_labels(labels_train)
    dataset_train.set_boxes(boxes_train)

    dataset_val = util_custom.UADataset_lite(transform=train_transform, target_transform=target_transform)
    dataset_val.set_images(images_val)
    dataset_val.set_labels(labels_val)
    dataset_val.set_boxes(boxes_val)

    train_loader = DataLoader(dataset_train, batch_size,
                              num_workers=4,
                              shuffle=True)

    val_loader = DataLoader(dataset_val, batch_size,
                            num_workers=4,
                            shuffle=True)

    logger.info("Finished creating data loader for model")

    st = time.perf_counter()

    for epoch in range(last_epoch + 1, num_epochs):
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, logging=logger, debug_steps=debug_steps, epoch=epoch)
        scheduler.step()

        if epoch % validation_epochs == 0 or epoch == num_epochs - 1:
            net.eval()
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            num = 0
            for _, data in enumerate(val_loader):
                images, boxes, labels = data
                images = images.to(DEVICE)
                boxes = boxes.to(DEVICE)
                labels = labels.to(DEVICE)
                num += 1
                with torch.no_grad():
                    confidence, proposed_boxes = net(images)
                    regression_loss, classification_loss = criterion(confidence, proposed_boxes, labels, boxes)
                    loss = regression_loss + classification_loss
                running_loss += loss.item()
                running_regression_loss += regression_loss.item()
                running_classification_loss += classification_loss.item()
            val_loss = running_loss / num
            val_regression_loss = running_regression_loss / num
            val_classification_loss = running_regression_loss / num
            print("epoch", epoch)
            print("  Validation Loss: {v:.4f}".format(v=val_loss))
            print("  Validatiion Regression Loss: {v:.4f}".format(v=val_regression_loss))
            print("  Validation Classification Loss: {v:.4f}".format(v=val_classification_loss))

            checkpoint_file_name = "vgg16-ssd:epoch" + str(epoch)
            model_path = os.path.join(checkpoint_folder, checkpoint_file_name)
            net.save(model_path)

    print("Total time to train...", time.perf_counter() - st, "seconds")








