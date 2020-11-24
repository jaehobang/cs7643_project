from abc import ABC, abstractmethod
import numpy as np
import skimage.measure
import torch
import math
import cv2
import torchvision.models as models

from PIL import Image
import time
import os
import copy


class FeatureExtractionMethod(ABC):

    @abstractmethod
    def run(self, images, desired_vector_size):
        pass


class VGG16Method_train(FeatureExtractionMethod):

    def __init__(self):
        self.vgg16 = models.vgg16(pretrained = True)
        self.vgg16.cuda()
        self.curr_loss = -1

    def preprocess_data(self, images):
        pass

    def __str__(self):
        return "Extract the features using vgg16 network with training"

    def save(self, save_directory):
        save_file = os.path.join(save_directory, f'vgg16_loss_{self.curr_loss}')
        torch.save(self.vgg16.state_dict(), save_file)
        return

    def train(self, dataloaders):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.vgg16.features.parameters(), lr=0.0001)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_epochs = 10

        self.train_model(self.vgg16, dataloaders, criterion, optimizer, device=device, num_epochs=num_epochs)


    def train_model(self, model, dataloaders, criterion, optimizer, device='cpu', num_epochs=25, is_inception=False):
        ## TODO: update self.curr_loss -> this is used for
        since = time.time()
        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1000000

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.

                        ##outputs = model(inputs)
                        ## TODO: we don't use the classifier
                        outputs = model.features(inputs)
                        outputs = model.avgpool(outputs)

                        #print(f"output shape: {outputs.size()}, labels shape: {labels.size()}")
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                self.curr_loss = epoch_loss

                print('{} Loss: {:.4f} '.format(phase, epoch_loss))


                # deep copy the model
                if phase == 'val' and self.curr_loss < best_loss:
                    best_loss = self.curr_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(self.curr_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history




    def run(self, images, desired_vector_size):

        #import os
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## we want to run everything on gpu 1

        vgg16 = models.vgg16(pretrained = True)
        vgg16.cuda()
        images_torch = torch.tensor(images, device = 'cpu').float()
        images_torch = images_torch.permute(0,3,1,2)
        data_loader = torch.utils.data.DataLoader(images_torch, batch_size = 32, shuffle = False)
        outputs = []
        features = []
        for i, data_pack in enumerate(data_loader):
            image_batch = data_pack
            image_batch = image_batch.cuda()
            ## TODO: add in adjustable average pool
            output = vgg16.features(image_batch)
            output = vgg16.avgpool(output)
            output_sum = torch.sum(output, axis = 1)
            output_sum = output_sum.cpu()
            output_np = output_sum.detach().numpy() #n, 512, 9, 9

            outputs.append( output_np )
            features.append( output.cpu().detach().numpy() )


        images_reshaped = np.concatenate(outputs, axis = 0)
        self.image_features = np.concatenate(features, axis = 0)

        images_reshaped = images_reshaped.reshape(images_reshaped.shape[0], images_reshaped.shape[1] * images_reshaped.shape[2])

        if images_reshaped.shape[1] > desired_vector_size:
            return images_reshaped[:,:desired_vector_size]

        elif images_reshaped.shape[1] < desired_vector_size:
            tmp = np.ndarray(shape = (images_reshaped.shape[0], desired_vector_size - images_reshaped.shape[1]))
            images_reshaped = np.concatenate([images_reshaped, tmp], axis = 1)
            return images_reshaped

        else: ## images_reshaped.shape[1] == desired_vector_size
            return images_reshaped




class VGG16Method(FeatureExtractionMethod):

    def __str__(self):
        return "Extract the features using vgg16 network"

    def run(self, images, desired_vector_size):
        import torchvision.models as models
        #import os
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## we want to run everything on gpu 1

        vgg16 = models.vgg16(pretrained = True)
        vgg16.cuda()
        images_torch = torch.tensor(images, device = 'cpu').float()
        images_torch = images_torch.permute(0,3,1,2)
        data_loader = torch.utils.data.DataLoader(images_torch, batch_size = 32, shuffle = False)
        outputs = []
        for i, data_pack in enumerate(data_loader):
            image_batch = data_pack
            image_batch = image_batch.cuda()
            output = vgg16.features(image_batch)
            output = output.cpu()
            output_np = output.detach().numpy() #n, 512, 9, 9


            outputs.append( np.sum(output_np, axis = 1) )

        images_reshaped = np.concatenate(outputs, axis = 0)
        images_reshaped = images_reshaped.reshape(images_reshaped.shape[0], images_reshaped.shape[1] * images_reshaped.shape[2])

        if images_reshaped.shape[1] > desired_vector_size:
            return images_reshaped[:,:desired_vector_size]

        elif images_reshaped.shape[1] < desired_vector_size:
            tmp = np.ndarray(shape = (images_reshaped.shape[0], desired_vector_size - images_reshaped.shape[1]))
            images_reshaped = np.concatenate([images_reshaped, tmp], axis = 1)
            return images_reshaped

        else: ## images_reshaped.shape[1] == desired_vector_size
            return images_reshaped




class BackgroundSubtractionMethod(FeatureExtractionMethod):

    def __str__(self):
        return "Extract features after using background subtraction"

    def run(self, images, desired_vector_size, debug = False):
        import cv2
        # fgbg only takes grayscale images, we need to convert
        ## check if image is already converted to grayscale -> channels = 1

        segmented_images = np.ndarray(shape=(images.shape[0], images.shape[1], images.shape[2]), dtype=np.uint8)
        print(f"segmented_images shape will be {segmented_images.shape}")
        history = 40
        dist2Threshold = 300
        detectShadows = False
        fgbg = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2Threshold,
                                                 detectShadows=detectShadows)

        # first round is to tune the values of the background subtractor
        for ii in range(10):
            image_gray = np.mean(images[ii], axis = 2).astype(np.uint8)
            fgbg.apply(image_gray)


        for ii in range(len(images)):
            image_gray = np.mean(images[ii], axis = 2).astype(np.uint8)
            segmented_images[ii] = fgbg.apply(image_gray)

        ## after we generate the segmented_images, we need to downsample this somehow...
        ## first step, let's just downsample
        import math
        wanted_width = int(math.sqrt(desired_vector_size))
        wanted_height = desired_vector_size // wanted_width
        width_skip_rate = images.shape[1] // wanted_width
        height_skip_rate = images.shape[2] // wanted_height
        images_downscaled = segmented_images[:, ::width_skip_rate, ::height_skip_rate]
        print(f"segmented images shape {segmented_images.shape}")
        print(f"images_downsampled shape {images_downscaled.shape}")
        ## we need to cut off the last element
        if images_downscaled.shape[1] > wanted_height:
            images_downscaled = images_downscaled[:,:wanted_height,:]
        if images_downscaled.shape[2] > wanted_width:
            images_downscaled = images_downscaled[:,:,:wanted_width]
        assert(images_downscaled.shape == (images.shape[0], wanted_height, wanted_width))

        images_reshaped = images_downscaled.reshape(len(images), wanted_width * wanted_height)

        if debug:
            return images_reshaped, segmented_images

        return images_reshaped


class DownSampleLanczosMethod(FeatureExtractionMethod):

    def __init__(self):
        self.name = 'LANCZOS'

    def __str__(self):

        return "Downsample the image by choosing max value in pixel blocks"


    def run(self, images, desired_vector_size):
        assert (desired_vector_size % 10 == 0)
        wanted_width = 10
        wanted_height = desired_vector_size // wanted_width

        size = (wanted_height, wanted_width)
        new_images_arr = np.ndarray(shape = (len(images), wanted_width, wanted_height, images.shape[3]))
        for i in range(len(images)):
            im = images[i]
            pil_im = Image.fromarray(im, 'RGB')
            pil_im = pil_im.resize(size, Image.LANCZOS)
            new_images_arr[i] = np.array(pil_im, dtype = np.uint8)

        ## now we have to reshape this
        new_images_arr = np.mean(new_images_arr, axis = 3)
        images_reshaped = new_images_arr.reshape(len(images), wanted_height * wanted_width)

        return images_reshaped


class DownSampleMaxMethod(FeatureExtractionMethod):

    def __str__(self):
        return "Downsample the image by choosing max value in pixel blocks"

    def run(self, images, desired_vector_size):
        assert(desired_vector_size % 10 == 0)
        wanted_width = 10
        wanted_height = desired_vector_size // wanted_width
        height_box_size = images.shape[1] // wanted_height
        width_box_size = images.shape[2] // wanted_width

        images_reshaped = skimage.measure.block_reduce(images, (1, height_box_size, width_box_size, 3), np.max)
        images_reshaped = np.squeeze(images_reshaped, axis=3)
        images_reshaped = images_reshaped.reshape(len(images), wanted_height * wanted_width)

        return images_reshaped





class DownSampleMeanMethod2(FeatureExtractionMethod):

    def __str__(self):
        return "Downsample the image by avg pixel blocks using block_reduce from skimage"

    def run(self, images, desired_vector_size):
        assert(desired_vector_size % 10 == 0)
        wanted_width = int(math.sqrt(desired_vector_size))
        wanted_height = desired_vector_size // wanted_width
        height_box_size = images.shape[1] // wanted_height
        width_box_size = images.shape[2] // wanted_width

        images_reshaped = skimage.measure.block_reduce(images, (1, height_box_size, width_box_size, 3), np.mean)
        images_reshaped = np.squeeze(images_reshaped, axis = 3)

        ## we need to cut things off if they overflow
        if images_reshaped.shape[1] > wanted_height:
            images_reshaped = images_reshaped[:, :wanted_height, :]
        if images_reshaped.shape[2] > wanted_width:
            images_reshaped = images_reshaped[:, :, :wanted_width]

        images_reshaped = images_reshaped.reshape(len(images), wanted_height * wanted_width)

        return images_reshaped




class DownSampleMeanMethod(FeatureExtractionMethod):
    def __init__(self):
        self.name = 'MEAN'

    def __str__(self):
        return "Downsample the image by avg pixel blocks"

    def run(self, images, desired_vector_size):
        assert(desired_vector_size % 10 == 0)
        import math
        wanted_width = int(math.sqrt(desired_vector_size)) ## will modify this to
        wanted_height = desired_vector_size // wanted_width
        height_box_size = images.shape[1] // wanted_height
        width_box_size = images.shape[2] // wanted_width
        images_downscaled = np.ndarray(shape = (images.shape[0], wanted_height, wanted_width))

        print(images_downscaled.shape)
        for i in range(0, wanted_height):
            for j in range(0, wanted_width):
                start_i = i * height_box_size
                end_i = (i + 1) * height_box_size
                start_j = j * width_box_size
                end_j = (j + 1) * width_box_size
                images_downscaled[:,i,j] = np.mean(images[:, start_i:end_i, start_j:end_j, :], axis = (1,2,3))

        images_reshaped = images_downscaled.reshape(len(images), wanted_width * wanted_height)
        return images_reshaped


class DownSampleSkippingMethod(FeatureExtractionMethod):
    def __init__(self):
        self.name = 'SKIP'

    def __str__(self):
        return "Downsample the image by skipping pixels"

    def run(self, images, desired_vector_size):
        assert(desired_vector_size % 10 == 0)
        wanted_width = 10
        wanted_height = desired_vector_size // wanted_width
        width_skip_rate = images.shape[1] // wanted_width
        height_skip_rate = images.shape[2] // wanted_height
        images_downscaled = images[:, ::width_skip_rate, ::height_skip_rate]
        images_downscaled = np.mean(images_downscaled, axis=3)
        print(images_downscaled.shape)
        images_reshaped = images_downscaled.reshape(len(images), wanted_width * wanted_height)


        return images_reshaped


class DownSampleSkippingCVMethod(FeatureExtractionMethod):

    def __str__(self):
        return "Downsample the image by skipping pixels"

    def run(self, images, desired_vector_size):
        assert(desired_vector_size % 10 == 0)
        wanted_width = 10
        wanted_height = desired_vector_size // wanted_width
        ## we will use the cv resize method to do things
        resized_images = []
        for i in range(len(images)):
            resized_images.append( cv2.resize(images[i], (wanted_width, wanted_height)) )
        images_downscaled = np.stack(resized_images, axis = 0)
        ## make sure stacking works as expected
        assert(images_downscaled.shape == (len(images), 10, 10, 3))
        images_downscaled = np.mean(images_downscaled, axis=3)
        print(images_downscaled.shape)
        images_reshaped = images_downscaled.reshape(len(images), wanted_width * wanted_height)


        return images_reshaped

