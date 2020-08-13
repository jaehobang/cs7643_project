
from abc import ABC, abstractmethod
import numpy as np
import skimage.measure
import torch
import math
import cv2

class FeatureExtractionMethod(ABC):

    @abstractmethod
    def run(self, images, desired_vector_size):
        pass



class VGG16Method(FeatureExtractionMethod):

    def __str__(self):
        return "Extract the features using vgg16 network"

    def run(self, images, desired_vector_size):
        import torchvision.models as models
        #import os
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ## we want to run everything on gpu 1

        vgg16 = models.vgg16(pretrained = True)
        vgg16.cuda()
        torch.set_default_tensor_type('torch.FloatTensor')
        images_torch = torch.Tensor(images)
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

