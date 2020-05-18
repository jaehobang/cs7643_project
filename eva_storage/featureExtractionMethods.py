
from abc import ABC, abstractmethod
import numpy as np
import skimage.measure




class FeatureExtractionMethod(ABC):

    @abstractmethod
    def run(self, images, desired_vector_size):
        pass



class VGG16Method(FeatureExtractionMethod):

    def __str__(self):
        return "Extract the features using vgg16 network"

    def run(self, images, desired_vector_size):
        import torchvision.models as models
        vgg16 = models.vgg16(pretrained = True)


        pass


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
        wanted_width = 10
        wanted_height = desired_vector_size // wanted_width
        height_box_size = images.shape[1] // wanted_height
        width_box_size = images.shape[2] // wanted_width

        images_reshaped = skimage.measure.block_reduce(images, (1, height_box_size, width_box_size, 3), np.mean)
        images_reshaped = np.squeeze(images_reshaped, axis = 3)
        images_reshaped = images_reshaped.reshape(len(images), wanted_height * wanted_width)

        return images_reshaped




class DownSampleMeanMethod(FeatureExtractionMethod):

    def __str__(self):
        return "Downsample the image by avg pixel blocks"

    def run(self, images, desired_vector_size):
        assert(desired_vector_size % 10 == 0)
        wanted_width = 10
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


