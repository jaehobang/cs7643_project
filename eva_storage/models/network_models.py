from abc import ABCMeta, abstractmethod


import os
import config
import torch
from logger import Logger



class NetworkTemplate(metaclass=ABCMeta):
    def __init__(self):
        self.model = None
        self.logger = Logger()

    @abstractmethod
    def train(self, dataset_loader, save_name, total_epochs, cuda = True, lr = 0.0001, weight_decay = 1e-6):
        pass

    @abstractmethod
    def execute(self, dataset_loader, cuda = True):
        pass


    def save(self, save_name, epoch=0):
          """
          Save the model
          We will save this in the
          :return: None
          """
          eva_dir = config.eva_dir
          dir = os.path.join(eva_dir, 'data', 'models', '{}-epoch{}.pth'.format(save_name, epoch))
          print("Saving the trained model as....", dir)

          torch.save(self.model.state_dict(), dir)


    def load(self, load_dir):
        if os.path.exists(load_dir):  ## os.path.exists works on folders and files

            self.model.load_state_dict(torch.load(load_dir))
            self.logger.info("Model load success!")

        else:
            self.logger.error("Checkpoint does not exist returning")
