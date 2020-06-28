"""
We want to create a wrapper around ssd, that can be easily used for training and evaluation
=> we could possibly define an annotation saving file as well


"""

from logger import Logger
import os
from others.amdegroot.data.seattle import SEATTLE_CLASSES
from others.amdegroot.ssd import build_ssd
import torch
from data.dataset import EvaluationDataset
from others.amdegroot.data import BaseTransform
import torch.backends.cudnn as cudnn
from timer import Timer
import numpy as np




class SSDLoader:

    def __init__(self, model_dir = None, labelmap = None, evaluation = True):
        self.logger = Logger()
        if not model_dir:
            model_dir = '/nethome/jbang36/eva_jaeho/others/amdegroot/weights/finalists/ssd300_UAD_0408_90000.pth'
        assert(os.path.isdir(model_dir))
        self.logger.info(f"model directory is {model_dir}")
        if not labelmap:
            labelmap = SEATTLE_CLASSES ## format is ('car', 'bus', 'others', 'van')
        assert(type(labelmap) == list or type(labelmap) == tuple)
        self.labelmap = labelmap
        self.model = self.load_model(model_dir, evaluation)


    def load_model(self, model_dir, evaluation = True):
        num_classes = len(self.labelmap) + 1
        if evaluation:
            net = build_ssd('test', 300, num_classes)
            net.load_state_dict(torch.load(model_dir))
            net.eval()
        else: ## evaluation = False
            net = build_ssd('train', 300, num_classes)
            net.load_state_dict(torch.load(model_dir))
            net.train()
        return net

    def detect(self, images, **kwargs):
        if 'dataset_mean' in kwargs.keys() and \
                (type(kwargs['dataset_mean']) == list or type(kwargs['dataset_mean']) == tuple):
            dataset_mean = kwargs['dataset_mean']
        else:
            dataset_mean = (92, 111, 120) ## default number for SEATTLE2 dataset
        dataset = EvaluationDataset(images, transform = BaseTransform(300, dataset_mean))
        if 'cuda' in kwargs.keys() and kwargs['cuda']:
            self.model = self.model.cuda()
            cudnn.benchmark = True
        box_list = self._detect_boxes(self.model, dataset, **kwargs)
        ## TODO:box_list is in box_list[label][frameid] format. Each element is a [xmin, ymin, xmax, ymax] format
        ###  Need to come back to this line for _detect_boxes
        ### shall we put it in the uad loading format??
        box_list = self.convert2jformat(box_list)

        return box_list

    def convert2jformat(self, box_list):
        ## we convert from amdegroot to j format for boxes_detected
        num_images = len(box_list[0])
        jlabels = [[] for _ in range(num_images)]
        jboxes = [[] for _ in range(num_images)]
        for i in range(num_images):
            for cat_i in range(len(box_list)):
                for element in box_list[cat_i]:
                    jlabels[i].append(self.labelmap[cat_i])
                    jboxes[i].append(box_list[cat_i][element])

        #jlabels = [['car', 'car', 'bus', 'others'], []......]
        #jboxes = [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax].....]
        return jlabels, jboxes


    def _detect_boxes(self, net, dataset, **kwargs):
        num_images = len(dataset)
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(len(self.labelmap) + 1)] ## for label in labelmap, for all images

        # timers
        _t = {'im_detect': Timer(), 'misc': Timer()}

        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)
            ### annotation might not be available

            x = torch.Variable(im.unsqueeze(0))
            if 'cuda' in kwargs.keys() and kwargs['cuda']:
                x = x.cuda()
            _t['im_detect'].tic()
            detections = net(x).data
            detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                ## x_min, y_min, x_max, y_max ?? - probably
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                        num_images, detect_time))


        return all_boxes

