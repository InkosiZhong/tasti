'''
This code allows you to create a TASTI for "night-street" using a target dnn run in realtime ("online").
Note that for performance reasons, we have reduced the intensity of the hyperparameters and are using a 
much smaller model. Look at the README.md file for information about how to get the data to run this code.
'''
import os
import cv2
import swag
import json
import tasti
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from scipy.spatial import distance
from collections import defaultdict
import torchvision.transforms as transforms
from tasti.examples.video_db import VideoDataset
from tasti.examples.video_db import video_db_is_close_helper
from tasti.examples.video_db import AggregateQuery
from tasti.examples.video_db import LimitQuery
from tasti.examples.video_db import SUPGPrecisionQuery
from tasti.examples.video_db import SUPGRecallQuery
from tasti.examples.video_db import LHSPrecisionQuery
from tasti.examples.video_db import LHSRecallQuery
from tasti.examples.video_db import AveragePositionAggregateQuery
from tasti.examples.video_db import TASTIConfig, PretrainConfig
from tasti.examples.offline.night_street import night_street_embedding_dnn_transform_fn
from tasti.examples.offline.night_street import night_street_target_dnn_transform_fn

# Feel free to change this!
ROOT_DATA_DIR = '/home/inkosizhong/Lab/VideoQuery/datasets/jackson-town-square'

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class Box:
    def __init__(self, box, object_name, confidence):
        self.box = box
        self.xmin = box[0]
        self.ymin = box[1]
        self.xmax = box[2]
        self.ymax = box[3]
        self.object_name = object_name
        self.confidence = confidence
        
    def __str__(self):
        return f'Box({self.xmin},{self.ymin},{self.xmax},{self.ymax},{self.object_name},{self.confidence})'
    
    def __repr__(self):
        return self.__str__()
    
class NightStreetOnlineIndex(tasti.Index):
    def get_target_dnn(self):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)
        return model
        
    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        return model
    
    def get_pretrained_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model
    
    def get_target_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-12-14')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-12-17')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=night_street_target_dnn_transform_fn
        )
        return video
    
    def get_embedding_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-12-14')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-12-17')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=night_street_embedding_dnn_transform_fn
        )
        return video
    
    def target_dnn_callback(self, target_dnn_output):
        boxes = target_dnn_output[0]['boxes'].detach().cpu().numpy()
        confidences = target_dnn_output[0]['scores'].detach().cpu().numpy()
        object_ids = target_dnn_output[0]['labels'].detach().cpu().numpy()
        label = []
        for i in range(len(boxes)):
            object_name = COCO_INSTANCE_CATEGORY_NAMES[object_ids[i]]
            if confidences[i] > 0.97 and object_name in ['car']:
                box = Box(boxes[i], object_ids[i], confidences[i])
                label.append(box)
        return label
        
    def is_close(self, label1, label2):
        objects = set()
        for obj in (label1 + label2):
            objects.add(obj.object_name)
        for current_obj in list(objects):
            label1_disjoint = [obj for obj in label1 if obj.object_name == current_obj]
            label2_disjoint = [obj for obj in label2 if obj.object_name == current_obj]
            is_redundant = video_db_is_close_helper(label1_disjoint, label2_disjoint)
            if not is_redundant:
                return False
        return True

class NightStreetOnlineConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = True
        self.do_training = True
        self.do_infer = True
        self.do_bucketting = True
        
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 7000
        self.nb_training_its = 12000
        
if __name__ == '__main__':
    config = NightStreetOnlineConfig()
    #config.eval() # use cache
    index = NightStreetOnlineIndex(config)
    index.init()

    query = AggregateQuery(index)
    query.execute(err_tol=0.1, confidence=0.1)

    query = LimitQuery(index)
    query.execute(want_to_find=4, nb_to_find=3)

    query = SUPGPrecisionQuery(index)
    query.execute(budget=100)

    query = SUPGRecallQuery(index)
    query.execute(budget=100)

    query = LHSPrecisionQuery(index)
    query.execute(budget=100)

    query = LHSRecallQuery(index)
    query.execute(budget=100)

    query = AveragePositionAggregateQuery(index)
    query.execute(err_tol=0.1, confidence=0.1)