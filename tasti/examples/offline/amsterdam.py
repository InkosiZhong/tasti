'''
This code allows you to reproduce the results in the paper corresponding to the "night-street" dataset.
The term 'offline' refers to the fact that all the target dnn outputs have already been computed.
If you like to run the 'online' version (target dnn runs in realtime), take a look at "night_street_online.py". 
Look at the README.md file for information about how to get the data to run this code.
'''
import os
import cv2
import swag
import json
import tasti
import torch
import pandas as pd
import numpy as np
import torchvision
from scipy.spatial import distance
from collections import defaultdict
from tqdm.autonotebook import tqdm
from tasti.examples.video_db import VideoDataset, LabelDataset
from tasti.examples.video_db import video_db_is_close_helper
from tasti.examples.video_db import AggregateQuery
from tasti.examples.video_db import LimitQuery
from tasti.examples.video_db import SUPGPrecisionQuery
from tasti.examples.video_db import SUPGRecallQuery
from tasti.examples.video_db import LHSPrecisionQuery
from tasti.examples.video_db import LHSRecallQuery
from tasti.examples.video_db import AveragePositionAggregateQuery
from tasti.examples.video_db import TASTIConfig, PretrainConfig

# Feel free to change this!
ROOT_DATA_DIR = '/home/inkosizhong/Lab/VideoQuery/datasets/amsterdam'
        
'''
Preprocessing function of a frame before it is passed to the Embedding DNN.
'''
def amsterdam_embedding_dnn_transform_fn(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

def amsterdam_target_dnn_transform_fn(frame):
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

class AmsterdamOfflineIndex(tasti.Index):
    def get_target_dnn(self):
        '''
        In this case, because we are running the target dnn offline, so we just return the identity.
        '''
        model = torch.nn.Identity()
        return model
        
    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        return model
    
    def get_pretrained_embedding_dnn(self):
        '''
        Note that the pretrained embedding dnn sometime differs from the embedding dnn.
        '''
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model
    
    def get_target_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-04-10')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-04-11')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=amsterdam_target_dnn_transform_fn
        )
        return video
    
    def get_embedding_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-04-10')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-04-11')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=amsterdam_embedding_dnn_transform_fn
        )
        return video
    
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        if train_or_test == 'train':
            labels_fp = os.path.join(ROOT_DATA_DIR, '2017-04-10-yolov5l.csv')
        else:
            labels_fp = os.path.join(ROOT_DATA_DIR, '2017-04-11-yolov5l.csv')
        labels = LabelDataset(
            labels_fp=labels_fp,
            length=len(target_dnn_cache),
            query_objects=self.config.query_objects
        )
        return labels
    
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
    
# TASTIConfig: default TASTI behavior
# PretrainConfig: skip mining and training step, using pretrained embedding NN
class AmsterdamOfflineConfig(TASTIConfig):
    def __init__(self):
        super().__init__()
        self.cache_root = 'cache/amsterdam'
        self.query_objects = ['car']

        # Change the following to False if you want to skip some steps.
        # self.do_mining = True
        # self.do_training = True
        # self.do_infer = True
        # self.do_bucketting = True
        
        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 7000
        self.nb_training_its = 12000
    
if __name__ == '__main__':
    config = AmsterdamOfflineConfig()
    index = AmsterdamOfflineIndex(config)
    index.init()

    query = AggregateQuery(index)
    query.execute_metrics(err_tol=0.01, confidence=0.05)

    # evaluate average performance
    from tasti.examples.test_utils import eval_nb_samples
    nb_samples = eval_nb_samples(query.execute, 10, err_tol=0.01, confidence=0.05)
    print('AggregateQuery:', nb_samples)
    
    query = AveragePositionAggregateQuery(index)
    query.execute_metrics(err_tol=0.005, confidence=0.05)

    query = LimitQuery(index)
    query.execute_metrics(want_to_find=5, nb_to_find=10)

    query = SUPGPrecisionQuery(index)
    query.execute_metrics(10000)

    query = SUPGRecallQuery(index)
    query.execute_metrics(10000)

    query = LHSPrecisionQuery(index)
    query.execute_metrics(10000)

    query = LHSRecallQuery(index)
    query.execute_metrics(10000)    
