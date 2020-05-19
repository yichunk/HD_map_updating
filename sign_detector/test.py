import os
import numpy as np
import json

from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
import random
from detectron2.utils.visualizer import Visualizer
import cv2
import pickle


from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import glob as glob

#import utils

#from map_boxes import mean_average_precision_for_boxes

import os
os.environ['QT_QPA_PLATFORM']='offscreen'


#                 0      1            2         3: black and white    4:red circle               5
CATEGORIES =  ['stop', 'yield', 'do_not_enter', 'other_regulatory', 'other_prohibitory', 'warning_pedestrians']


def test_on_image(cfg):

    file_names = glob.glob('test_imgs/*.jpg')
    output_dir = 'outputs/'
   

    os.makedirs(output_dir, exist_ok = True) 
    for i, file_name in enumerate(file_names):
        img = cv2.imread(file_name)
        outputs = predictor(img)

        scores = outputs["instances"].scores.to('cpu').numpy()
        pred_classes = outputs["instances"].pred_classes.to('cpu').numpy()
        bbox = outputs["instances"].pred_boxes.tensor.to('cpu').numpy()


        for obj_cnt in range(scores.shape[0]):
            cv2.rectangle(img, (bbox[obj_cnt][0], bbox[obj_cnt][1]), (bbox[obj_cnt][2], bbox[obj_cnt][3]), (255,0,0), 2)
        
        cv2.imwrite(os.path.join(output_dir, 'output_{}.jpg'.format(i)), img)



if __name__ == "__main__":

    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9 # set threshold for this model
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORIES)
    
    predictor = DefaultPredictor(cfg)

    test_on_image(cfg)

    #validate(cfg, Meta_data, split)
