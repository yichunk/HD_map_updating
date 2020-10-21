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

import utils

from map_boxes import mean_average_precision_for_boxes

import argparse

import os
os.environ['QT_QPA_PLATFORM']='offscreen'


parser = argparse.ArgumentParser(description='Calculate mAP and draw PR Curve')
parser.add_argument(
    '--path_to_model',
    default='./output_faster_rcnn/model_final.pth',
    type=str,
    help='Path to read the annotation dictionary')



'''
CATEGORIES =  ['stop', 'yield', 'do_not_enter', 'no_parking', 'speed_limit', 'turn_right', 'go_straight',
               'turn_left', 'no_right_turn', 'no_straight_through', 'no_left_turn','warning_pedestrians', 'bike_lane']
'''

#                 0      1            2         3: black and white    4:red circle               5
CATEGORIES =  ['stop', 'yield', 'do_not_enter', 'other_regulatory', 'other_prohibitory', 'warning_pedestrians']




def validate(cfg, Meta_data, split):
    notation_file = utils.load_obj(split)
    
    anns = []
    dets = []

    
    for i, image in enumerate(notation_file):
        print('Image {}/{} is under processing...'.format(i, len(notation_file)))
        image_height = image['height']
        image_width = image['width']

        # processing annotations
        ground_truth = np.zeros(len(CATEGORIES))
        
        objs = image["annotations"]

        #print('processing annotations...')
        for obj in objs:
            class_id = obj['category_id']
            ann_image_id = image['file_name']
            ann_label_name = CATEGORIES[class_id]
            ann_bbox = obj['bbox'] # x0, y0, x1, y1
            ann_bbox[0] /= image_width
            ann_bbox[1] /= image_height
            ann_bbox[2] /= image_width
            ann_bbox[3] /= image_height    

            ann = [ann_image_id, ann_label_name, ann_bbox[0], ann_bbox[2], ann_bbox[1], ann_bbox[3]]
            #print(ann)
            anns.append(ann)
        
        # processing predictions
        file_name = image['file_name']
        img = cv2.imread(file_name)
        outputs = predictor(img)

        
        scores = outputs["instances"].scores.to('cpu').numpy()
        pred_classes = outputs["instances"].pred_classes.to('cpu').numpy()
        bbox = outputs["instances"].pred_boxes.tensor.to('cpu').numpy()

        N = scores.shape[0]

        #print("processing predictions...")
        for i in range(N):
            det_image_id = image['file_name']
            class_id = pred_classes[i]
            det_label_name = CATEGORIES[class_id]
            det_bbox = bbox[i] # x0, y0, x1, y1
            det_bbox[0] /= image_width
            det_bbox[1] /= image_height
            det_bbox[2] /= image_width
            det_bbox[3] /= image_height   

            det = [det_image_id, det_label_name, scores[i], det_bbox[0], det_bbox[2], det_bbox[1], det_bbox[3]]
            #print(det)
            dets.append(det) 


    # ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
    # det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
    mean_ap, average_precisions = mean_average_precision_for_boxes(anns, dets)



if __name__ == "__main__":

    args = parser.parse_args()

    split = 'test_6'
    dataset_name = "mtsd_" + split
    DatasetCatalog.register(dataset_name, lambda d=split: utils.load_obj(split))
    MetadataCatalog.get(dataset_name).set(thing_classes=CATEGORIES)
    Meta_data = MetadataCatalog.get(dataset_name)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0 # set threshold for this model
    cfg.MODEL.WEIGHTS = args.path_to_model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORIES)

    cfg.merge_from_file('configs/small_objects.yaml')
    
    predictor = DefaultPredictor(cfg)

    validate(cfg, Meta_data, split)
