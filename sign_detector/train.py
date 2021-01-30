# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import os
import numpy as np
import json
import argparse

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


parser = argparse.ArgumentParser(description='Test the detectron2 dictionary file')
parser.add_argument(
    '--path_to_pkl',
    default='mtsd_fully_annotated/detectron2_annotations/train_syn_6.pkl',
    type=str,
    help='Path to read the annotation dictionary')
parser.add_argument(
    '--output_dir',
    default='output_syn_data',
    type=str,
    help='Directory to store all the outputs (models, tensorboard files, etc)')


#                 0      1            2         3: black and white    4:red circle               5
CATEGORIES =  ['stop', 'yield', 'do_not_enter', 'other_regulatory', 'other_prohibitory', 'warning_pedestrians']



def load_obj(file_path):
    #file_path =  os.path.join('mtsd_fully_annotated', 'detectron2_annotations', name + '.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def train(args):

    # Parameters
    output_dir = args.output_dir
    # Define train & valid dataset
    train_dataset_name = 'mtsd_train'
    # Register train dataset
    d = args.path_to_pkl
    DatasetCatalog.register(train_dataset_name, lambda d=d: load_obj(d))
    MetadataCatalog.get(train_dataset_name).set(thing_classes=CATEGORIES)
    Meta_data = MetadataCatalog.get(train_dataset_name)
    #print(Meta_data)
    dataset_dicts = load_obj(d)

    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    #cfg.SOLVER.MAX_ITER = 30000   # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 180000   # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORIES)  # only has one class (ballon)
    cfg.OUTPUT_DIR = output_dir
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # update weights
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
