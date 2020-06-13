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
import argparse



parser = argparse.ArgumentParser(description='Test the detectron2 dictionary file')
parser.add_argument(
    '--path_to_pkl',
    default='mtsd_fully_annotated/detectron2_annotations/test_6.pkl',
    type=str,
    help='Path to read the annotation dictionary')
parser.add_argument(
    '--path_to_model',
    default='model_final.pth',
    type=str,
    help='Path to read the weights')
parser.add_argument(
    '--path_to_seq',
    default='model_final.pth',
    type=str,
    help='Path to read the sequence')
parser.add_argument(
    '--output_dir',
    default='output_faster_rcnn',
    type=str,
    help='Directory to store all the outputs (models, tensorboard files, etc)')



#                 0      1            2         3: black and white    4:red circle               5
CATEGORIES =  ['stop', 'yield', 'do_not_enter', 'other_regulatory', 'other_prohibitory', 'warning_pedestrians']


def load_obj(file_path):
    #file_path =  os.path.join('mtsd_fully_annotated', 'detectron2_annotations', name + '.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def recorder_video(img_list, file_name):
    
    if len(img_list) == 0:
        return
    height, width, layers = img_list[0].shape
    size = (width,height)
 
    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
 
    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()



def validate_on_argoverse_single_seq(cfg, Meta_data, path_to_seq_dir):

    i = 0

    PATH_TO_TEST_IMAGES_DIR = path_to_seq_dir
    TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
    TEST_IMAGE_PATHS.sort(reverse=False)
    

    scores_list = []
    boxes_list = []
    images_list = []
    category_id_list = []
    output_images_list = []
    category_map = CATEGORIES
    print('processing on outputs...')

    for img_path in TEST_IMAGE_PATHS:
        img = cv2.imread(img_path)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1],
                   metadata=Meta_data, 
                   scale=0.8, 
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        #if len(outputs["instances"]) > 0:
        file_name = img_path.split('/')[-1]

        scores_list.append(outputs["instances"].scores.to('cpu').numpy())
        boxes_list.append(outputs["instances"].pred_boxes.tensor.to('cpu').numpy())
        images_list.append(file_name)
        category_id_list.append(outputs["instances"].pred_classes.to('cpu').numpy())

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        #file_path = os.path.join(save_dir, file_name)
        #cv2.imwrite(file_path, v.get_image()[:, :, ::-1])
        output_images_list.append(v.get_image()[:, :, ::-1])
        i += 1
    
    
    save_dir = os.path.join(cfg.OUTPUT_DIR, 'records')
    #save_dir = './output/records'
    os.makedirs(save_dir, exist_ok=True)
    
    save_name = path_to_seq_dir.split('/')[-1]
    save_npz_path = os.path.join(save_dir, '{}.npz'.format(save_name))
    print('saving {}...'.format(save_npz_path))
    np.savez(save_npz_path, images_list=images_list, boxes_list=boxes_list, scores_list=scores_list,category_id_list=category_id_list, category_map=category_map)
    save_video_path = os.path.join(save_dir, '{}.mp4'.format(save_name))
    print('saving {}...'.format(save_video_path))
    recorder_video(output_images_list, save_video_path)
    

if __name__ == "__main__":
    args = parser.parse_args()

    output_dir = args.output_dir

    d = args.path_to_pkl
    dataset_name = "mtsd"
    DatasetCatalog.register(dataset_name, lambda d=d: load_obj(d))
    MetadataCatalog.get(dataset_name).set(thing_classes=CATEGORIES)
    Meta_data = MetadataCatalog.get(dataset_name)
    #print(Meta_data)
    dataset_dicts = load_obj(d)
   

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9 # set threshold for this model
    cfg.MODEL.WEIGHTS = args.path_to_model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORIES)
    predictor = DefaultPredictor(cfg)
    cfg.OUTPUT_DIR = output_dir


    #path_to_argo_dir = os.path.join('argoverse-tacking', 'train1')
    

    path_to_argo_dir = 'ring_front_center'
    
    validate_on_argoverse_single_seq(cfg, Meta_data,path_to_argo_dir)

    #validate_on_argoverse_multiple_seqs(cfg, Meta_data, path_to_argo_dir)
