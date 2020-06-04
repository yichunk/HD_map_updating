'''Script for converting MTSD annotations (1 json file per image) to coco format (single json file)

This script annotates MTSD for 1 class: warning sign. All non-warning signs are not labelled.

Change IMAGE_DIR, ANNOTATION_DIR according to the directory where the images and xml annotation files are stored

Change name of output .json file as needed


'''

# Adapted from shapes_to_coco.py from pycococreator by Waspinator

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from detectron2.structures import BoxMode
import pickle
import argparse


''''
Input:  annotations and images
Output: dictionary in detectron2 format stored as .pkl
'''

parser = argparse.ArgumentParser(description='Transfer the imgs and anns in given direcotries')
parser.add_argument(
    '--path_to_img',
    default='mtsd_fully_annotated/split_images/test',
    type=str,
    help='Path to read the spilt images')
parser.add_argument(
    '--path_to_ann',
    default='mtsd_fully_annotated/split_annotations/test',
    type=str,
    help='Path to read the spilt annotations')
parser.add_argument(
    '--path_to_pkl',
    default='mtsd_fully_annotated/detectron2_annotations/test_6.pkl',
    type=str,
    help='Path to store the pkl file')



#                 0      1            2         3: black and white    4:red circle               5
CATEGORIES =  ['stop', 'yield', 'do_not_enter', 'other_regulatory', 'other_prohibitory', 'warning_pedestrians']


def save_obj(obj, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.json'] #read the annotation files
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types]) #| is bitwise OR
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def save_dataset_dict(args):
    #IMAGE_DIR = os.path.join(ROOT_DIR, "split_images", split) #shapes_train2018
    #ANNOTATION_DIR = os.path.join(ROOT_DIR, "split_annotations", split)

    IMAGE_DIR       = args.path_to_img
    ANNOTATION_DIR  = args.path_to_ann

    dataset_dicts = []


    cate_id_cnt = np.zeros(len(CATEGORIES))

    image_id = 1
    segmentation_id = 1
    annotation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)

            #print(image_info)
            #coco_output["images"].append(image_info)


            record = {}
            record["file_name"] = image_filename
            record["image_id"] = image_info['id']
            record["height"] = image_info['height']
            record["width"] = image_info['width']

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                for annotation_filename in annotation_files: #open the file from the list annotation_files
                    with open(annotation_filename) as json_file:
                        annotation_data = json.load(json_file) #annotation_data is a python dict


                    objs = []
                    #print('objs', objs)
                    num_objects_max = len(annotation_data['objects'])

                    for num_object in range(num_objects_max):
                        if 'regulatory--stop--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 0
                        elif 'regulatory--yield--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 1
                        elif 'regulatory--no-entry--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 2
                        elif 'regulatory--no-parking--g2' in annotation_data['objects'][num_object]['label']:
                            class_id = 4
                        elif 'regulatory--maximum-speed-limit' in annotation_data['objects'][num_object]['label']:
                            if 'led' in annotation_data['objects'][num_object]['label']:
                                class_id = -1
                            elif 'g1' in annotation_data['objects'][num_object]['label']:
                                class_id = -1
                            elif '90' in annotation_data['objects'][num_object]['label']:
                                class_id = -1
                            elif '100' in annotation_data['objects'][num_object]['label']:
                                class_id = -1
                            elif '110' in annotation_data['objects'][num_object]['label']:
                                class_id = -1
                            elif '120' in annotation_data['objects'][num_object]['label']:
                                class_id = -1
                            elif '130' in annotation_data['objects'][num_object]['label']:
                                class_id = -1
                            else:
                                class_id = 3
                        elif 'regulatory--turn-right--g3' in annotation_data['objects'][num_object]['label']:
                            class_id = 3
                        elif 'regulatory--go-straight--g3' in annotation_data['objects'][num_object]['label']:
                            class_id = 3
                        elif 'regulatory--turn-left--g2' in annotation_data['objects'][num_object]['label']:
                            class_id = 3
                        elif 'regulatory--no-right-turn--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 4
                        elif 'regulatory--no-straight-through--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 4
                        elif 'regulatory--no-left-turn--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 4
                        elif 'warning--pedestrians-crossing--g4' in annotation_data['objects'][num_object]['label']:
                            class_id = 5
                        elif 'regulatory--bicycles-only--g3' in annotation_data['objects'][num_object]['label']:
                            class_id = -1
                        else:
                            class_id = -1
                        
                        if class_id != -1:
                            # print out info and count different categories
                            

                            x_min = annotation_data['objects'][num_object]['bbox']['xmin']
                            y_min = annotation_data['objects'][num_object]['bbox']['ymin']

                            x_max = annotation_data['objects'][num_object]['bbox']['xmax']
                            y_max = annotation_data['objects'][num_object]['bbox']['ymax']

                            poly = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_max), (x_min, y_max)]
                            #poly = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_max), (x_min, y_max)]
                            
                            obj = {
                            "bbox": [x_min, y_min, x_max, y_max],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly],
                            "category_id": class_id,
                            "iscrowd": 0
                            }
                            #if obj is not None:
                            objs.append(obj)
                            print('category:', CATEGORIES[class_id])
                            #print('obj:', obj)
                            cate_id_cnt[class_id] += 1
                            
                        num_object = num_object + 1 #continue to keep reading the bbox and label from next object
                        annotation_id = annotation_id + 1

                    if len(objs) > 0:
                        print('objs', objs)
                        print('image saved....')
                        record["annotations"] = objs
                        dataset_dicts.append(record)
            image_id = image_id + 1

    
    #print(dataset_dicts)
    save_obj(dataset_dicts, arg.path_to_pkl)

    for i, cnt in enumerate(cate_id_cnt):
        print(CATEGORIES[i]+ ':' + str(cnt))

    return dataset_dicts


if __name__ == "__main__":
    args = parser.parse_args()
    save_dataset_dict(args)

    #save_dataset_dict(split = 'val')
    #save_dataset_dict(split = 'train')
    #tsave_dataset_dict(split = 'test')

    '''
    from detectron2.data import DatasetCatalog, MetadataCatalog
    d = 'MTSD_samples'
    DatasetCatalog.register("MTSD_samples", lambda d=d: load_obj(d))
    MetadataCatalog.get("MTSD_samples").set(thing_classes=CATEGORIES)
    Meta_data = MetadataCatalog.get("MTSD_samples")
    #print(Meta_data)
    dataset_dicts = load_obj(d)
    import random
    from detectron2.utils.visualizer import Visualizer
    import cv2

    i = 0

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        print(d['annotations'])
        #print(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=Meta_data, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        #cv2_imshow(vis.get_image()[:, :, ::-1])
        cv2.imwrite(str(i) + '.jpg', vis.get_image()[:, :, ::-1])
        i += 1

    '''