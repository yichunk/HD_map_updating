
import datetime
import json
import os
import shutil
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import argparse



parser = argparse.ArgumentParser(description='split images and annotation into target and non-target based on the given imgs and anns')
parser.add_argument(
    '--path_to_img',
    default='mtsd_fully_annotated/split_images/train',
    type=str,
    help='Path to the orignal images')
parser.add_argument(
    '--path_to_ann',
    default='mtsd_fully_annotated/split_annotations/train',
    type=str,
    help='Path to store the spilt annotations')
parser.add_argument(
    '--path_to_crop_object',
    default='mtsd_fully_annotated/new_split_imgs/train_crop_objects',
    type=str,
    help='Path to store the croped objects')
parser.add_argument(
    '--path_to_target_img',
    default='mtsd_fully_annotated/new_split_imgs/train_target',
    type=str,
    help='Path to store the spilt images')
parser.add_argument(
    '--path_to_target_ann',
    default='mtsd_fully_annotated/new_split_anns/train_target',
    type=str,
    help='Path to store the spilt annotations')
parser.add_argument(
    '--path_to_non_target_img',
    default='mtsd_fully_annotated/new_split_imgs/train_non_target',
    type=str,
    help='Path to store the spilt images')
parser.add_argument(
    '--path_to_non_target_ann',
    default='mtsd_fully_annotated/new_split_anns/train_non_target',
    type=str,
    help='Path to store the spilt annotations')


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

def process_split(args):

    IMAGE_DIR = args.path_to_img
    ANNOTATION_DIR = args.path_to_ann

    img_target_dest_dir = args.path_to_target_img
    if not os.path.isdir(img_target_dest_dir):
        os.mkdir(img_target_dest_dir)

    ann_target_dest_dir = args.path_to_target_ann
    if not os.path.isdir(ann_target_dest_dir):
        os.mkdir(ann_target_dest_dir)

    img_non_target_dest_dir = args.path_to_non_target_img
    if not os.path.isdir(img_non_target_dest_dir):
        os.mkdir(img_non_target_dest_dir)

    ann_non_target_dest_dir = args.path_to_non_target_ann
    if not os.path.isdir(ann_non_target_dest_dir):
        os.mkdir(ann_non_target_dest_dir)
    
    crop_object_dir = args.path_to_crop_object
    if not os.path.isdir(crop_object_dir):
        os.mkdir(crop_object_dir)

    image_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        
        # go through each image
        for image_filename in image_files:
            #image = Image.open(image_filename)   #should not need these 3 lines
            #image_info = pycococreatortools.create_image_info(
            #    image_id, os.path.basename(image_filename), image.size)
            #coco_output["images"].append(image_info)
            im = Image.open(image_filename)

            is_target = False
            ann_name = None
            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
                for annotation_filename in annotation_files: #open the file from the list annotation_files
                    with open(annotation_filename) as json_file:
                        annotation_data = json.load(json_file) #annotation_data is a python dict
                    
                    

                    num_objects_max = len(annotation_data['objects'])

                    for num_object in range(num_objects_max):
                        #modify the conditions/add conditions here depending on what signs you want to pick out
                        #can write to multiple .txt files for different categories
                        if 'regulatory--stop--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 1
                        elif 'regulatory--yield--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 2
                        elif 'regulatory--no-entry--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 3
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
                                class_id = 5
                        elif 'regulatory--turn-right--g3' in annotation_data['objects'][num_object]['label']:
                            class_id = 6
                        elif 'regulatory--go-straight--g3' in annotation_data['objects'][num_object]['label']:
                            class_id = 7
                        elif 'regulatory--turn-left--g2' in annotation_data['objects'][num_object]['label']:
                            class_id = 8
                        elif 'regulatory--no-right-turn--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 9
                        elif 'regulatory--no-straight-through--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 10
                        elif 'regulatory--no-left-turn--g1' in annotation_data['objects'][num_object]['label']:
                            class_id = 11
                        elif 'warning--pedestrians-crossing--g4' in annotation_data['objects'][num_object]['label']:
                            class_id = 12
                        else:
                            class_id = -1


                        if class_id != -1: #target
                            is_target = True

                            #Save the crop obejcts
                            x_min = annotation_data['objects'][num_object]['bbox']['xmin']
                            y_min = annotation_data['objects'][num_object]['bbox']['ymin']

                            x_max = annotation_data['objects'][num_object]['bbox']['xmax']
                            y_max = annotation_data['objects'][num_object]['bbox']['ymax']
                            crop = im.crop((x_min, y_min, x_max, y_max))
                            save_path = '{}_{}_{}.jpg'.format(image_id, class_id, num_object)
                            save_path = os.path.join(crop_object_dir, save_path)
                            print('croped image: {} is saved'.format(save_path))
                            crop.save(save_path, quality=95)

                        #else: #non target
                        num_object = num_object + 1
            if len(annotation_files) == 1:
                filename = image_filename
                afilename = annotation_files[0]
                
                if is_target == True:
                    shutil.copy(filename, img_target_dest_dir)
                    shutil.copy(afilename, ann_target_dest_dir) 
                else:
                    shutil.copy(filename, img_non_target_dest_dir)
                    shutil.copy(afilename, ann_non_target_dest_dir) 
            
            im.close()
            image_id = image_id + 1



if __name__ == "__main__":
    args = parser.parse_args()
    process_split(args)
