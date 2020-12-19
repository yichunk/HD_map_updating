from PIL import Image, ImageDraw, ImageFilter
import json
import random
import argparse
import os
import fnmatch
import re
import random
''''
Input1:     annotations and images of non-target background
Input2;     collected target object
Output:     new generated sythetic image and annotation
'''

parser = argparse.ArgumentParser(description='Generate new sythetic data')
parser.add_argument(
    '--path_to_img',
    default='mtsd_fully_annotated/new_split_imgs/train_non_target',
    type=str,
    help='Path to read the non-target images')
parser.add_argument(
    '--path_to_ann',
    default='mtsd_fully_annotated/new_split_anns/train_non_target',
    type=str,
    help='Path to read the non-target annotations')
parser.add_argument(
    '--path_to_objects',
    default='mtsd_fully_annotated/new_split_imgs/train_crop_objects',
    type=str,
    help='Path to read the non-target annotations')    
parser.add_argument(
    '--path_to_out_img',
    default='mtsd_fully_annotated/sythetic_imgs',
    type=str,
    help='Path to store the synthetic images')
parser.add_argument(
    '--path_to_out_ann',
    default='mtsd_fully_annotated/sythetic_anns',
    type=str,
    help='Path to store the synthetic annotations')


id_to_sign_label = [ '',      #0
                    'regulatory--stop--g1', #1 
                    'regulatory--yield--g1', #2
                    'regulatory--no-entry--g1', #3 
                    'regulatory--no-parking--g2', #4
                    'regulatory--maximum-speed-limit-15--g3', #5
                    'regulatory--turn-right--g3', #6
                    'regulatory--go-straight--g3', #7
                    'regulatory--turn-left--g2', #8
                    'regulatory--no-right-turn--g1', #9
                    'regulatory--no-straight-through--g1', #10
                    'regulatory--no-left-turn--g1', #11
                    'warning--pedestrians-crossing--g4' #12
                   ]

id_to_sign_label_weights = [0, 
                            4, 
                            4, 
                            4,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            4]

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

def random_paste_one_object(foreground, background, backgroubd_ann, object_label):
    img_w, img_h = background.size
    obj_to_background_ratio = random.uniform(0.01, 0.1)
    offset_x = round(random.uniform(0.2, 0.8) * img_w)
    offset_y = round(random.uniform(0.2, 0.8) * img_h)
    
    new_obj_w = round(obj_to_background_ratio * img_w)
    new_obj_h = round(obj_to_background_ratio * img_h)
    #print(new_obj_w)
    #print(new_obj_h)
    foreground = foreground.resize((new_obj_w, new_obj_h))
    background.paste(foreground, (offset_x, offset_y))
    if len(backgroubd_ann['objects']) != 0:
        new_obj = backgroubd_ann['objects'][-1].copy()
        new_obj['bbox']['xmin'] = offset_x
        new_obj['bbox']['ymin'] = offset_y
        new_obj['bbox']['xmax'] = offset_x + new_obj_w
        new_obj['bbox']['ymax'] = offset_y + new_obj_h
        new_obj['label'] = object_label
        backgroubd_ann['objects'].append(new_obj)

def generate_syn_data(args):
    IMAGE_DIR       = args.path_to_img
    ANNOTATION_DIR  = args.path_to_ann
    OBJ_DIR         = args.path_to_objects
    obj_lists       = []

    for i in range(13):
        obj_lists.append([])

    for root, _, files in os.walk(OBJ_DIR):
        image_files = filter_for_jpeg(root, files)
        for image_file in image_files:
            pure_file_name = image_file.split('/')[-1]
            pure_file_name = pure_file_name.split('.')[0]
            label_idx = int(pure_file_name.split('_')[1])
            #print(label_idx, len(obj_lists))
            obj_lists[label_idx].append(image_file)
        

    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        for image_filename in image_files:
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename) 
                for annotation_filename in annotation_files: #open the file from the list annotation_files
                    with open(annotation_filename) as json_file:
                        annotation_data = json.load(json_file) #annotation_data is a python dict
                    
                    background_img = Image.open(image_filename)
                    sign_cate = random.choices(list(range(13)), weights=id_to_sign_label_weights,k=1) 
                    sign_cate = sign_cate[0]
                    object_img_name = random.choice(obj_lists[sign_cate])
                    object_img = Image.open(object_img_name)
                    print(sign_cate)
                    print(image_filename)
                    print(annotation_filename)
                    random_paste_one_object(object_img, background_img, annotation_data, id_to_sign_label[sign_cate])

                    save_img_name = image_filename.split('/')[-1]
                    save_ann_name = annotation_filename.split('/')[-1]

                    save_img_name = os.path.join(args.path_to_out_img, save_img_name)
                    save_ann_name = os.path.join(args.path_to_out_ann, save_ann_name)
                    
                    with open(save_ann_name, 'w') as outfile:
                        json.dump(annotation_data, outfile)

                    background_img.save(save_img_name, quality=95)


                

    
if __name__ == "__main__":
    args = parser.parse_args()
    generate_syn_data(args)
    '''
    im1 = Image.open('MTSD_samples/North_America/f82itz69069z_LC7xM6UIA.jpg')
    im2 = Image.open('MTSD_samples/7_5_6.jpg') #image_id_class_id_num_object

    ann1 = None
    with open('MTSD_samples/North_America/f82itz69069z_LC7xM6UIA.json') as json_file:
        ann1 = json.load(json_file) #annotation_data is a python dict

    random_paste_one_object(im2, im1, ann1, id_to_sign_label[5])

    
    with open('output.json', 'w') as outfile:
        json.dump(ann1, outfile)
    
    im1.save('output.jpg', quality=95)
    '''