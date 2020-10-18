import pickle
import os

import cv2


def save_obj(obj, name):
    file_path =  os.path.join('mtsd_fully_annotated', 'detectron2_annotations', name + '.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    file_path =  os.path.join('mtsd_fully_annotated', 'detectron2_annotations', name + '.pkl')
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

