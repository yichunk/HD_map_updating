from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer
import cv2
import os
import pickle
import argparse


parser = argparse.ArgumentParser(description='Test the detectron2 dictionary file')
parser.add_argument(
    '--path_to_pkl',
    default='mtsd_fully_annotated/detectron2_annotations/train_syn_6.pkl',
    type=str,
    help='Path to read the annotation dictionary')

CATEGORIES =  ['stop', 'yield', 'do_not_enter', 'other_regulatory', 'other_prohibitory', 'warning_pedestrians']


def load_obj(file_path):
    #file_path =  os.path.join('mtsd_fully_annotated', 'detectron2_annotations', name + '.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    args = parser.parse_args()
    file_path = args.path_to_pkl
    DatasetCatalog.register("MTSD_samples", lambda file_path=file_path: load_obj(file_path))
    MetadataCatalog.get("MTSD_samples").set(thing_classes=CATEGORIES)
    Meta_data = MetadataCatalog.get("MTSD_samples")
    dataset_dicts = load_obj(file_path)
    

    i = 0

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        print(d['annotations'])
        #print(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=Meta_data, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        #cv2_imshow(vis.get_image()[:, :, ::-1])
        cv2.imwrite(str(i) + '.jpg', vis.get_image()[:, :, ::-1])
        i += 1