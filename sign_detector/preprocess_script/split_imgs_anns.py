
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='copy split images and annotation based on the given splited')
parser.add_argument(
    '--root_dir',
    default='mtsd_fully_annotated',
    type=str,
    help='Path to the mtsd dataset')
parser.add_argument(
    '--path_to_img',
    default='mtsd_fully_annotated/new_split_imgs/test',
    type=str,
    help='Path to store the spilt images')
parser.add_argument(
    '--path_to_ann',
    default='mtsd_fully_annotated/new_split_anns/test',
    type=str,
    help='Path to store the spilt annotations')
parser.add_argument(
    '--path_to_split_file',
    default='mtsd_fully_annotated/splits/test_allsigns.txt',
    type=str,
    help='Path to the split files')


# Based on the .txt files provided by Mapillary in folder splits, copy out the relevant images into
# the train_images, val_images, or test_images folders
def main(args):
    ROOT_DIR = args.root_dir
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

    SPLIT_FILE = args.path_to_split_file

    # Change this depending on which folder to copy to
    img_dest_dir = args.path_to_img
    if not os.path.exists(img_dest_dir):
        os.makedirs(img_dest_dir)

    ann_dest_dir = args.path_to_ann
    if not os.path.exists(ann_dest_dir):
        os.makedirs(ann_dest_dir)

    # Change .txt file name based on whether you want to pick out train, val, or test images
    with open(SPLIT_FILE, "r") as fd:
        lines = fd.read().splitlines()
        max_lines = len(lines) #print number of images to be copied
        print(len(lines))
        for num_line in range(max_lines):
            filename = os.path.join(IMAGE_DIR, lines[num_line]) + ".jpg"
            shutil.copy(filename, img_dest_dir)
            afilename = os.path.join(ANNOTATION_DIR, lines[num_line]) + ".json"
            shutil.copy(afilename, ann_dest_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
