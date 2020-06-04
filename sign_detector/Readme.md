# Sign Detector

#### Build the detectorn2 dictionary file based on Mapillary Traffic Sign Dataset

1. Request the dataset: https://www.mapillary.com/dataset/trafficsign

2. Split the images and annotations based on the split files (provided in split_files)

   ```
   python split_imgs_anns.py --root_dir [root_dir_of_dataset] --path_to_img [path_to_store_imgs] --path_to_ann [path_to_store_annotations] --path_to_split_file [path_to_the_split_file]
   ```

3. Generate the annotation file in detectron2 format (Pre-defined 6 classes)

   ```
   python trafficsigns_to_detectron_6_class.py --path_to_img [path_to_splited_imgs] --path_to_ann [--path_to_splited_annotations] --path_to_pkl [path_to_the_annotation_file]
   ```

   ![Pre-defined_6_classes](Pre-defined_6_classes.png)

4. Test the annotation file

   ```
   python test_dict_6_class.py --path_to_pkl [path_to_the_annotation_file]
   ```
   ![annotation_example](annotation_example.jpg)
   



#### Test on the pre-build model

1. Set up the enviornment: 

   Use environment.yml to build the needed environement

   ```
   conda env create -f environment.yml
   ```

2. Download model weight:

   Download link: https://drive.google.com/drive/folders/1tF7upcJhw_5rtj11odlgaO4ZlW2AjMPn?usp=sharing

3. Test on images:

   Put the images in the test_imgs directory

   Excuete the script: python test.py

   Example output result:

   

![test_out_example](test_out_example.jpg)