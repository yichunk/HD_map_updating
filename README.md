# HD_map_updating
## Directory Structure
- data: test data for development
- third_party: third_party git modules
    - colmap
    - ORB-SLAM
    - argoverse-api
- triangulation: 3d points localization of 2d traffic sign
- 3D_projection: transformation from colmap coordinates to argoverse coordinates
- sign_detector: traffic sign detector, based on Detectron2

## High Level Pipeline
1. Run traffic sign detection on raw images `./sign_detector`
2. Run Colmap SfM on raw images `colmap_script.txt`
3. Project Colmap points to argoverse map `3D_projection`

## Step 1: Run Traffic Sign Detection
1. [Visit the ReadMe](/sign_detector)
## Step 2: Run Colmap
1. Install [Colmap](https://colmap.github.io/) from the premade [binaries](https://demuc.de/colmap/#download). For example, `COLMAP-3.5-mac-no-cuda.zip`
2. After unpackaging it in your user directory, you can use the command line `~/COLMAP.app/Contents/MacOS/colmap`
3. Download the [sample v1.1](https://s3.amazonaws.com/argoai-argoverse/tracking_sample_v1.1.tar.gz) 3D tracking dataset from argoverse. Configure it to only include the following folders, so that it has the following structure. Here we only use two cameras out of the 7 that exist on the camera ring and have placed it in our home directory:
````
    /path/to/argoverse-tracking/sample/...
    +── c6911883-1843-3727-8eaa-41dc8cda8993
    │   +── ring_front_center
    │    │   +── image1.jpg
    │    │   +── image2.jpg
    │    │   +── ...
    │    │   +── imageN.jpg
    │   +── ring_front_left
    │    │   +── image1.jpg
    │    │   +── image2.jpg
    │    │   +── ...
    │    │   +── imageN.jpg
````
4. Run your extractor (1 minute). We specify the location of
- the database to to save the intermediate colmap database. 
- the directory to where the argoverse images are stored. This method will extract features from all of the images in the folder including those in subfoolders.
````
~/COLMAP.app/Contents/MacOS/colmap feature_extractor --database_path ~/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993/database.db --image_path ~/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993 --ImageReader.single_camera_per_folder 1
````
5. Run your matcher (6 hours and 2.6gb). We specify
- database_path: the database path from step 4
````
~/COLMAP.app/Contents/MacOS/colmap exhaustive_matcher --database_path ~/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993/database.db
````
6. Run the mapper. We specify
- database_path: the database path from step 4
- output_path: the path to store intermediate binary results. This needs to be manually created.
````
~/COLMAP.app/Contents/MacOS/colmap mapper --database_path ~/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993/database.db --image_path ~/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993 --output_path ~/argoverse-tracking/sample/tmp
````
7. Run the model converter. We specify
- input_path: the path from step 6 that stored the intermediate binary results
- output_path: the path to save the final colmap models, which is input to the 3D projection module. 
directory to save final colmap models, which is input to the 3D projection module.
````
~/COLMAP.app/Contents/MacOS/colmap model_converter --input_path ~/argoverse-tracking/sample/tmp --output_path ~/argoverse-tracking/sample/output --output_type TXT
````
Your end folder should look like
````
    /path/to/argoverse-tracking/sample/...
    +── tmp
    │   +── 0
    │    │   +── cameras.bin
    │    │   +── images.bin
    │    │   +── points3D.bin
    │    │   +── project.ini
    +── output
    │    +── cameras.txt
    │    +── images.txt
    │    +── points3D.txt
    +── c6911883-1843-3727-8eaa-41dc8cda8993
    │   +── ring_front_center
    │    │   +── image1.jpg
    │    │   +── ...
    │   +── ring_front_left
    │    │   +── image1.jpg
    │    │   +── ...
````
## Step 3: Project Colmap points to argoverse
1. Install the [argoverse API](https://github.com/argoai/argoverse-api)
2. Modify the items below in projection.py. You will also need to include the original data in the sample folder.
- flags that determine which ring cameras to use:
````
use_fc, use_fl, use_sl, use_rl, use_rr, use_sr, use_fr = [1, 1, 0, 0, 0, 0, 0]
````
- path to the argoverse dataset
````
tracking_dataset_dir = '~/argoverse-tracking/sample/'
````
- log_index determines which log in the argoverse dataset directory to use
````
log_index = 0
````
- colmap_output_dir points to the results from COLMAP
````
colmap_output_dir = '~/argoverse-tracking/sample/output/'
````
- detection results directory
````
detection_output_dir = '~/argoverse-tracking/sample/detector_output/'
````
- where to save computed traffic sign locations
````
location_output_dir = '~/argoverse-tracking/sample/projection_output/'
````
3. cd into the 3D projection folder and run project.py
````
python projection.py
````
## Docker
We prepare a development-ready docker image at [docker hub](https://hub.docker.com/r/kuoyichun1102/colmap_detectron).
Each time pushing to `prod` branch will trigger a build in the docker hub, so one should work on `prod` branch carefully.