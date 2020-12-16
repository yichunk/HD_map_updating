Three new scripts are updated:
        new_projection.ipynb: updated script for uptading HD map for new traffic signs, with some visualization utilities
        removed_sign_detection.ipynb: scrpt for updating HD map for removed traffic signs
        generate_point_clouds.ipynb: utility script for generating 3D point clouds from LIDAR scans, can be used for manul labelling ground truth traffic sign positions



To run the scripts, change several paths at the beginning of the codes:
	$tracking_dataset_dir
	$colmap_output_dir
	$detection_output_dir

Those directories should be structured in the followin way:

$tracking_dataset_dir
    -c6911883-1843-3727-8eaa-41dc8cda8993
        -ring_front_center
        -ring_front_left
        -vehicle_calibration_info.json
        ....

$colmap_output_dir
    -c6911883-1843-3727-8eaa-41dc8cda8993
        -cameras.txt
        -images.txt
        -points3D.txt

$detection_output_dir
    -c6911883-1843-3727-8eaa-41dc8cda8993.npz


Please download the detection outputs at the following link:
    https://drive.google.com/drive/folders/1t6gmON7Qwk1jKJrp18cMOG6ID9lWcoHP?usp=sharing

Please download the colmap outputs at the following link:
    https://drive.google.com/drive/folders/133zP9p-r4ZraxxorP_XetZnjSCUonWKs?usp=sharing
    
