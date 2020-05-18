1.Speficy directories at the beginning of projection.py, the file structure is shown below:

    tracking_dataset_dir: directory to argoverse dataset 
    $tracking_dataset_dir
         -(log1)8a15674a-ae5c-38e2-bc4b-f4156d384072
              -ring_front_center
              -ring_front_left
              -...
         -(log2)c6911883-1843-3727-8eaa-41dc8cda8993
         -...

    tracking_dataset_dir: directory to argoverse dataset 
    $tracking_dataset_dir
         -(log1)8a15674a-ae5c-38e2-bc4b-f4156d384072
              -ring_front_center
              -ring_front_left
              -...
         -(log2)c6911883-1843-3727-8eaa-41dc8cda8993
              -...
         -...

    colmap_output_dir: directory to colmap outputs
    $colmap_output_dir
         -(log1)8a15674a-ae5c-38e2-bc4b-f4156d384072
              -cameras.txt
              -images.txt
              -points3D.txt
         -(log2)c6911883-1843-3727-8eaa-41dc8cda8993
              -...
         -... 

    detection_output_dir: directory to detection outputs
    $detection_output_dir
         -(log1)8a15674a-ae5c-38e2-bc4b-f4156d384072.npz
         -(log2)c6911883-1843-3727-8eaa-41dc8cda8993.npz
         -... 

    location_output_dir: output directory to save projected traffic sign locations, the output is saved as $location_output_dir/(log_id)_projected.npz


2.use_fc, use_fl, use_sl, use_rl, use_rr, use_sr, use_fr are flags that determine which cameras data to use.
They represent front_center, front_left, side_left, rear_left, rear_right, side_right, front_right respectivly. 
1 means use and 0 means not use


3.After speficying the above directories and flags, simply frun python3.6 projection.py
