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

## Pipeline
1. Run traffic sign detection on raw images `./sign_detector`
2. Run Colmap SfM on raw images `colmap_script.txt`
3. Project Colmap points to argoverse map `3D_projection`

## Docker
We prepare a development-ready docker image at [docker hub](https://hub.docker.com/r/kuoyichun1102/colmap_detectron).
Each time pushing to `prod` branch will trigger a build in the docker hub, so one should work on `prod` branch carefully.