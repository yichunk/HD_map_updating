import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class ColmapProjector:
    """Project traffic sign 2d position to colmap 3d position
    """
    
    def __init__(self, intrinsics, ref_img, ref_pose, ref_bbox, src_img, src_pose):
        """
        args:
            intrinsics: camera intrinsics. 3x3 numpy array
            ref_img: reference image. src_img is relative to ref_img.
                     value range in [0,1]
            ref_pose: camera pose of reference image obtained from Colmap
                      3x4 numpy array [R|t]
            ref_bbox: bounding box of interested region. 1x4 numpy array,
                      [x1, y1, x2, y2] from top-left to bottom-right
            src_img: source image
            src_pose: camera pose of source image obtained from Colmap.
                      3x4 numpy array [R|t]
        """
        super().__init__()
        self.intrinsics = intrinsics
        self.ref_img = ref_img
        self.ref_pose = ref_pose
        self.ref_bbox = ref_bbox
        self.src_img = src_img
        self.src_pose = src_pose

        # use large number of orb feature points to ensure
        # there are some points inside bbox (a bad idea)
        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _is_inside(self, kp, bbox):
        """determine if a cv2 keypoint is inside a bbox
        """
        x, y = kp.pt[0], kp.pt[1]
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
        
    def extract_features(self):
        """extract features from ref_img and imgs
        """
        kps, des = self.orb.detectAndCompute(self.ref_img, None)
        kp_ref, des_ref = [], []
        for kp, de in zip(kps, des):
            if self._is_inside(kp, self.ref_bbox):
                kp_ref.append(kp)
                des_ref.append(de)

        kp_src, des_src = self.orb.detectAndCompute(self.src_img, None)

        self.kp_ref = kp_ref
        self.des_ref = np.array(des_ref)

        self.kp_src = kp_src
        self.des_src = np.array(des_src)

    def match_features(self):
        """match features between ref image and src image
        """
        matches = self.matcher.match(self.des_ref,self.des_src)
        matches = sorted(matches, key = lambda x:x.distance)
        self.matches = matches


    def project(self):
        """project 2d traffic sign feature points to 3d colmap point
        Returns:
            3d traffic sign colmap point 
        """

        ref_proj = self.intrinsics @ self.ref_pose
        src_proj = self.intrinsics @ self.src_pose

        ref_points2d = []
        src_points2d = []

        # here heuristically use top 10 matches as traffic sign representative
        for match in self.matches[:10]:
            ref_points2d.append([self.kp_ref[match.queryIdx].pt[0], self.kp_ref[match.queryIdx].pt[1]])
            src_points2d.append([self.kp_src[match.trainIdx].pt[0], self.kp_src[match.trainIdx].pt[1]])
        ref_points2d = np.array(ref_points2d).T
        src_points2d = np.array(src_points2d).T
        points3d = cv2.triangulatePoints(ref_proj, src_proj, ref_points2d, src_points2d)
        points3d /= points3d[3,:]

        return np.mean(points3d, axis=1)[:3]

if __name__ == "__main__":
    ref_img = cv2.imread("data/ref_img.jpg")
    src_img = cv2.imread("data/src_img.jpg")

    ref_R = Rotation.from_quat([-0.00507002, 0.00227284, -0.00148047, 0.999983]).as_dcm().astype(np.float32)
    ref_t = np.array([0.111909, 0.00141431, 0.781914], dtype=np.float32).reshape((3,1))
    src_R = Rotation.from_quat([-0.00493567, 0.00275127, -7.77088e-05, 0.999984]).as_dcm().astype(np.float32)
    src_t = np.array([0.0901664, -0.00513701, -1.16926], dtype=np.float32).reshape((3,1))
    ref_pose = np.hstack([ref_R, ref_t])
    src_pose = np.hstack([src_R, src_t])

    ref_bbox = np.array([1492, 268, 1633, 426])

    intrinsics = np.array([
        [1406.620, 0, 960],
        [0, 1406.620, 600],
        [0, 0, 1]
    ], dtype=np.float32)

    cv2.rectangle(ref_img, tuple(ref_bbox[:2]), tuple(ref_bbox[2:]), (0,204,0), 2)
    cv2.imshow("bbox", ref_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projector = ColmapProjector(intrinsics, ref_img, ref_pose, ref_bbox,src_img, src_pose)
    projector.extract_features()
    projector.match_features()
    points3d = projector.project()
    print("Traffic sign at {} in Colmap".format(points3d))