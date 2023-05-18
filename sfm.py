import os
import numpy as np
import cv2
from sklearn.preprocessing import scale
from cv2.xfeatures2d import matchGMS
import glob
import argparse
import time
import datetime

import pytheia as pt

min_num_inlier_matches = 100

class SFM:
    def __init__(self):
        self.video_frames = {}
        self.correspondences = []
        self.min_inliers = 45
        self.scale_factor = 1.0
        self.focal_length = 790.0
        self.principal_point = [325.0, 196.0]
        self.features = {}
        self.recon_obj = pt.sfm.Reconstruction()
        self.view_graph_obj = pt.sfm.ViewGraph()
        self.builder = pt.sfm.TrackBuilder(3, 20)
        self.global_estimator_options = pt.sfm.ReconstructionEstimatorOptions()
        self.incremental_estimator_options = pt.sfm.ReconstructionEstimatorOptions()
        self.reconstruction = []

    def recordAndStoreFrames(self):
        """
        Records from webcam and stores frames in an array
        """
        count = 1
        while(True):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            frame_id = "video_frame_" + str(count)
            self.video_frames[frame_id] = frame
            if cv2.waitKey(1) & 0xFF == ord('a'):
                break
        cap.release()
        return
    
    def storeFramesFromVideo(self, path_to_vid):
        """
        Records from a video file and stores in an array
        """
        cap = cv2.VideoCapture(path_to_vid)
        ret, frame = cap.read()
        count = 1
        while (ret):
            frame_id = "video_frame_" + str(count)
            self.video_frames[frame_id] = frame
            ret, frame = cap.read()
            count += 1
        return
    
    def storeFramesFromDataset(self, path_to_dataset):
        for idx, filename in enumerate(os.listdir(path_to_dataset)):
            frame = cv2.imread(os.path.join(path_to_dataset, filename))
            if frame is not None:
                frame_id = "video_frame_" + str(idx + 1)
                self.video_frames[frame_id] = frame
        return
    
    def featureExtraction(self, img_key):
        """
        Function to extract features from an image using AKAZE Feature extractor
        """
        print("Extracting Features for image: ", img_key, "\n")
        vw_id = self.recon_obj.ViewIdFromName(img_key)
        camForImg = self.recon_obj.View(vw_id).Camera()
        img = self.video_frames[img_key]
        if(self.scale_factor != 1.0):
            img = cv2.resize(img, (-1,-1),fx=self.scale_factor, fy=self.scale_factor)
        feature = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_KAZE, 0, 3, 0.0001, 4, 4)       
        keypoints, descriptors = feature.detectAndCompute(img, None)
        return vw_id, img, keypoints, descriptors
    
    def featureCorrespondences(self,matches, img1_ft, img2_ft):
        """
        To find correspondences between matched points
        """
        correspondences = []
        for match in matches:
            img1_pt = np.array(img1_ft[match.queryIdx].pt)
            img2_pt = np.array(img2_ft[match.trainIdx].pt)
            correspondences.append(pt.matching.FeatureCorrespondence(pt.sfm.Feature(img1_pt), pt.sfm.Feature(img2_pt)))
        return correspondences
    
    def imageMatcher(self, frame1, frame2):
        gms_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = gms_matcher.match(self.features[frame1]["descriptors"], self.features[frame2]["descriptors"])
        best_matches = matchGMS(self.features[frame1]["img_shape"], self.features[frame2]["img_shape"],
        self.features[frame1]["keypoints"], self.features[frame2]["keypoints"], matches)
        print("Total Number of matches found: ",len(best_matches))

        if len(best_matches) < self.min_inliers:
            print('The number of inlier matches are too low')
            return False, None
        
        # Creating correspondences
        correspondences = self.featureCorrespondences(
            best_matches, self.features[frame1]["keypoints"], self.features[frame2]["keypoints"])

        # Initializing two view estimation (Theia allows initialization of SFM pipeline and setting up initial parameters)
        two_view_init = pt.sfm.EstimateTwoViewInfoOptions()
        two_view_init.max_sampson_error_pixels = 1.0
        two_view_init.max_ransac_iterations = 100

        # Choosing Prosac as the estimation algorithm
        two_view_init.ransac_type = pt.sfm.RansacType(1)

        # estimating camera pose for both frames
        prior_cam_1 = self.recon_obj.View(frame1).Camera().CameraIntrinsicsPriorFromIntrinsics()
        prior_cam_2 = self.recon_obj.View(frame2).Camera().CameraIntrinsicsPriorFromIntrinsics()
        # The below function generates relative position information between two images and the number of inliers verified
        estimated, twoview_info, verified_indices = pt.sfm.EstimateTwoViewInfo(two_view_init, prior_cam_1, prior_cam_2, correspondences)

        if (estimated):
            if(len(verified_indices) < self.min_inliers):
                print("After estimation and geometric verification, the number of inliers were reduced below threshold")
                return False,None
            else:
                verified_matches = []
                for i in range(0, len(verified_indices)):
                    verified_matches.append(best_matches[verified_indices[i]])
                verified_correspondences = self.featureCorrespondences(verified_matches, self.features[frame1]["keypoints"], self.features[frame2]["keypoints"])
                for i in range(len(verified_matches)):
                    self.builder.AddFeatureCorrespondence(frame1, verified_correspondences[i].feature1, 
                                                    frame2, verified_correspondences[i].feature2)
                return True, twoview_info
        else:
            print("The estimation was not successful")
            return False, None
    
    def camera_intrinsics(self):
        """
        Function to define camera intrinsic matrix in Theia
        """
        cam_int = pt.sfm.CameraIntrinsicsPrior()
        cam_int.focal_length.value = [float(self.focal_length)* self.scale_factor]
        cam_int.aspect_ratio.value = [1.0]
        cam_int.principal_point.value = [float(self.principal_point[0]) * self.scale_factor, float(self.principal_point[1]) * self.scale_factor]
        cam_int.radial_distortion.value = [0, 0, 0, 0]
        cam_int.tangential_distortion.value = [0, 0]
        cam_int.skew.value = [0]
        cam_int.image_width = int(640 * self.scale_factor)
        cam_int.image_height = int(480 * self.scale_factor)
        # Assuming the camera intrinsic matrix to follow Pinhole model
        cam_int.camera_intrinsics_model_type = "PINHOLE"
        return cam_int
    
    def setFeatureAndViewId(self):
        print("Setting Features and Views....\n")
        cam_int = self.camera_intrinsics()
        for idx, key in enumerate(self.video_frames):
            vw_id = self.recon_obj.AddView(key, 0, idx)
            view = self.recon_obj.MutableView(vw_id)
            view.SetCameraIntrinsicsPrior(cam_int)
            view_id, frame, keypoints, descriptors = self.featureExtraction(key)
            self.features[view_id] = {"keypoints": keypoints, "descriptors": descriptors,"img":frame, "img_shape":frame.shape[:2][::-1]}
        return
    
    def generateViewGraph(self):
        ids = self.recon_obj.ViewIds()
        for i in range(0, len(ids)):
            for j in range(i + 1, len(ids)):
                first_id = ids[i]
                second_id = ids[j]
                estimated, two_view_info = self.imageMatcher(first_id, second_id)
                if(estimated):
                    self.view_graph_obj.AddEdge(first_id, second_id, two_view_info)
                    print("Edge between {} and {} added".format(first_id, second_id))
                else:
                    print("No Match was found!")
        # Once all data is set, we can set cameras for reconstruction
        pt.sfm.SetCameraIntrinsicsFromPriors(self.recon_obj)
        print("View Graph generation is done!")
        return
    
    def configureGlobalEstimator(self):
        # http://www.theia-sfm.org/sfm.html#the-reconstruction-estimator
        self.global_estimator_options.num_threads = 10
        self.global_estimator_options.rotation_filtering_max_difference_degrees = 5.0
        self.global_estimator_options.bundle_adjustment_robust_loss_width = 1.0
        self.global_estimator_options.bundle_adjustment_loss_function_type = pt.sfm.LossFunctionType(0)
        self.global_estimator_options.subsample_tracks_for_bundle_adjustment = False
        self.global_estimator_options.filter_relative_translations_with_1dsfm = True
        self.global_estimator_options.intrinsics_to_optimize = pt.sfm.OptimizeIntrinsicsType.FOCAL_LENGTH
        self.global_estimator_options.min_triangulation_angle_degrees = 3.0
        self.global_estimator_options.triangulation_method = pt.sfm.TriangulationMethodType(2)
        self.global_estimator_options.global_position_estimator_type = pt.sfm.GlobalPositionEstimatorType.LEAST_UNSQUARED_DEVIATION
        self.global_estimator_options.global_rotation_estimator_type = pt.sfm.GlobalRotationEstimatorType.ROBUST_L1L2
        estimator = pt.sfm.GlobalReconstructionEstimator(self.global_estimator_options)
        return estimator

    def configureIncrementalEstimator(self):
        self.incremental_estimator_options.num_threads = 10
        self.incremental_estimator_options.rotation_filtering_max_difference_degrees = 5.0
        self.incremental_estimator_options.bundle_adjustment_robust_loss_width = 1.0
        self.incremental_estimator_options.bundle_adjustment_loss_function_type = pt.sfm.LossFunctionType(0)
        self.incremental_estimator_options.subsample_tracks_for_bundle_adjustment = False
        self.incremental_estimator_options.filter_relative_translations_with_1dsfm = True
        self.incremental_estimator_options.intrinsics_to_optimize = pt.sfm.OptimizeIntrinsicsType.FOCAL_LENGTH
        self.incremental_estimator_options.min_triangulation_angle_degrees = 3.0
        self.incremental_estimator_options.triangulation_method = pt.sfm.TriangulationMethodType(2)
        estimator = pt.sfm.GlobalReconstructionEstimator(self.incremental_estimator_options)
        return estimator
        
    def createPointCloud(self):
        pt.io.WritePlyFile("./reconstruction.ply", self.recon_obj, [0 ,255, 0], 2)
        return
    
    def sfm_global_pipeline(self):
        # if there are no video_frames
        if(not self.video_frames):
           print("There are no video frames!")
           return
        self.setFeatureAndViewId()
        self.generateViewGraph()
        self.builder.BuildTracks(self.recon_obj)
        estimator = self.configureGlobalEstimator()
        self.reconstruction = estimator.Estimate(self.view_graph_obj, self.recon_obj)
        return
    
    def sfm_incremental_pipeline(self):
        # if there are no video_frames
        if(not self.video_frames):
           print("There are no video frames!")
           return
        self.setFeatureAndViewId()
        self.generateViewGraph()
        self.builder.BuildTracks(self.recon_obj)
        estimator = self.configureIncrementalEstimator()
        self.reconstruction = estimator.Estimate(self.view_graph_obj, self.recon_obj)
        return
    
if __name__== "__main__":
    user_input = int(input("Enter choice of SFM Pipeline: \n 1 - Global \n 2 - Incremental \n"))
    sfm_obj = SFM()
    img_path = "./frames/"
    sfm_obj.storeFramesFromDataset(img_path)
    now = datetime.datetime.now()
    # Current Time
    lines = ["SFM Start Time(ransac):", str(now.strftime("%Y-%m-%d %H:%M:%S"))]
    with open("Time.txt", "a") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    if(user_input == 1):
        sfm_obj.sfm_global_pipeline()
    else:
        sfm_obj.sfm_incremental_pipeline()
    sfm_obj.createPointCloud()
    now = datetime.datetime.now()
    lines = ["SFM End Time(ransac):", str(now.strftime("%Y-%m-%d %H:%M:%S"))]
    with open("Time.txt", "a") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
