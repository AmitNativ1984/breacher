import time
import os
import cv2
import argparse
import numpy as np
import logging
FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger("depth_breacher")

import threading
from distutils.util import strtobool

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped, Vector3Stamped
from image_geometry import PinholeCameraModel
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import tf

bridge = CvBridge()

breachPointPublisher = rospy.Publisher("/breacher/breachPoint", PointStamped, queue_size=10)
breachImagePublisher = rospy.Publisher("/breacher/breachImage", Image, queue_size=10)
colorImageRawPublisher = rospy.Publisher("/breacher/breachImageColor", Image, queue_size=1)
vec3DPublisher = rospy.Publisher("/breacher/breachPoint3D", PointStamped, queue_size=10)
breach3DOpticalFramePublisher = rospy.Publisher("/breacher/breachPoint3DCamOpt", PointStamped, queue_size=10)
breachPointWorldPublisher = rospy.Publisher("/breacher/breachPointWorld", PointStamped, queue_size=10)

class DepthBreacher(object):
    
    def __init__(self):
        print('DepthBreacher instance created')
        self.noise= 500. #[mm] depth noise
        self.detections = None
        self.distance2window = None
        self.wallThickness = 0.
        self.breach_zone = None
        self.pointingFinger = None
        self.mode = 'pointingFinger' # can be opne of: 'area';'pointingFinger'
        self.depthCamModel = PinholeCameraModel()
        self.nonValidMarginRatio = 0.1    # ratio taken from image width and height. the pixel of world point must be in only in valid area
    
    def getDepthCameraModel(self, CameraInfo_msg):
        self.depthCamInfo_Msg = CameraInfo_msg
        self.depthCamModel.fromCameraInfo(CameraInfo_msg)
        self.depthCamWidth = CameraInfo_msg.width
        self.depthCamHeight = CameraInfo_msg.height

        self. minValidRow = int(self.depthCamHeight * self.nonValidMarginRatio)
        self. maxValidRow = int(self.depthCamHeight * (1 - self.nonValidMarginRatio))
        self. minValidCol = int(self.depthCamWidth * self.nonValidMarginRatio)
        self. maxValidCol = int(self.depthCamWidth * (1 - self.nonValidMarginRatio))
    
    @staticmethod
    def tfWorld2BaseLink(ImageMsg, worldPoint3d, transformTimeStamp):
        listner.waitForTransform("/base_link", "/map", transformTimeStamp, rospy.Duration(1.0))
        worldPoint3d.header.stamp = transformTimeStamp
        base_link_point3d = listner.transformPoint("/base_link", worldPoint3d)
        return base_link_point3d
    
    @staticmethod
    def tfFrameWorld2PositionXYZ(ImageMsg, transformTimeStamp):
        listner.waitForTransform("/map", "/base_link", ImageMsg.header.stamp, rospy.Duration(1.0))
        currCamTranslation, currCamRotation = listner.lookupTransform("/map", "/base_link", ImageMsg.header.stamp)

        return currCamTranslation, currCamRotation
    
    @staticmethod
    def tfPointXYZWorld_to_pointXYZCamOpt(ImageMsg, worldXYZ, transformTimeStamp):
        listner.waitForTransform(ImageMsg.header.frame_id, "/map", transformTimeStamp, rospy.Duration(1.0))
        worldXYZ.header.stamp = transformTimeStamp
        point_in_cam_optical_frame = listner.transformPoint(ImageMsg.header.frame_id, worldXYZ)
        return point_in_cam_optical_frame 

    @staticmethod
    def tfPointXYZCamOp_to_pointXYZWorld(camOptXYZ, transformTimeStamp):
        listner.waitForTransform("/map", camOptXYZ.header.frame_id, transformTimeStamp, rospy.Duration(1.0))
        camOptXYZ.header.stamp = transformTimeStamp
        point_world_coordinates = listner.transformPoint("/map", camOptXYZ)
        return point_world_coordinates 
    
    @staticmethod
    def subtructVectors3D(pointXYZ, cameraXYZ):
        dx = pointXYZ.point.x - cameraXYZ[0]
        dy = pointXYZ.point.y - cameraXYZ[1]
        dz = pointXYZ.point.z - cameraXYZ[2]
    
        return dx, dy, dz
    @staticmethod
    def getPossibleBreachZones(depth, distance2breach, frame_margin=15):
        breachZones = np.ones_like(depth)
        
        breachZones[depth < distance2breach] = 0
        breachZones[depth == 0] = 1
        breachZones = breachZones.astype(np.uint8)
        kernel = np.ones((25,25), np.uint8)
        breachZones = cv2.morphologyEx(breachZones, cv2.MORPH_OPEN, kernel)
        
        # kernel = np.ones((5, 5), np.uint8)
        # breachZones = cv2.morphologyEx(breachZones, cv2.MORPH_CLOSE, kernel)

        breachZones[:frame_margin, :] = 0
        breachZones[-frame_margin:, :] = 0
        breachZones[:, :frame_margin] = 0
        breachZones[:, -frame_margin:] = 0
        
        return breachZones

    def getWorldPointXYZCallback(self, breachPointWorldXYZ):
               
        self.breachPointWorldXYZ = breachPointWorldXYZ


    @staticmethod
    def getBreachAreaID(breach_zones, cc_img, pixel, n_labels, stats, centroids, mode='pointingFinger'):
        
        dist = np.inf
        label_id = -1
        for currlabel, centroid in enumerate(centroids):
            if breach_zones[cc_img == currlabel][0] > 0:                                    
                
                R = np.sqrt((centroid[0] - pixel[0]) ** 2 +
                            (centroid[1] - pixel[1]) ** 2)
                if R < dist:
                    dist = R
                    label_id = currlabel
          
        return label_id
    
    def getColorImageRaw(self, colorImageRaw):
        self.colorImageRaw = bridge.imgmsg_to_cv2(colorImageRaw, desired_encoding='passthrough')
        self.colorImageRaw = self.colorImageRaw.astype(np.uint8)

    def depthImageCallback(self, depthImageMsg):
        """
        bbox: hatch bbox give in (xmin, ymin, xmax, ymax)
        noise: disantce measurement noise [mm]
        disance2hatch: distance breach plane given in [mm] (assuming perpedicular position to plane)
        """

        self.depthImageMsg = depthImageMsg
        depth = bridge.imgmsg_to_cv2(depthImageMsg, desired_encoding='passthrough')       
        # depth = cv2.GaussianBlur(depth, (11, 11), self.noise)
        
        #todo: remove
        depthCamInfo_Msg = self.depthCamInfo_Msg
        depthCamTimeStamp = depthImageMsg.header.stamp

        # get current breach point in world coordinates (x,y,z)
        breachPointWorldXYZ = self.breachPointWorldXYZ
        
        # get vector from current position to breach point in base_link coordinate system
        cam2BreachPoint3DPoint = self.tfWorld2BaseLink(depthImageMsg, breachPointWorldXYZ, transformTimeStamp=depthCamTimeStamp)
        
        # get distance to target by subtracting target world XYZ from current robot position
        # cam2BreachPoint3DPoint = self.subtructVectors3D(breachPointWorldXYZ, depthCamWorldXYZ)
        distance2target = 1E3 * np.linalg.norm(np.array([cam2BreachPoint3DPoint.point.x, cam2BreachPoint3DPoint.point.y, cam2BreachPoint3DPoint.point.z]))
        distance2breach = distance2target + self.noise + self.wallThickness

        # convert breach point in world coordinates to camera optical coordinates (x, y, z)
        # breachPointCameraOpticalXYZ = self.point3DBaseLink2CameraPixels(depthCamInfo_Msg, breachPointWorldXYZ)
        breachPointCameraOpticalXYZ = self.tfPointXYZWorld_to_pointXYZCamOpt(depthImageMsg, breachPointWorldXYZ, transformTimeStamp=depthCamTimeStamp)
        
        # get breach point in camera pixels:
        u, v = self.depthCamModel.project3dToPixel((breachPointCameraOpticalXYZ.point.x, breachPointCameraOpticalXYZ.point.y, breachPointCameraOpticalXYZ.point.z))
        if u <= self.minValidRow or u >= self.maxValidRow or \
            v <= self.minValidCol or v >= self.maxValidRow:
            print("projected world breach point to pixels {}, {} non valid in depth cam".format(u,v))
            # return -1


        breach_zones = self.getPossibleBreachZones(depth, distance2breach)

        
        # select breach zone which is closest to breach point (u,v)
        
        n_labels, cc_img, stats, centroids = cv2.connectedComponentsWithStats(breach_zones, connectivity=8)
        cc_img = np.array(cc_img).astype(np.uint8)
                
        label_id = self.getBreachAreaID(breach_zones, cc_img, (u, v), n_labels, stats, centroids)
        if label_id == -1:
            print("invalid connected component label id [-1]")
            return
        
        u_new = centroids[label_id, 0]
        v_new = centroids[label_id, 1]

        breach_zone = np.zeros_like(cc_img)
        breach_zone[cc_img == label_id] = 1
        # breach point in [u,v, distance]
        breachPoint = PointStamped()
        breachPoint.header.stamp = depthCamTimeStamp
        breachPoint.header.frame_id = 'd415_depth_frame'
        breachPoint.point.x = centroids[label_id, 0]
        breachPoint.point.y = centroids[label_id, 1]
        breachPoint.point.z = distance2breach

        # breach point in camera optical frame coordinates
        ray = np.array(self.depthCamModel.projectPixelTo3dRay((u_new,v_new)))
        ray /= np.linalg.norm(ray)
        breachPointCameraOpticalCoordinates = PointStamped()
        breachPointCameraOpticalCoordinates.header.stamp = depthCamTimeStamp
        breachPointCameraOpticalCoordinates.header.frame_id = depthImageMsg.header.frame_id
        breachPointCameraOpticalCoordinates.point.x = ray[0] * distance2target/1E3
        breachPointCameraOpticalCoordinates.point.y = ray[1] * distance2target/1E3
        breachPointCameraOpticalCoordinates.point.z = ray[2] * distance2target/1E3

        # transform breach point in camera optical frame to world XYZ
        breachPointWorld = self.tfPointXYZCamOp_to_pointXYZWorld(breachPointCameraOpticalCoordinates, depthCamTimeStamp)
        breachPointWorldPublisher.publish(breachPointWorld)

        # breach vector
        breachVec3D = cam2BreachPoint3DPoint
        vec3DPublisher.publish(breachVec3D)

        # breach point in camera optical frame
        breach3DOpticalFramePublisher.publish(breachPointCameraOpticalXYZ)
        
        # publish results
        breachZone_ = np.zeros((breach_zone.shape[0], breach_zone.shape[1], 3))
        breachZone_[...,0] = breach_zones
        breachZone_[...,1] = breach_zones
        breachZone_[...,2] = breach_zones
        breachZone_ *= breachZone_.max() * 255
        cv2.circle(breachZone_, (int(u_new), int(v_new)), 5, (255, 0, 0), -1)
        breachPointPublisher.publish(breachPoint)
        breachImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor((breachZone_).astype(np.uint8), cv2.COLOR_BGR2RGB), encoding="passthrough"))
        
        breach_zone = cv2.resize(breach_zone, (self.colorImageRaw.shape[1], self.colorImageRaw.shape[0]), interpolation=cv2.INTER_NEAREST)
        colorImageRaw = self.colorImageRaw.copy()
        colorImageRaw[breach_zone == 1, 0] = 255
        cv2.circle(colorImageRaw, (8*int(u), 8*int(v)), 5, (0, 255, 0), -1)

        
        newImageMsg = bridge.cv2_to_imgmsg(cv2.cvtColor(colorImageRaw, cv2.COLOR_BGR2RGB), encoding="passthrough")
        newImageMsg.header.stamp = rospy.Time.now()
        colorImageRawPublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(colorImageRaw, cv2.COLOR_BGR2RGB), encoding="passthrough"))
        
        print("breachPoint in world coordinates (x,y,z) = {} [m]".format(breachPointWorld.point.x, breachPointWorld.point.y, breachPointWorld.point.z/1E3))
    
        return


if __name__=="__main__":
    depth_breacher=DepthBreacher()
    
    rospy.init_node('hatch_detector', anonymous=True)
    rospy.Subscriber("/d415/aligned_depth_to_color/image_raw", Image, depth_breacher.depthImageCallback, queue_size=1, buff_size=2 ** 24)  
    rospy.Subscriber("/d415/color/image_raw", Image, depth_breacher.getColorImageRaw, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber("/seeker/windowCenterWorldPoint", PointStamped, depth_breacher.getWorldPointXYZCallback, queue_size=1)
    rospy.Subscriber("/d415/aligned_depth_to_color/camera_info", CameraInfo, depth_breacher.getDepthCameraModel, queue_size=1)
    listner = tf.TransformListener()

    rospy.spin()
