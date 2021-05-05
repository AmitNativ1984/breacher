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

breachPointPublisher = rospy.Publisher("/seeker/breachPoint", PointStamped, queue_size=10)
breachImagePublisher = rospy.Publisher("seeker/breachImage", Image, queue_size=10)
colorImageRawPublisher = rospy.Publisher("seeker/breachImageColor", Image, queue_size=10)

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
    
    def getDepthCameraModel(self, CameraInfo_msg):
        self.depthCamInfo_Msg = CameraInfo_msg
        self.depthCamModel.fromCameraInfo(CameraInfo_msg)
        self.depthCamWidth = CameraInfo_msg.width
        self.depthCamHeight = CameraInfo_msg.height
    
    @staticmethod
    def getFrameWorldPositionXYZ(ImageMsg):
        listner.waitForTransform("/map", "/base_link", ImageMsg.header.stamp, rospy.Duration(1.0))
        currCamTranslation, currCamRotation = listner.lookupTransform("/map", "/base_link", ImageMsg.header.stamp)

        return currCamTranslation, currCamRotation
    
    @staticmethod
    def pointXYZWorld_to_pointXYZCamOpt(ImageMsg, worldXYZ):
        listner.waitForTransform(ImageMsg.header.frame_id, "/map", ImageMsg.header.stamp, rospy.Duration(1.0))
        point_in_cam_optical_frame = listner.transformPoint(ImageMsg.header.frame_id, worldXYZ)
        return point_in_cam_optical_frame 
    
    @staticmethod
    def calcDistanceFromCamera(pointXYZ, cameraXYZ):
        dx = pointXYZ.point.x - cameraXYZ[0]
        dy = pointXYZ.point.y - cameraXYZ[1]
        dz = pointXYZ.point.z - cameraXYZ[2]
    
        return np.sqrt(dx**2 + dy**2 + dz**2)
    @staticmethod
    def getPossibleBreachZones(depth, distance2breach, frame_margin=20):
        breachZones = np.ones_like(depth)
        
        breachZones[depth < distance2breach] = 0
        breachZones[depth == 0] = 1
        breachZones = breachZones.astype(np.uint8)
        # kernel = np.ones((9,9), np.uint8)
        # depthROI = cv2.morphologyEx(depthROI, cv2.MORPH_OPEN, kernel)
        
        # kernel = np.ones((5, 5), np.uint8)
        # depthROI = cv2.morphologyEx(depthROI, cv2.MORPH_CLOSE, kernel)

        breachZones[:frame_margin, :] = 0
        breachZones[-frame_margin:, :] = 0
        breachZones[:, :frame_margin] = 0
        breachZones[:, -frame_margin:] = 0
        
        return breachZones

    def getWorldPointXYZCallback(self, breachPointWorldXYZ):
               
        self.breachPointWorldXYZ = breachPointWorldXYZ

    def getPlaneEquation(self):
        """
        plane eq. given its normal n=(A,B,C), passing through point on plance (x0,y0,z0):
        P = A(x-x0) + B(y-y0) +C(z-z0) = 0
        in our case the point on the place is the opposite normal vector coordinates, since point of view is origin

        """

        A, B, C = -1 * self.surfaceNormalVec.vector # TODO: what should be the sign of the normal vector?
        D = -(A**2 + B**2 + C**2)
        self.surfacePlane = np.array([A, B, C, D])

    @staticmethod
    def getBreachAreaID(breach_zones, cc_img, pixel, n_labels, stats, centroids, mode='pointingFinger'):
        
        dist = np.inf
        label_id = -1
        for currlabel, centroid in enumerate(centroids):
            if breach_zones[cc_img == currlabel][0] > 0:                                    
                # listner.waitForTransform(self.depthImageMsg.header.frame, self.breachPointWorldXYZ.header.frame, self.depthImageMsg.header.stamp,
                #                         rospy.Duration(4.0))
                
                
                # breachPointInCurrDepthCamPixels = listner.transformPoint(self.depthImageMsg.header.frame , self.breachPointWorldXYZ)
                
                R = np.sqrt((centroid[0] - pixel[0]) ** 2 +
                            (centroid[1] - pixel[1]) ** 2)
                if R < dist:
                    dist = R
                    label_id = currlabel

            if label_id == -1:
                print("could not detect area with depth valid for breach according to area size")
           
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

        timeStamp = depthImageMsg.header.stamp
        self.depthImageMsg = depthImageMsg
        depth = bridge.imgmsg_to_cv2(depthImageMsg, desired_encoding='passthrough')       
        depth = cv2.GaussianBlur(depth, (11, 11), self.noise)
        
        depthCamInfo_Msg = self.depthCamInfo_Msg

        # get the current camera poistion in world coordinates (x,y,z)
        depthCamWorldXYZ, depthCamWorldRotation = self.getFrameWorldPositionXYZ(depthCamInfo_Msg)

        # get current breach point in world coordinates (x,y,z)
        breachPointWorldXYZ = self.breachPointWorldXYZ
        
        # get distance to target by subtracting target world XYZ from current robot position
        distance2target = self.calcDistanceFromCamera(breachPointWorldXYZ, depthCamWorldXYZ)
        distance2breach = distance2target + self.noise + self.wallThickness

        # convert breach point in world coordinates to camera optical coordinates (x, y, z)
        breachPointCameraOpticalXYZ = self.pointXYZWorld_to_pointXYZCamOpt(depthCamInfo_Msg, breachPointWorldXYZ)
        
        breach_zones = self.getPossibleBreachZones(depth, distance2breach)

        # get breach point in camera pixels:
        u, v = self.depthCamModel.project3dToPixel((breachPointCameraOpticalXYZ.point.x, breachPointCameraOpticalXYZ.point.y, breachPointCameraOpticalXYZ.point.z))
        
        # select breach zone which is closest to breach point (u,v)
        
        n_labels, cc_img, stats, centroids = cv2.connectedComponentsWithStats(breach_zones, connectivity=8)
        cc_img = np.array(cc_img).astype(np.uint8)
                
        label_id = self.getBreachAreaID(breach_zones, cc_img, (u, v), n_labels, stats, centroids)
        if label_id == -1:
            print("invalid connected component label id [-1]")
            return
        
        breach_zone = np.zeros_like(cc_img)
        breach_zone[cc_img == label_id] = 1
      
        # breach point in [u,v, distance]
        breachPoint = PointStamped()
        breachPoint.header.stamp = timeStamp
        breachPoint.header.frame_id = 'd415_aligned_depth_to_color'
        breachPoint.point.x = centroids[label_id, 0]
        breachPoint.point.y = centroids[label_id, 1]
        breachPoint.point.z = distance2breach

        # breach point in camera optical frame coordinates
        ray = np.array(self.depthCamModel.projectPixelTo3dRay((u,v)))
        ray /= np.linalg.norm(ray)
        
        breachPointCameraOpticalCoordinates = PointStamped()
        breachPointCameraOpticalCoordinates.header.stamp = timeStamp
        breachPointCameraOpticalCoordinates.header.frame_id = 'd415_aligned_depth_to_color'
        breachPointCameraOpticalCoordinates.point.x = ray[0] * distance2target
        breachPointCameraOpticalCoordinates.point.y = ray[1] * distance2target
        breachPointCameraOpticalCoordinates.point.z = ray[2] * distance2target

        # publish results
        breachPointPublisher.publish(breachPoint)
        breachImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor((breach_zones / breach_zones.max() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB), encoding="passthrough"))
        
        try:
            self.colorImageRaw[breach_zone == 1, 0] = 255
            cv2.circle(self.colorImageRaw, (int(breachPoint.point.x), int(breachPoint.point.y)), 5, (0, 255, 0), -1)
        except Exception:
            pass
        
        colorImageRawPublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(self.colorImageRaw, cv2.COLOR_BGR2RGB), encoding="passthrough"))
        print("breachPoint (px,py)=({}, {}), distance= {} [m]".format(breachPoint.point.x, breachPoint.point.y, breachPoint.point.z/1E3))
    
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
