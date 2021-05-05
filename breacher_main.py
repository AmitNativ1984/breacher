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
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped, Vector3Stamped
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
        self.depth = None
        self.breach_zone = None
        self.pointingFinger = None

        self.mode = 'pointingFinger' # can be opne of: 'area';'pointingFinger'
    
    def getCurrCamWorldXYZ(self):
        listner.waitForTransform("/map", "/base_link", self.depthImageMsg.header.stamp, rospy.Duration(4.0))
        currCamTranslation, currCamRotation = listner.lookupTransform("/map", "/base_link", self.depthImageMsg.header.stamp)

        return currCamTranslation, currCamRotation
    
    def getDepthROI(self):
        depthROI = np.ones_like(self.depth)
        
        # get current camera world XYZ positoin:
        depthCamCurrPosition, depthCamCurrRot = self.getCurrCamWorldXYZ()
        dx = self.breachPointWorldXYZ.point.x - depthCamCurrPosition[0]
        dy = self.breachPointWorldXYZ.point.y - depthCamCurrPosition[1]
        dz = self.breachPointWorldXYZ.point.z - depthCamCurrPosition[2]

        distance2target = np.sqrt(dx**2 + dy**2 + dz**2)

        distance2breach = distance2target + self.noise + self.wallThickness
        
        depthROI[self.depth < self.distance2breach] = 0
        depthROI[self.depth == 0] = 1
        depthROI = depthROI.astype(np.uint8)
        # kernel = np.ones((9,9), np.uint8)
        # depthROI = cv2.morphologyEx(depthROI, cv2.MORPH_OPEN, kernel)
        
        # kernel = np.ones((5, 5), np.uint8)
        # depthROI = cv2.morphologyEx(depthROI, cv2.MORPH_CLOSE, kernel)

        margin = 20
        depthROI[:margin, :] = 0
        depthROI[-margin:, :] = 0
        depthROI[:, :margin] = 0
        depthROI[:, -margin:] = 0
        
        return depthROI

    def getBreachPointWorldXYZ(self, breachPointWorldXYZ):
               
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

    def getBreachAreaID(self, depthROI, cc_img, n_labels, stats, centroids):
        if self.mode == 'pointingFinger':
            if self.pointingFinger is None:
                print('no pointingFinger recived')
                return -1
   
            dist = np.inf
            label_id = -1
            for currlabel, centroid in enumerate(centroids):
                if depthROI[cc_img == currlabel][0] > 0:                                    
                    listner.waitForTransform(self.depthImageMsg.header.frame, self.breachPointWorldXYZ.header.frame, self.depthImageMsg.header.stamp,
                                            rospy.Duration(4.0))
                    
                    
                    breachPointInCurrDepthCamPixels = listner.transformPoint(self.depthImageMsg.header.frame , self.breachPointWorldXYZ)
                    
                    R = np.sqrt((centroid[0] - self.pointingFinger.x) ** 2 +
                                (centroid[1] - self.pointingFinger.y) ** 2)
                    if R < dist:
                        dist = R
                        label_id = currlabel
            
        elif self.mode =='area':
            # breach area is assumed to be the largest
            label_area = -1 * np.ones((n_labels))
            for label, stat in enumerate(stats):
                if depthROI[cc_img == label][0] > 0:
                    label_area[label] = stat[-1]

            label_id = np.argmax(label_area)

            if label_id == -1:
                print("could not detect area with depth valid for breach according to area size")
           
        return label_id
    
    def getColorImageRaw(self, colorImageRaw):
        self.colorImageRaw = bridge.imgmsg_to_cv2(colorImageRaw, desired_encoding='passthrough')
        self.colorImageRaw = self.colorImageRaw.astype(np.uint8)

    def depthImageCallback(self, depthImage):
        """
        bbox: hatch bbox give in (xmin, ymin, xmax, ymax)
        noise: disantce measurement noise [mm]
        disance2hatch: distance breach plane given in [mm] (assuming perpedicular position to plane)
        """

        timeStamp = depthImage.header.stamp
        self.depthImageMsg = depthImage
        self.depth = bridge.imgmsg_to_cv2(depthImage, desired_encoding='passthrough')       
        self.depth = cv2.GaussianBlur(self.depth, (11, 11), self.noise)
        depthROI = self.getDepthROI()
        n_labels, cc_img, stats, centroids = cv2.connectedComponentsWithStats(depthROI, connectivity=8)
        cc_img = np.array(cc_img).astype(np.uint8)
                
        label_id = self.getBreachAreaID(depthROI, cc_img, n_labels, stats, centroids)
        if label_id == -1:
            print("invalid connected component label id [-1]")
            return
        
        breach_zone = np.zeros_like(cc_img)
        breach_zone[cc_img == label_id] = 1

        breach_point = centroids[label_id]
      
      
        breachPoint = PointStamped()
        breachPoint.header.stamp = timeStamp
        breachPoint.header.frame_id = 'd415_aligned_depth_to_color'
        breachPoint.point.x = centroids[label_id, 0]
        breachPoint.point.y = centroids[label_id, 1]
        breachPoint.point.z = self.distance2breach()

        
        # publish results
        breachPointPublisher.publish(breachPoint)
        breachImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor((depthROI / depthROI.max() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB), encoding="passthrough"))
        
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
    rospy.Subscriber("/seeker/windowCenterWorldPoint", PointStamped, depth_breacher.getBreachPointWorldXYZ, queue_size=1)

    listner = tf.TransformListener()

    rospy.spin()
