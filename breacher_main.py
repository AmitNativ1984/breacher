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
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

bridge = CvBridge()

breachPointPublisher = rospy.Publisher("/seeker/breachPoint", PointStamped, queue_size=10)
breachImagePublisher = rospy.Publisher("seeker/breachImage", Image, queue_size=10)
detectionImagePublisher = rospy.Publisher("seeker/breachImageColor", Image, queue_size=10)

class DepthBreacher(object):
    
    def __init__(self):
        print('DepthBreacher instance created')
        self.distanceFromSurface=None
        self.noise=100 #[mm] depth noise
        self.detections = None
        self.distanceFromSurface = 3000
        self.wallThickness = 500
        self.depth = None
        self.breach_zone = None
        self.pointingFinger = None

        self.mode = 'area'#'pointingFinger'

    def getDistanceCallback(self, distance):
        self.distanceFromSurface = distance

    def getPointFingerCallback(self, pointingFinger):
        self.pointingFinger = pointingFinger.point
        self.pointingFinger.x = int(self.pointingFinger.x)
        self.pointingFinger.y = int(self.pointingFinger.y)
    
    def distance2breach(self):
        return self.distanceFromSurface + self.noise + self.wallThickness
    
    def getDepthROI(self):
        depthROI = np.zeros_like(self.depth)
        depthROI[self.depth > self.distance2breach()] = 1
        depthROI = depthROI.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        depthROI = cv2.morphologyEx(depthROI, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((9,9), np.uint8)
        depthROI = cv2.morphologyEx(depthROI, cv2.MORPH_CLOSE, kernel)
        return depthROI

    
    def getBreachAreaID(self, depthROI, cc_img, n_labels, stats):
        if self.mode == 'pointingFinger':
            if self.pointingFinger is None:
                print('no pointingFinger recived')
                return -1
   
            if depthROI[self.pointingFinger.y, self.pointingFinger.x] == 0:
                print('pointing fingure not pointing to area with depth valid for breach')
                return -1

            else:
                label_id = cc_img[self.pointingFinger.y, self.pointingFinger.x]
            
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
    
    def getDetectionImage(self, detectionImage):
        self.detectionImage = bridge.imgmsg_to_cv2(detectionImage, desired_encoding='passthrough')
        self.detectionImage = self.detectionImage.astype(np.uint8)

    def depthImageCallback(self, depthImage):
        """
        bbox: hatch bbox give in (xmin, ymin, xmax, ymax)
        noise: disantce measurement noise [mm]
        disance2hatch: distance breach plane given in [mm] (assuming perpedicular position to plane)
        """

        timeStamp = depthImage.header.stamp
        self.depth = bridge.imgmsg_to_cv2(depthImage, desired_encoding='passthrough')       

        depthROI = self.getDepthROI()
        n_labels, cc_img, stats, centroids = cv2.connectedComponentsWithStats(depthROI, connectivity=8)
        cc_img = np.array(cc_img).astype(np.uint8)
                
        label_id = self.getBreachAreaID(depthROI, cc_img, n_labels, stats)
        if label_id == -1:
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
        breachImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(breach_zone * 255, cv2.COLOR_BGR2RGB), encoding="passthrough"))
        
        try:
            self.detectionImage[breach_zone == 1, 0] = 255
            cv2.circle(self.detectionImage, (int(breachPoint.point.x), int(breachPoint.point.y)), 5, (0, 255, 0), -1)
        except Exception:
            pass
        
        detectionImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(self.detectionImage, cv2.COLOR_BGR2RGB), encoding="passthrough"))
        print("breachPoint (x,y,z)=({}, {}, {})".format(breachPoint.point.x, breachPoint.point.y, breachPoint.point.z))
    
        return


if __name__=="__main__":
    depth_breacher=DepthBreacher()
    
    rospy.init_node('hatch_detector', anonymous=True)
    rospy.Subscriber("/d415/aligned_depth_to_color/image_raw", Image, depth_breacher.depthImageCallback, queue_size=1, buff_size=2 ** 24)  
    rospy.Subscriber("/pointingfinger/TargetPoint", PointStamped, depth_breacher.getPointFingerCallback, queue_size = 10)
    rospy.Subscriber("/d415/color/image_raw", Image, depth_breacher.getDetectionImage, queue_size=1, buff_size=2 ** 24)
    rospy.spin()
