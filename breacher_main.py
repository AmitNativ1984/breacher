import time
import os
import cv2
import argparse
import numpy as np
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
        self.distance2hatch=None
        self.noise=100 #[mm] depth noise
        self.detections = None
        self.distance2hatch = 3000
        self.breachDepthUncertainty = 500
        
    def getDistanceCallback(self, distance):
        self.distance2hatch = distance
    
    def getDetectionsCallback(self, detections):
        self.detections = detections.detections

    def getDetectionImageCallback(self, img):
        self.detectionImage = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        self.detectionImage = self.detectionImage.astype(np.uint8)

    def getPointFingerCallback(self, pointingFinger):
        self.pointingFinger = pointingFinger.point
        self.pointingFinger.x = int(self.pointingFinger.x)
        self.pointingFinger.y = int(self.pointingFinger.y)

    def getActiveBbox(self, cls_id = 1):
        for box_ind, detection in enumerate(self.detections): 
            # && cls_id
            if detection.results[0].id != cls_id:
                continue

            bxmin = int(detection.bbox.center.x - detection.bbox.size_x / 2)
            bymin = int(detection.bbox.center.y - detection.bbox.size_y / 2)
            bxmax = int(detection.bbox.center.x + detection.bbox.size_x / 2)
            bymax = int(detection.bbox.center.y + detection.bbox.size_y / 2)
            bymax = int(detection.bbox.center.y + detection.bbox.size_y / 2)

            if self.pointingFinger.x >= bxmin and self.pointingFinger.x <= bxmax and self.pointingFinger.y >= bymin and self.pointingFinger.y <= bymax:
                return box_ind
               
        return -1
    
    def filterActiveBoxBreachZone(self, breach_zone):
        bbox_ind = self.getActiveBbox()
        bbox = self.detections[bbox_ind].bbox
        xmin = int(bbox.center.x - bbox.size_x)
        ymin = int(bbox.center.y - bbox.size_y)
        xmax = int(bbox.center.x + bbox.size_x)
        ymax = int(bbox.center.y + bbox.size_y)

        breach_zone[:ymin, :] = 0
        breach_zone[ymax:, :] = 0
        breach_zone[:, :xmin] = 0
        breach_zone[:, xmax:] = 0

        return breach_zone

    def depthImageCallback(self, depthImage):
        """
        bbox: hatch bbox give in (xmin, ymin, xmax, ymax)
        noise: disantce measurement noise [mm]
        disance2hatch: distance breach plane given in [mm] (assuming perpedicular position to plane)
        """

        timeStamp = depthImage.header.stamp
        depth = bridge.imgmsg_to_cv2(depthImage, desired_encoding='passthrough')
        # segmenting image
        breach_zone = np.zeros_like(depth)
        breach_zone[depth > self.distance2hatch + self.noise + self.breachDepthUncertainty] = 1
        
        breach_zone = breach_zone.astype(np.uint8)

        try:
            self.detectionImageOut = self.detectionImage.copy()
        except Exception:
            print("no active detections, can't crossvalidate with pointing finger")
            return
        
        # breach_zone = self.filterActiveBoxBreachZone(breach_zone)       
        self.detectionImageOut[breach_zone == 1, 0] = 255
        
        contours, hierarchy = cv2.findContours(breach_zone,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        breach_zone = cv2.cvtColor(breach_zone,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(self.detectionImageOut, contours, -1, (255,0,0), 3)
        area=[]
        cX=[]
        cY=[]
        for contour in contours:
            # calculate moments for each contour
            M = cv2.moments(contour)
            # calculate x,y coordinate of center
            area.append(cv2.contourArea(contour))
            cX.append(int(M["m10"] / M["m00"]))
            cY.append(int(M["m01"] / M["m00"]))

        if len(cX) > 0:
            # this is where crossing with window bbox will take place
            cX = cX[0]
            cY = cY[0]  
        
            breachPoint = PointStamped()
            breachPoint.header.stamp = timeStamp
            breachPoint.header.frame_id = 'd415_aligned_depth_to_color'
            breachPoint.point.x = cX
            breachPoint.point.y = cY
            breachPoint.point.z = self.distance2hatch
            
            cv2.circle(breach_zone, (breachPoint.point.x, breachPoint.point.y), 5, (255, 0, 0), -1)

            for detection in self.detections:
                cv2.circle(breach_zone, (int(detection.bbox.center.x), int(detection.bbox.center.y)), 5, (0, 255, 0), -1)

            cv2.circle(self.detectionImageOut, (self.pointingFinger.x, self.pointingFinger.y), 5, (0, 255, 0), -1)

            # publish results
            breachPointPublisher.publish(breachPoint)
            breachImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(breach_zone, cv2.COLOR_BGR2RGB), encoding="passthrough"))
            detectionImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(self.detectionImageOut, cv2.COLOR_BGR2RGB), encoding="passthrough"))
            print("breachPoint (x,y,z)=({}, {}, {})".format(breachPoint.point.x, breachPoint.point.y, breachPoint.point.z))
        
        return


if __name__=="__main__":
    depth_breacher=DepthBreacher()
    
    rospy.init_node('hatch_detector', anonymous=True)
    rospy.Subscriber("/d415/aligned_depth_to_color/image_raw", Image, depth_breacher.depthImageCallback, queue_size=1, buff_size=2 ** 24)  
    rospy.Subscriber("/d415/aligned_depth_to_color/image_raw", Image, depth_breacher.depthImageCallback, queue_size=1, buff_size=2 ** 24)  
    rospy.Subscriber("/seeker/detectionImage", Image, depth_breacher.getDetectionImageCallback, queue_size = 10)
    rospy.Subscriber("/seeker/detections", Detection2DArray, depth_breacher.getDetectionsCallback, queue_size = 10)
    rospy.Subscriber("/pointingfinger/TargetPoint", PointStamped, depth_breacher.getPointFingerCallback, queue_size = 10)
    rospy.spin()
