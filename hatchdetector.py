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

class DepthBreacher(object):
    
    def __init__(self):
        print('DepthBreacher instance created')
        self.distance2hatch=None
        self.noise=100 #[mm] depth noise
        self.detections = None
        self.distance2hatch = 3000
    def getDistanceCallback(self, distance):
        self.distance2hatch = distance
    
    def getDetectionsCallback(self, detections):
        self.detections = detections

    def depthImageCallback(self, depthImage, debug=False):
        """
        bbox: hatch bbox give in (xmin, ymin, xmax, ymax)
        noise: disantce measurement noise [mm]
        disance2hatch: distance breach plane given in [mm] (assuming perpedicular position to plane)
        """

        timeStamp = depthImage.header.stamp
        depth = bridge.imgmsg_to_cv2(depthImage, desired_encoding='passthrough')
        # segmenting image
        segmented_image = np.zeros_like(depth)
        segmented_image[depth > self.distance2hatch + self.noise] = 1
        
        breach_zone=np.zeros_like(depth)
        breach_zone[depth > 0] = 1
        # breach_zone[bbox[0]:bbox[2], bbox[1]:bbox[3]]=1
        breach_zone *= segmented_image * 255
        breach_zone = breach_zone.astype(np.uint8)
        if debug:
            cv2.imshow('depth', (depth / depth.max() * 255).astype(np.uint8))
            cv2.imshow('breach_zone', (breach_zone * 255).astype(np.uint8))
            cv2.waitKey(1)
        
        contours, hierarchy = cv2.findContours(breach_zone,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        breach_zone = cv2.cvtColor(breach_zone,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(breach_zone, contours, -1, (255,0,0), 3)
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


            # publish results
            breachPointPublisher.publish(breachPoint)
            breachImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(breach_zone, cv2.COLOR_BGR2RGB), encoding="passthrough"))
            print("breachPoint (x,y,z)=({}, {}, {})".format(breachPoint.point.x, breachPoint.point.y, breachPoint.point.z))
        
        pass


if __name__=="__main__":
    depth_breacher=DepthBreacher()
    
    rospy.init_node('hatch_detector', anonymous=True)
    rospy.Subscriber("/d415/aligned_depth_to_color/image_raw", Image, depth_breacher.depthImageCallback, queue_size=1, buff_size=2 ** 24)  
    rospy.spin()
