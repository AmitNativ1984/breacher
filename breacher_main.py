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

X = 0
Y = 1
Z = 2

class DepthBreacher(object):
    
    def __init__(self):
        self.noise= 0.25 #[m] depth noise
        self.detections = None
        self.distance2window = None
        self.infDistance = 20 #[m]
        self.breach_zone = None
        self.pointingFinger = None
        self.mode = 'pointingFinger' # can be opne of: 'area';'pointingFinger'
        self.depthCamModel = PinholeCameraModel()
        self.nonValidMarginRatio = 0.1    # ratio taken from image width and height. the pixel of world point must be in only in valid area
        self.minBlobAreaPixels = 100
        self.cameraRayInitiated = False
        rospy.init_node('depth_breach', anonymous=True)
        rospy.loginfo('node depth_breach created')
        self.init_subsrcibers()
        self.init_publishers()

    def init_subsrcibers(self):
        rospy.Subscriber("/d415/aligned_depth_to_color/image_raw", Image, self.depthImageCallback, queue_size=1, buff_size=2 ** 24)  
        rospy.Subscriber("/d415/color/image_raw", Image, self.getColorImageRaw, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber("/seeker/windowCenterWorldPoint", PointStamped, self.getWorldPointXYZCallback, queue_size=1)
        rospy.Subscriber("/d415/aligned_depth_to_color/camera_info", CameraInfo, self.getDepthCameraModel, queue_size=1)
        rospy.Subscriber("/surface_normal", Vector3Stamped, self.getSurfaceNormal, queue_size=10)

        rospy.logdebug("subscribers init successfull")
    
    def init_publishers(self):
        self.breachImagePublisher = rospy.Publisher("/breacher/breachImage", Image, queue_size=10)
        self.colorImagePublisher = rospy.Publisher("/breacher/breachImageColor", Image, queue_size=1)
        self.breachPointWorldPublisher = rospy.Publisher("/breacher/breachPointWorld", PointStamped, queue_size=10)
        self.projectionImagePublisher = rospy.Publisher("/breacher/projectionImage", Image, queue_size=10)
        self.thresholdProjectionPublisher = rospy.Publisher("/breach/threholdedProjectio", Image, queue_size=10)
        rospy.logdebug("publishers init successfull")
    
    def getDepthCameraModel(self, CameraInfo_msg):
        self.depthCamInfo_Msg = CameraInfo_msg
        self.depthCamModel.fromCameraInfo(CameraInfo_msg)
        self.depthCamWidth = CameraInfo_msg.width
        self.depthCamHeight = CameraInfo_msg.height

        self.minValidRow = int(self.depthCamHeight * self.nonValidMarginRatio)
        self.maxValidRow = int(self.depthCamHeight * (1 - self.nonValidMarginRatio))
        self.minValidCol = int(self.depthCamWidth * self.nonValidMarginRatio)
        self.maxValidCol = int(self.depthCamWidth * (1 - self.nonValidMarginRatio))

        if not self.cameraRayInitiated:
            self.initateCameraRays()
    
    def initateCameraRays(self):               
        self.camRays = np.zeros((self.depthCamHeight, self.depthCamWidth, 3))
        row = range(0, self.depthCamHeight)    
        col = range(0, self.depthCamWidth)    

        cols, rows = np.meshgrid(col, row)
        
        self.camRays[..., X] = (cols - self.depthCamModel.cx()) / self.depthCamModel.fx()
        self.camRays[..., Y] = (rows - self.depthCamModel.cy()) / self.depthCamModel.fy()
        norm = np.sqrt(self.camRays[..., X]**2 + self.camRays[..., Y]**2 + 1)
        self.camRays[..., X] /= norm
        self.camRays[..., Y] /= norm
        self.camRays[..., Z] = 1.0 / norm
              
        self.cameraRayInitiated = True
        rospy.loginfo("created camera rays")

    def projectCamOptRaysTo3D(self, depth):
        """
            project depth image to 3d space (in camera opt coordinates) by multiplying ray of a pixel by its depth value
        """
        R = np.dstack((depth, depth, depth))
        return R * self.camRays
    
    
    def getSurfaceNormal(self, surface_normal_vec):
        self.surfaceNormal = surface_normal_vec
        
    @staticmethod
    def projectXYZonSurfaceNormal(xyz, surfaceNormal):
        projetion = surfaceNormal.vector.x * xyz[..., X] + surfaceNormal.vector.y * xyz[..., Y] + surfaceNormal.vector.z * xyz[..., Z]
        return projetion
        
    @staticmethod
    def thresholdProjection(projection, surfaceNormal, noise=0.):
        normalVec = np.array([surfaceNormal.vector.x, surfaceNormal.vector.y, surfaceNormal.vector.z])
        bwImg = np.zeros((projection.shape))
        bwImg[projection/1E3 > np.linalg.norm(normalVec) + noise] = 1
        return bwImg

    @staticmethod
    def tranformXYZ(sourceXYZ, T):
        """ T is 4x4 Mat of tranformation in homegenous coordinates (x,y,z,1) """

        x = sourceXYZ[...,X]    # MxN
        y = sourceXYZ[...,Y]    # MxN
        z = sourceXYZ[...,Z]    # MxN

        # xyz1: 4xM*N
        xyz1 = np.vstack((x.reshape((1, -1)),
                         y.reshape((1, -1)),
                         z.reshape((1, -1)),
                         np.ones_like(x).reshape((1, -1))))
        
        xyz1World = np.dot(T, xyz1)

        worldXYZ = np.dstack((xyz1World[X, ...].reshape((x.shape[0], -1)),
                             xyz1World[Y, ...].reshape((y.shape[0], -1)),
                             xyz1World[Z, ...].reshape((z.shape[0], -1))))

        return worldXYZ     # MxNx3

    @staticmethod
    def getTransformMatrix4_toWorld(ImageMsg):
        """ get 4x4 transofrmation matrix from camera optical frame to map """
        M = listner.asMatrix("/map", ImageMsg.header)
        return M
    
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
    def tfPointWorld2CamOpt(ImageMsg, worldXYZ, transformTimeStamp):
        listner.waitForTransform(ImageMsg.header.frame_id, "/map", transformTimeStamp, rospy.Duration(1.0))
        worldXYZ.header.stamp = transformTimeStamp
        point_in_cam_optical_frame = listner.transformPoint(ImageMsg.header.frame_id, worldXYZ)
        return point_in_cam_optical_frame 

    @staticmethod
    def tfPointCamOpt2World(camOptXYZ, transformTimeStamp):
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


    def getWorldPointXYZCallback(self, breachPointWorld):
               
        self.breachPointWorld = breachPointWorld

    @staticmethod
    def getBreachAreaID(breach_zones, cc_img, pixel, n_labels, stats, centroids, minBlobAreaPixels):
        
        dist = np.inf
        label_id = -1
        for currlabel, (centroid, stat) in enumerate(zip(centroids, stats)):
            areaPixels = stat[-1]
            if breach_zones[cc_img == currlabel][0] > 0:                                    
                
                R = np.sqrt((centroid[0] - pixel[0]) ** 2 +
                            (centroid[1] - pixel[1]) ** 2)
                if R < dist and areaPixels > minBlobAreaPixels:
                    dist = R
                    label_id = currlabel
          
        return label_id
    
    def getColorImageRaw(self, colorImageRaw):
        self.colorImageRaw = bridge.imgmsg_to_cv2(colorImageRaw, desired_encoding='passthrough')
        self.colorImageRaw = self.colorImageRaw.astype(np.uint8)

    def createPointStamped(self, timestamp, frame_id, x, y, z):
        pointstamped = PointStamped()
        pointstamped.header.stamp = timestamp
        pointstamped.header.frame_id = frame_id
        pointstamped.point.x = x
        pointstamped.point.y = y
        pointstamped.point.z = z

        return pointstamped

    def removeImageMargins(self, img):
        img[:self.minValidRow, :] = 0
        img[self.maxValidRow:, :] = 0   
        img[:, :self.minValidCol] = 0
        img[:, self.maxValidCol:] = 0
        return img
    
    def depthImageCallback(self, depthImageMsg):
        """
        bbox: hatch bbox give in (xmin, ymin, xmax, ymax)
        noise: disantce measurement noise [mm]
        disance2hatch: distance breach plane given in [mm] (assuming perpedicular position to plane)
        """

        self.depthImageMsg = depthImageMsg
        surfaceNormal = self.surfaceNormal
        depthOrg = bridge.imgmsg_to_cv2(depthImageMsg, desired_encoding='passthrough')             
        depth = depthOrg.copy()
        depth[depth == 0] = self.infDistance * 1E3
        depth = cv2.GaussianBlur(depth, (25, 25), 0.25)
        depthCamTimeStamp = depthImageMsg.header.stamp

        # convert current depth map to world XYZ coordinates using camera rays and depth map
        camOptXYZ = self.projectCamOptRaysTo3D(depth)
        T = self.getTransformMatrix4_toWorld(depthImageMsg)
        # transform xyz points in camera opt to map coordinates
        worldXYZ = self.tranformXYZ(camOptXYZ, T)
        projection = self.projectXYZonSurfaceNormal(worldXYZ, surfaceNormal)
        projectionImgMsg = CvBridge().cv2_to_imgmsg( (np.abs((np.abs(projection) - np.abs(projection).min())) / np.abs(projection).max() * 255).astype(np.uint8), 'passthrough')
        self.projectionImagePublisher.publish(projectionImgMsg)
        
        # remove margins (margins always have non valid depth readings)
        breach_zones = self.thresholdProjection(projection, surfaceNormal, noise=self.noise)
        breach_zones = self.removeImageMargins(breach_zones)
        breach_zone_ImageMsg = CvBridge().cv2_to_imgmsg((breach_zones * 255).astype(np.uint8), 'passthrough')
        self.thresholdProjectionPublisher.publish(breach_zone_ImageMsg)
        # get current breach point in world coordinates (x,y,z)
        breachPointWorld = self.breachPointWorld
        # get vector from current position to breach point
        vecCamOpt2BreachPointWorld = self.tfWorld2BaseLink(depthImageMsg, breachPointWorld, transformTimeStamp=depthCamTimeStamp)
        distance2target = 1E3 * np.linalg.norm(np.array([vecCamOpt2BreachPointWorld.point.x, vecCamOpt2BreachPointWorld.point.y, vecCamOpt2BreachPointWorld.point.z]))
        distance2breach = distance2target + self.noise

        # get breach point in camera optical frame coordinates
        breachPointCamOpt = self.tfPointWorld2CamOpt(depthImageMsg, breachPointWorld, transformTimeStamp=depthCamTimeStamp)
        
        # get breach point in camera pixels:
        u, v = self.depthCamModel.project3dToPixel((breachPointCamOpt.point.x, breachPointCamOpt.point.y, breachPointCamOpt.point.z))
        if u <= self.minValidCol or u >= self.maxValidCol or \
            v <= self.minValidRow or v >= self.maxValidRow:
            rospy.logdebug("Breach point projected from map to camera pixels {}, {} is inside non valid margins of depth cam".format(u,v))
            return -1
      
        # correct pixels of breach point by center of blob closest to initial breach point
        n_labels, cc_img, stats, centroids = cv2.connectedComponentsWithStats(breach_zones.astype(np.uint8), connectivity=8)
        cc_img = np.array(cc_img).astype(np.uint8)
        label_id = self.getBreachAreaID(breach_zones, cc_img, (u, v), n_labels, stats, centroids, self.minBlobAreaPixels)
        
        # verify background is not selected 
        if label_id == -1:
            rospy.logdebug("Invalid connected component label id=[-1]. Got backgound")
            return -1
        
        u_new = centroids[label_id, 0]
        v_new = centroids[label_id, 1]

        breach_zone = np.zeros_like(cc_img)
        breach_zone[cc_img == label_id] = 1
        
        # create new breach point in camera optical coordinates 
        breachPointCam = self.createPointStamped(depthCamTimeStamp, 'd415_depth_frame', centroids[label_id, 0], centroids[label_id, 1], distance2breach)

        # convert new breach point from camera optical to world
        ray = np.array(self.depthCamModel.projectPixelTo3dRay((u_new,v_new)))
        ray /= np.linalg.norm(ray)
        breachPointCamOpt = self.createPointStamped(depthCamTimeStamp, depthImageMsg.header.frame_id,
                                                    ray[0] * distance2target/1E3,
                                                    ray[1] * distance2target/1E3,
                                                    ray[2] * distance2target/1E3)

        breachPointWorld = self.tfPointCamOpt2World(breachPointCamOpt, depthCamTimeStamp)

        # publish results
        self.breachPointWorldPublisher.publish(breachPointWorld)
        rospy.loginfo("Updated breachPoint in map coordinates (x,y,z) = {} [m]".format(breachPointWorld.point.x, breachPointWorld.point.y, breachPointWorld.point.z/1E3))

        # publish BW image possible breach zones, original breach point and updated breach point
        breachZone_ = np.zeros((breach_zone.shape[0], breach_zone.shape[1], 3))
        breachZone_[...,0] = breach_zones
        breachZone_[...,1] = breach_zones
        breachZone_[...,2] = breach_zones
        breachZone_ *= breachZone_.max() * 255
        cv2.circle(breachZone_, (int(u_new), int(v_new)), 5, (255, 0, 0), -1)
        cv2.circle(breachZone_, (int(u), int(v)), 5, (0, 0, 255), -1)
        self.breachImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor((breachZone_).astype(np.uint8), cv2.COLOR_BGR2RGB), encoding="passthrough"))
        
        # publish color image with mask of selected breach area and updated breach point
        breach_zone = cv2.resize(breach_zone, (self.colorImageRaw.shape[1], self.colorImageRaw.shape[0]), interpolation=cv2.INTER_NEAREST)
        colorImageRaw = self.colorImageRaw.copy()
        colorImageRaw[breach_zone == 1, 0] = 200
        cv2.circle(colorImageRaw, (8*int(u), 8*int(v)), 5, (0, 255, 0), -1)       
        cv2.circle(colorImageRaw, (8*int(u_new), 8*int(v_new)), 5, (255, 0, 0), -1)       
        newImageMsg = bridge.cv2_to_imgmsg(cv2.cvtColor(colorImageRaw, cv2.COLOR_BGR2RGB), encoding="passthrough")
        newImageMsg.header.stamp = rospy.Time.now()
        self.colorImagePublisher.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(colorImageRaw, cv2.COLOR_BGR2RGB), encoding="passthrough"))       
    
        return


if __name__=="__main__":
    depth_breacher=DepthBreacher()  
    listner = tf.TransformListener()
    rospy.spin()
