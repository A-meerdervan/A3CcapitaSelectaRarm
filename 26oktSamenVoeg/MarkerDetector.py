# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:11:43 2018

@author: Alex
"""

import numpy as np
import cv2
from skimage import morphology
from skimage.measure import regionprops
from skimage.measure import label
import time
import matplotlib.pyplot as plt
import gobalConst as cn
import visualEnvironmentOverlay as visOverlay
import math

class MarkerDetector:
    def __init__(self,plotImages,useImShow=False):
        index = cn.REAL_webcamIndex
        self.cap = cv2.VideoCapture(index)
        time.sleep(1)
        self.plotImages = plotImages
        self.useImShow = useImShow
        self.markerSizeRelativeTreshold = cn.REAL_markerAreaTreshold

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        firstImg = self.grabScreenshot()
        plt.imshow(firstImg)
        plt.show()
        self.videoWr = cv2.VideoWriter(cn.REAL_videoPath,cv2.VideoWriter_fourcc(*'MJPG'),30,(cn.REAL_webcamWindowWidth,cn.REAL_webcamWindowHeight))
        

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def grabScreenshot(self, showFrame = False):
        # Capture frame-by-frame
        ret, frame = self.cap.read()

        if showFrame:
            while (True):
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        return frame

    # Return [small marker, large marker] locations in pixels using a difference image.
    def detectMarker(self, img):
        kernel = np.ones((3,3),np.float32)/9
        img = cv2.filter2D(img,-1,kernel)
        th = 0.15*255 # set threshold value
        # convert to binary image
        ret,thresh = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
        regions = label(thresh, neighbors=8)
        regions = regionprops(regions)
        if self.plotImages:
            if self.useImShow:
                plt.imshow(thresh)
                plt.show()
            else:
                while (True):
                    cv2.imshow('frame',thresh)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cv2.destroyAllWindows()

        # retrieve the position of the markers
#        positions = np.array([[0,0]])
        rCnt = 0
        areas = np.array([])
        centroids = np.array([0,0])
        for r in regions:
            # only if the region is large enough, use it.
            if r.area > self.markerSizeRelativeTreshold:
                rCnt += 1
                areas = np.append(areas,r.area)
                centroids = np.vstack((centroids,r.centroid))
#                positions = np.vstack((positions, [r.centroid]))
        if not rCnt == 2:
            print('The nr of markers detected was not 2 for this color. it was '+str(rCnt) )
            # Jump out of the function so it can be called again.
            return None
#            raise(NameError('The nr of markers detected was not 2 for this color. it was '+str(rCnt) ))

        centroids = np.delete(centroids, 0, 0)  # delete dummy
        # Return [small marker, large marker]
        if areas[0] < areas[1]:
            return centroids
        else:
            return np.array([centroids[1],centroids[0]])


    def detectMarkerPositionsFromFrame(self, frame):
#        t = time.time()
        markersDetected = False
        tryCnt = 0
        while (markersDetected == False):
                
            """ Algorithm for detecting the markers is copied from Martha """
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except:
                print(frame)
                plt.imshow(frame)
                plt.show()
                raise(NameError('frame went wrong'))
    
            # subtract the average from each of the channels
            diffBlue = cv2.subtract(frame[:,:,0], gray)
            diffRed = cv2.subtract(frame[:,:,2], gray)
    
            locsBlue = self.detectMarker(diffBlue)
            locsRed = self.detectMarker(diffRed)
            if (locsBlue is None) or (locsRed is None):
                print('marker detection failed, try again nr ',tryCnt)
                # Loop to try and succesfully get a new frame:
                gotFrame = False
                while(gotFrame == False):
                    frame = self.grabScreenshot(False)
                    if frame is not None:
                        gotFrame = True
                    else: print("Frame was None, try againnnnnnnnnnnnn")
            else: markersDetected = True # to exit this while loop
            tryCnt += 1
        # Determine them in order:
        # from the base to the top is, small red, large blue, large red, small blue
        # return the marker locations in that order.
        asItShouldBe = np.array([locsRed[0],locsBlue[1],locsRed[1],locsBlue[0]])
        # The above did not work and a fix apeared to be this strange reordering:
        markerPosFix = []
        for loc in asItShouldBe:
            fixedLoc = [loc[1],loc[0]]
            # using insert reverses the order, which is somehow nessasary.
            markerPosFix.append(fixedLoc)
#        print(markerPosFix)

#        cv2.imwrite('markerImg.png', frame)
        return np.array(markerPosFix)
#        return asItShouldBe

    def getAnglesFromWebcam(self,envWalls,simGoalLoc):
        # Try to grab a frame, and if unsuccesul, try again
        gotFrame = False
        while(gotFrame == False):
            frame = self.grabScreenshot(False)
            if frame is not None:
                gotFrame = True
            else: print("Frame was None, try againnnnnnnnnnnnn")
        # detect the positions of the markers in the frame
        mkrPos = self.detectMarkerPositionsFromFrame(frame)
        # compute the angles of the robot arm using the positions of the markers
        th = self.computeAngles(mkrPos)

        # visualise the environment
        imgOverlay = visOverlay.getImgWithWallOverlay(frame,envWalls,simGoalLoc,mkrPos)
        cv2.imshow('frame',imgOverlay)
        self.videoWr.write(imgOverlay)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

#        print('grabbed frame!')
        return th, [0,0,0]

    def releaseVideo(self):
        self.videoWr.release()
        
        return
    
    # This function returns the joint angles as defined by us. 
    # Angles range from -179 unto + 180. 
    def computeAngles(self, markers):
        m1 = markers[0];    m2 = markers[1]
        m3 = markers[2];    m4 = markers[3]
        # This is a trick to add a non existing point, shifted to the left of m1
        # This makes the assumption that the webcam is nicely centered. Then 
        # it reasons as if m1 is attached to an arm which ends at m1shifted.
        # This makes it possible to use the same formula for the angle. 
        m1copy = list(m1)
        m1shifted = np.array(m1copy) + np.array([-100,0])
        th1 = self.getJointAngle(m1shifted,m2,m1)
        th2 = self.getJointAngle(m1,m3,m2)
        th3 = self.getJointAngle(m2,m4,m3)
        return np.array([th1,th2,th3])
    
    # Returns an angle in rad. The angle is taken between 3 points, where centroid
    # is the middle of the 3 angle defining points. Points expected in format [x,y]
    def getJointAngle(self,p1,p2,centroid):
        angle = math.atan2(p1[1] - centroid[1], p1[0] - centroid[0]) - math.atan2(p2[1] - centroid[1], p2[0] - centroid[0])
        # adjust
        if angle > 0:
            return angle - np.radians(180)
        else:
            return angle + np.radians(180)

#mrk = MarkerDetector(True,True)
#mar = mrk.getAnglesFromWebcam([0],[0])