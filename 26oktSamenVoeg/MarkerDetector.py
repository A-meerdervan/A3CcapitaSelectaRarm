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

class MarkerDetector:
    def __init__(self,plotImages,useImShow=False):
        index = 0
        self.cap = cv2.VideoCapture(index)
        time.sleep(2)
        self.plotImages = plotImages
        self.useImShow = useImShow
        self.markerSizeRelativeTreshold = cn.REAL_markerAreaTreshold

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

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
#                print('area ',r.area)
#                print('pos ',r.centroid)
#                positions = np.vstack((positions, [r.centroid]))
        if not rCnt == 2:
            raise(NameError('The nr of markers detected was not 2 for this color. it was '+str(rCnt) ))

        centroids = np.delete(centroids, 0, 0)  # delete dummy
        # Return [small marker, large marker]
        if areas[0] < areas[1]:
            return centroids
        else:
            return np.array([centroids[1],centroids[0]])


    def detectMarkerPositions(self, frame):
#        t = time.time()
        """ Algorithm for detecting the markers is copied from Martha """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # subtract the average from each of the channels
        diffBlue = cv2.subtract(frame[:,:,0], gray)
        diffRed = cv2.subtract(frame[:,:,2], gray)

        locsBlue = self.detectMarker(diffBlue)
        locsRed = self.detectMarker(diffRed)
        # Determine them in order:
        # from the base to the top is, small red, large blue, large red, small blue
        # return the marker locations in that order.
        asItShouldBe = np.array([locsRed[0],locsBlue[1],locsRed[1],locsBlue[0]])
        # The above did not work and a fix apeared to be this strange reordering:
        markerPosFix = []
        for loc in asItShouldBe:
            fixedLoc = [loc[1],loc[0]]
            # using insert reverses the order, which is somehow nessasary. 
            markerPosFix.insert(0,fixedLoc)
        return np.array(markerPosFix)
            

    def computeAngles(self, mkr):
        th = []
#        mkr = np.vstack((zeroPosition, markers))
        for i in range(1,len(mkr)):
            d_y = mkr[i-1,1] - mkr[i,1]
            d_x = mkr[i,0] - mkr[i-1,0]

            t = np.arctan2(d_y,d_x)
            th = np.append(th, t)

        for i in range(1, len(th)):
            j = len(th) - i
            th[j] = th[j] - th[j-1]

        return th

