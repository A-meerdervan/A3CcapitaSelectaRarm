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
        self.plotImages = plotImages
        self.useImShow = useImShow
        self.markerSizeRelativeTreshold = cn.REAL_markerAreaTreshold

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
        return frame

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
        positions = np.array([[0,0]])
        rCnt = 0
        for r in regions:
            # only if the region is large enough, use it. 
            if r.area > self.markerSizeRelativeTreshold:
                rCnt += 1
#                print('area ',r.area)
#                print('pos ',r.centroid)
                positions = np.vstack((positions, [r.centroid]))
        if not rCnt == 2:
            raise(NameError('The nr of markers detected was not 2 for this color. it was '+str(rCnt) ))
        positions = np.delete(positions, 0, 0)  # delete dummy
        return positions


    def detectMarkerPositions(self, frame):
#        t = time.time()
        """ Algorithm for detecting the markers is copied from Martha """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # subtract the average from each of the channels
        diffBlue = cv2.subtract(frame[:,:,0], gray)
        diffRed = cv2.subtract(frame[:,:,2], gray)

        p1 = self.detectMarker(diffBlue)
        p2 = self.detectMarker(diffRed)
        return [p1,p2]
    