# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:17:40 2018

@author: arnol

IMPLEMENTS A CLASS FOR DETECTING THE IMAGES. THE FIRST LINES BEFORE THE CLASS ARE
USED TO SNAP PICTURES AND TEST THE WEBCAM
"""

import numpy as np
import cv2
from skimage import morphology
from skimage.measure import regionprops
from skimage.measure import label
import time


#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # works
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # works
#cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

#while(True):
#    # Capture frame-by-frame
#    ret, frame = cap.read()
#
#    # Our operations on the frame come here
##    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#    # Display the resulting frame
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        print(frame.shape)
#        break
#
#    if cv2.waitKey(10) == ord('s'):
#        cv2.imwrite('testImg1.png', frame)
#        print('saved image')
#
#
## When everything done, release the capture
#
#cap.release()
#cv2.destroyAllWindows()




class MarkerDetector:
    def __init__(self):
        index = 0
        self.cap = cv2.VideoCapture(index)

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
        threshClean = morphology.remove_small_objects(thresh, min_size=300, connectivity=8)
        regions = label(threshClean, neighbors=8)
        regions = regionprops(regions)

        while (True):
            cv2.imshow('frame',threshClean)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

        # retrieve the position of the markers
        positions = np.array([[0,0]])
        for r in regions:
            positions = np.vstack((positions, [r.centroid]))
        positions = np.delete(positions, 0, 0)  # delete dummy

        return positions


    def detectMarkerPositions(self, frame):
        t = time.time()
        """ Algorithm for detecting the markers is copied from Martha """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # subtract the average from each of the channels
        diffRed = cv2.subtract(frame[:,:,0], gray)
        diffGreen = cv2.subtract(frame[:,:,1], gray)
        diffBlue = cv2.subtract(frame[:,:,2], gray)

        p1 = self.detectMarker(diffRed)
        p2 = self.detectMarker(diffGreen)
        p3 = self.detectMarker(diffBlue)

        # sum up the channels
#        img = diffRed + diffGreen + diffBlue
#
#        kernel = np.ones((3,3),np.float32)/9
#        img = cv2.filter2D(img,-1,kernel)
#
#        th = 0.15*255 # set threshold value
#        # convert to binary image
#        ret,thresh = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
#        threshClean = morphology.remove_small_objects(thresh, min_size=300, connectivity=8)
#
##        regions = regionprops(threshClean, cache=True)
#
#        r = label(threshClean, neighbors=8) # label the different regions
#        regions = regionprops(r, cache=True)
#        while (True):
#            cv2.imshow('frame',threshClean)
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break
#        cv2.destroyAllWindows()

#        nrMarkers = 4
#        if (len(regions) != nrMarkers):
#            # not good, find out what the right regions are
#            # TODO: select right regions if that is a problem.
#            continue

        # retrieve the position of the markers
#        positions = np.array([[0,0]])
#        for r in regions:
#            positions = np.vstack((positions, [r.centroid]))
#        positions = np.delete(positions, 0, 0)  # delete dummy

#        print('Detection took: ', time.time() - t)

        return [p1,p2,p3]

    def computeAngles(self, zeroPosition, markers):
        th = []
        mkr = np.vstack((zeroPosition, markers))
        for i in range(1,len(mkr)):
            d_y = mkr[i-1,1] - mkr[i,1]
            d_x = mkr[i,0] - mkr[i-1,0]

            t = np.arctan2(d_y,d_x)
            th = np.append(th, t)

        for i in range(1, len(th)):
            j = len(th) - i
            th[j] = th[j] - th[j-1]

        return th



#
#detector = MarkerDetector()
#time.sleep(1)
#frame = detector.grabScreenshot(True)
#
###load image
##img = cv2.imread('testImage2.png')
#pos = detector.detectMarkerPositions(frame)
#print(pos, len(pos))
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#diffRed = cv2.subtract(img[:, :, 0], gray)
#red = img[:,:,0]
#
#while (True):
#    cv2.imshow('frame',diffRed)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cv2.destroyAllWindows()


