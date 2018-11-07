# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:07:35 2018

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
import MarkerDetector

# used if this bool is True
testWebcamFeed = False
index = 1
saveImgPath = 'testImg1.png'
# -----------------------------
# used if testWebcamFeed == False:

# Detect markers from a saved image:
plotImages = False
useImShow = True
savedImgPath = 'testImg2.png'

if testWebcamFeed:
    cap = cv2.VideoCapture(index)
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        if cv2.waitKey(10) == ord('s'):
            cv2.imwrite(saveImgPath, frame)
            print('saved image')
    
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
else:
    # Detect markers from a saved image:
    detector = MarkerDetector.MarkerDetector(plotImages,useImShow)
    time.sleep(1)
    #frame = detector.grabScreenshot(True)
    
    ##load image
    frame = cv2.imread(savedImgPath)
    pos = detector.detectMarkerPositions(frame)
    print(pos, len(pos))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # subtract the average from each of the channels
    diffBlue = cv2.subtract(frame[:,:,0], gray)
    diffGreen = cv2.subtract(frame[:,:,1], gray)
    diffRed = cv2.subtract(frame[:,:,2], gray)

