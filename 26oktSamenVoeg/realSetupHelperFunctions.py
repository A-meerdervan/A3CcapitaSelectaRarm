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
import MarkerDetector as cam
import visualEnvironmentOverlay as visOverlay
import SimulationEnvironment as sim
from scipy.misc import imread

testOverlay = False

# used if this bool is True
testWebcamFeed = True
index = 0
saveImgPath = 'testImg1.png'
# -----------------------------
# used if testWebcamFeed == False:

if testOverlay:
    """These are to make it work, but would not be given"""
    ## From inspecting a webcam image with the arm at angle 1 = 90 degrees a2,a3 = 0,0
    ## Done with testImg1
    env = sim.SimulationEnvironment()
    envNr = 3
    env.getEnv(envNr)
    env.createRandomGoal()
    """These are inputs from outside:"""
    # get an env
    envWalls = env.envWalls
    simGoalLoc = env.createRandomGoal()
#    img = imread('kapo2Pi.PNG')
    detector = cam.MarkerDetector(False)
#    markers = detector.detectMarkerPositionsFromFrame(img)
    
    th = detector.getAnglesFromWebcam(env.envWalls,env.goal)
#    print(markers)
    print(th)

    # A loop that opens a window and keeps plotting the overlayed image.

#    while (True):
#        imgOverlay = visOverlay.getImgWithWallOverlay(img,envWalls,simGoalLoc,markers)
#        cv2.imshow('frame',img)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    cv2.destroyAllWindows()

elif testWebcamFeed:
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
