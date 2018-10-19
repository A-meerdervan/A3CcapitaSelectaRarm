# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:28:13 2018

@author: arnol
"""
import numpy as np

class Robot:
    def __init__(self, jointLenght = [100, 100, 80, 20], jointAngles = [90,0,0], width = 10):
        self.jointLength = jointLenght   # in pixels. [link 1, link 2, link 3, endeffector stick]
        self.reach = np.sum(self.jointLength)
        self.jointAngles = jointAngles   # in radians
        self.width = width               # in pixels
#        self.endeffectorWidth = 2        # width of the slim endeffector piece
        self.maxJointAngle = np.radians(np.array([170,170]))
        self.standardDeviation = 0.001
        self.stepSize = np.radians(1.4)             # stepsize in degrees

    def computeJointLocations(self, zeroPosition):
        # t_1 is angle at the base
        # set the lengths of the robot links. Unit is pixels. Each pixel is
        l = self.jointLength

        # jointAngles is in rads!!
        x_0 = zeroPosition[0]
        y_0 = zeroPosition[1]
        x_1 = x_0 + l[0] * np.cos(self.jointAngles[0])
        y_1 = y_0 - l[0] * np.sin(self.jointAngles[0])    # inverse is due to the layout of the pygame window
        x_2 = x_1 + l[1] * np.cos(self.jointAngles[0] + self.jointAngles[1])
        y_2 = y_1 - l[1] * np.sin(self.jointAngles[0] + self.jointAngles[1])
        x_3 = x_2 + l[2] * np.cos(self.jointAngles[0] + self.jointAngles[1] + self.jointAngles[2])
        y_3 = y_2 - l[2] * np.sin(self.jointAngles[0] + self.jointAngles[1] + self.jointAngles[2])
        # compute location of the endeffector stick
        x_ee = x_2 + (l[2]+l[3]) * np.cos(self.jointAngles[0] + self.jointAngles[1] + self.jointAngles[2])
        y_ee = y_2 - (l[2]+l[3]) * np.sin(self.jointAngles[0] + self.jointAngles[1] + self.jointAngles[2])

        return [x_0,y_0,x_1,y_1,x_2,y_2,x_3,y_3,x_ee,y_ee]

    def moveJoints(self, joint, addNoise):
        """ move one of the joints by the stepsize. joint is defined as the output.
        [j1 clockw., j1 countr clockw., j2 clo....] """
        noise = np.random.normal(0, self.standardDeviation)
        self.jointAngles = self.jointAngles + self.stepSize*joint + noise*joint

        # check angles
        for j in self.jointAngles:
            if (j > 3.141592653589793):
                j = j - 3.141592653589793
            elif(j < -3.141592653589793):
                j = j + 3.141592653589793

        return self.jointAngles
