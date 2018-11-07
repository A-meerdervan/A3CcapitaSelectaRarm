# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:28:13 2018

@author: arnol
"""
import numpy as np
import gobalConst as cn

class Robot:
    def __init__(self, jointLenght = [100, 100, 80, 20], jointAngles = [90,0,0], width = 10):
        self.jointLength = cn.rob_JointLenght   # in pixels. [link 1, link 2, link 3, endeffector stick]
        self.reach = np.sum(self.jointLength)
        self.jointAngles =np.radians(jointAngles)# cn.rob_ResetAngles   # in radians # angles of death: np.array([2.44346096, -2.44346096, 2.44346096])
        self.width = cn.rob_JointWidth               # in pixels
#        self.endeffectorWidth = 2        # width of the slim endeffector piece
        self.maxJointAngle = cn.rob_MaxJointAngle
        self.standardDeviation = cn.rob_NoiseStandDev
        self.stepSize = cn.rob_StepSize            # stepsize in degrees

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
        noise = 0.0
        # Only add normal distributed noise if it is requested
        if addNoise:
            noise = np.random.normal(0, self.standardDeviation)
        # update the robots current angles using the noise and the command in joint
        self.jointAngles = self.jointAngles + self.stepSize*joint + noise*joint

        # check if the robot does not clash with itself
        badAngle = False
        if (self.jointAngles[0] > self.maxJointAngle[0]) or (self.jointAngles[0] < self.maxJointAngle[1]): badAngle = True;
        elif (self.jointAngles[1] > self.maxJointAngle[2]) or (self.jointAngles[1] < self.maxJointAngle[3]): badAngle = True;
        elif (self.jointAngles[2] > self.maxJointAngle[4]) or (self.jointAngles[2] < self.maxJointAngle[5]): badAngle = True;
        if badAngle:
            self.jointAngles = self.jointAngles - self.stepSize*joint - noise*joint

        return self.jointAngles