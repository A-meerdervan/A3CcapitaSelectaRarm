# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:07:02 2018

@author: arnold
"""
import pygame
import numpy as np
#import time
import math
import Robot
#import threading
#import gobalConst as cn

#from runConfig import rc # has high level constants
import runConfig as rc # has high level constants


#import copy

#pygame.init()
#screen = pygame.display.set_mode((400, 400))
#def importPygame():
#    import pygame

class SimulationEnvironment:
    def __init__(self, randomGoal, WINDOW_WIDTH = 400, WINDOW_HEIGHT = 400):
        self.screen = 0 #pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.zeroPosition = ([rc.sim_WINDOW_WIDTH/2, rc.sim_WINDOW_HEIGHT])   # reference for drawing
        self.WINDOW_HEIGHT = rc.sim_WINDOW_HEIGHT
        self.WINDOW_WIDTH = rc.sim_WINDOW_WIDTH
        self.robot = Robot.Robot()
        self.envWalls = np.array([[(120,400), (120,300)], [(120,300), (40,300)], [(40,300),
                      (40,140)], [(40,140), (280,140)], [(280,140), (280,400)], [(280,400),(120,400)]])
        self.envWallSide = ['l', 'b', 'l', 't', 'r', 'b']
        self.envPoints = self.wallsTOPoints(self.envWalls)
        self.addNoise = rc.sim_AddNoise
        self.goal = rc.sim_Goal
        self.senseDistances = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        self.distanceEndEffector = np.array([0,0])
        self.randomGoal = rc.sim_RandomGoal
        self.ctr = 0
        self.nrOfGoalHits = 0 # keep track of the number of times the goal has been reached
#        self.reward = 0
        self.clock = pygame.time.Clock()

        if (randomGoal):
            self.goal = self.createRandomGoal()

    def step(self, action):
        # map the 6 outputs from the NN to the actions one can take
        # returns whether there is a collision (True), the reward, and whether the goal
        # has been reached
        a = np.zeros((6,1))   # added to convert the output of the agent to the simulation env
        a[action - 1] = 1
#        action = a

        a = a.reshape((3, 2))
        actionMap = [1,-1]

        act = a * actionMap
        act = np.max(act, axis=1) + np.min(act, axis=1)

        # move the robot
        self.robot.moveJoints(act, self.addNoise)

        [col, dist] = self.checkNOCollision()
        [r, reachGoal] = self.computeReward(dist, not col)

        done = False
        if (not col or reachGoal):
            done = True

        # increase count
        self.ctr = self.ctr + 1

        state = self.getState()

        self.reward = r # temporary line!
#        self.clock.tick(10)

        return [state, r, done, self.ctr]  # return [state, reward, done, timestep]


    def reset(self):
        """resets the robot and the environment, and also returns the state"""
#        if not pygame.display.get_init():
#            pygame.display.init()
#            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        self.robot.jointAngles = rc.rob_ResetAngles

        self.ctr = 0

        if (self.randomGoal):
            self.goal = self.createRandomGoal()

        # check the environment for collisions (needed for the state)
        [col, dist] = self.checkNOCollision()
        # compute reward (as it computes distance to endeffector)
        [r, reachGoal] = self.computeReward(dist, not col)

        return self.getState()

    def getState(self):
        s1 = np.resize(self.senseDistances/400, (12,1)) # normalized between 0 and 1
        s2 = np.resize(self.robot.jointAngles/3.141592653589793, (3,1)) # normalized between -1 and 1
        s3 = np.resize(self.distanceEndEffector/400, (2,1)) # normalized between 0 and 1
        s = np.reshape(np.vstack((s1,s2,s3)), (1,17))
        # this reshapes the state into the format that is needed for the Neural net
        s = np.reshape(s,[np.prod(s.shape)])
        return s

    def setGoal(self, goal):
        self.goal = goal

    def setEnvironment(self, envWalls, envSides, setGoal):
        self.envWalls = envWalls
        self.envWallSide = envSides

        if (setGoal):
            self.goal = self.createRandomGoal()

        return

    def computeReward(self, minDistance, collision):
        # compute distance to goal
        xy = self.robot.computeJointLocations(self.zeroPosition)
        xy_ee = np.array([xy[8], xy[9]])

        d = self.goal - xy_ee
        self.distanceEndEffector = d
        d = np.sqrt(d[0]**2 + d[1]**2)

        reward = 0.0

        reachedGoal = False
        if (d < 5):
            # reached goal
            reachedGoal = True
            reward = 100.0
            # Increase the number of times the goal was hit
            self.nrOfGoalHits = self.nrOfGoalHits + 1
            # If the goal has been reached, then exit this function
            return [reward/200.0, reachedGoal]
        else:
            reachedGoal = False
# This is commented out to try only sparse rewards, LETS SEE!
        g = -0.01
        offset = 0.6
        reward = offset * math.exp(g * d) - offset

        thresholdWall = 20.0
        wallReward = 100.0

        if collision:
            reward = reward - wallReward
# Commented this to try without the linear reward until the wall, LETS SEE!
#        elif minDistance < thresholdWall:
#            reward = reward - (thresholdWall - minDistance) * (wallReward/thresholdWall)

#        if collision:
#            reward = reward - wallReward

        return [reward/200.0, reachedGoal]

    # TODO: Manage screen in A3C file --> return all objects to be drawn
    def render(self):
        if (self.screen == 0):
            self.screen = pygame.display.set_mode((rc.sim_WINDOW_WIDTH, rc.sim_WINDOW_HEIGHT))

        [wall, goal] = self.drawEnvironment()
        robot = self.drawRobot()

        return [wall, robot, goal]

    def drawEnvironment(self):
#        pygame.event.pump()

        self.screen.fill((0,0,0)) # black out screen

        colour = (255,0,0)
        thickness = 10
        ctr = 0
        # draw each wall
        walls = np.array([], dtype=None)
        for w in self.envWalls:
            if (self.envWallSide[ctr] == 'l'):
                # (left, top, width, height)
                rect = pygame.Rect(w[1][0]-thickness,min(w[0][1],w[1][1]) ,thickness, abs(w[1][1]-w[0][1]))
            elif (self.envWallSide[ctr] == 'r'):
                rect = pygame.Rect(w[1][0],min(w[0][1],w[1][1]) ,thickness, abs(w[1][1]-w[0][1]))
            elif (self.envWallSide[ctr] == 't'):
                rect = pygame.Rect(min(w[0][0],w[1][0]),w[1][1]-thickness , abs(w[1][0]-w[0][0]), thickness)
            elif (self.envWallSide[ctr] == 'b'):
                rect = pygame.Rect(min(w[0][0],w[1][0]),w[1][1] , abs(w[1][0]-w[0][0]), thickness)

            pygame.draw.rect(self.screen, colour, rect)
            walls = np.append(walls, rect)
            ctr += 1

        # draw goal
        pygame.draw.circle(self.screen, (0,255,0), self.goal, 5, 0)

        return [walls, self.goal]

    def drawRobot(self):
        """ draw the robot on the screen"""
        [x_0, y_0, x_1,y_1,x_2,y_2,x_3,y_3,x_ee,y_ee] = self.robot.computeJointLocations(self.zeroPosition)

        colour = (255,255,255)
        pointlist =  [(x_0,y_0), (x_1,y_1),(x_2,y_2),(x_3,y_3)]
        endEffector = [(x_3,y_3),(x_ee,y_ee)]

        thickness = self.robot.width
        pygame.draw.lines(self.screen, colour, False, pointlist, thickness)
        pygame.draw.lines(self.screen, colour, False, endEffector, 2)

        pygame.display.flip()

        return [(x_0, y_0), (x_1,y_1),(x_2,y_2),(x_3,y_3),(x_ee,y_ee)]

    def isPointInEnvironment(self, point):
        """ find four walls surrounding the point. If no four points can be found, the point is not
        in the environment """
        ctr = 0
        j1 = point   # just because it is easier than changing all vars
        wallID = np.array([99,99,99,99])
        # find the matching wall at each side
        for w in self.envWalls:

            if (self.envWallSide[ctr] == 'l' and j1[0] >= w[0,0] and max(w[:,1]) >= j1[1] and min(w[:,1]) <= j1[1]):
                # find a correct wall on the left
                if (wallID[0] == 99):
                    wallID[0] = ctr
                else:
                    # test which wall is closer to the joint
                    if (abs(self.envWalls[wallID[0],0,0]-j1[0]) > abs(w[0,0]-j1[0])):
                        wallID[0] = ctr

            elif (self.envWallSide[ctr] == 'r' and j1[0] <= w[0,0] and max(w[:,1]) >= j1[1] and min(w[:,1]) <= j1[1]):
                # find a wall on the right
                if (wallID[1] == 99):
                    wallID[1] = ctr
                else:
                    # test which wall is closer to the joint
                    if (abs(self.envWalls[wallID[1],0,0]-j1[0]) > abs(w[0,0]-j1[0])):
                        wallID[1] = ctr

            elif (self.envWallSide[ctr] == 't' and j1[1] >= w[0,1] and max(w[:,0]) >= j1[0] and min(w[:,0]) <= j1[0]):
                # find a top wall
                if (wallID[2] == 99):
                    wallID[2] = ctr
                else:
                    # test which wall is closer to the joint
                    if (abs(self.envWalls[wallID[2],0,1]-j1[1]) > abs(w[0,1]-j1[1])):
                        wallID[2] = ctr

            elif (self.envWallSide[ctr] == 'b' and j1[1] <= w[0,1] and max(w[:,0]) >= j1[0] and min(w[:,0]) <= j1[0]):
                # find a bottom wall

                if (wallID[3] == 99):
                    wallID[3] = ctr
                else:
                    # test which wall is closer to the joint
                    if (abs(self.envWalls[wallID[3],0,1]-j1[1]) > abs(w[0,1]-j1[1])):
                        wallID[3] = ctr

            ctr += 1
        # check if there are environment walls on all sides
        z = np.where(wallID == 99)
        if (len(z[0]) != 0):
            return [False, wallID]
        else:
            return [True, wallID]

    def createRandomGoal(self):
        # create a random goal that sits within the environment AND reach of the robot
        minx = np.min(self.envWalls[:,0,0])
        maxx = np.max(self.envWalls[:,0,0])
        miny = np.min(self.envWalls[:,0,1])
        maxy = np.max(self.envWalls[:,0,1])

        pointCorrect = False
        while(not pointCorrect):
            x = np.random.randint(minx, maxx)
            y = np.random.randint(miny, maxy)

            [pointCorrect, w] = self.isPointInEnvironment(np.array([x,y]))

            d = np.array([x,y]) - self.zeroPosition
            d = np.sqrt(d[0]**2 + d[1]**2)
            if (d > self.robot.reach):
                pointCorrect = False

        return np.array([x,y])

    def wallsTOPoints(self,walls):
        """environment is defined in walls. This can be converted into the corner
        points of the environment"""
        points = walls[0,0]  # init array with first
        for i in range(1, len(walls)):
            points = np.vstack((points, walls[i,0]))

        return points

    def findCornerPoint(self,cornerPoints, j1, j2):
        """uses two joint positions to see with which points the joint could possibly
        collide"""
        points = np.array([0,0])  # fill with dummy
        for c in cornerPoints:
            if (c[0] > min(j1[0], j2[0]) and c[0] < max(j1[0], j2[0])
            and c[1] > min(j1[1], j2[1]) and c[1] < max(j1[1], j2[1])):
                points = np.vstack((points, c))

        points = np.delete(points, 0, 0)  # delete dummy

        return points

    def computeLine(self,j1, j2):
        # compute slope
        a  = (j2[1]-j1[1])/(j2[0]-j1[0])
        # compute bias
        b = j1[1] - a*j1[0]

        return [a, b]  # return: y = a * x + b

    def computeDistanceLineANDPoint(self,line, point):
         """ The line is defined by two variables, a and b: y=ax+b
         and the point: [x,y]. Implemented function is standard distance
         between line and point. Returns the distance and the point on the line that
         is closest """

         d = abs((-1*line[0]*point[0]) + point[1] + (-1*line[1]))
         d = d / np.sqrt((-1*line[0])**2 + 1)

         x = ((point[0]+line[0]*point[1])+(line[0]*(-1*line[1]))) / ((-1*line[0])**2 + 1)
         y = ((-1*line[0])*(-point[0]-line[0]*point[1])+line[1]) / ((-1*line[0])**2 + 1)

         return [d, x, y]

    def computeDistanceJointsTOWalls(self, joint, walls, jointID):
        ctr = 0
        for w in walls:
            if w != 99:
                wall = self.envWalls[w]

                if (wall[0,0] == wall[1,0]):
                    # vertical line
                    d = abs(wall[0,0] - joint[0])
                else:
                    d = abs(wall[0,1] - joint[1])

                self.senseDistances[jointID, ctr] = d
            else:
                self.senseDistances[jointID, ctr] = 0

            ctr = ctr + 1
        return

    def computeMinDistanceJointsTOWalls(self, joint, walls):
        dist = 100

        for w in walls:
            if w != 99:
                wall = self.envWalls[w]

                if (wall[0,0] == wall[1,0]):
                    # vertical line
                    d = abs(wall[0,0] - joint[0])
                else:
                    d = abs(wall[0,1] - joint[1])

                dist = min(dist, d)
        return dist

    def checkNOCollisionWithItself(self):
        ang = self.robot.jointAngles

#        angCum = 0
        for i in range(0, len(ang)):
#            angCum = angCum + ang[i]
            # maximum angle of each joint with respect to the other joint
            if ang[i] > 2.9670597283903604 or ang[i] < -2.9670597283903604:
                return False

        return True

    def checkNOCollision(self):
        # cannot collide with itself now
#        c = self.checkNOCollisionWithItself()
#        if not c:
#            return [c, 0]

        """ checks if the robot is fully in the environment. Returns True if there is no collision"""
        [x_0,y_0,x_1,y_1,x_2,y_2,x_3,y_3,x_ee,y_ee] = self.robot.computeJointLocations(self.zeroPosition)

        # check two middle joints
        w = self.robot.width/2
        l = [x_1,y_1,x_2,y_2,x_3,y_3]
        distance = 100   # compute distance --> used for the reward function
#        robotInEnvironment = True
        for i in range(0,3):
            # compute distance to joint, needed for the state
            pSense = np.array([l[2*i],l[2*i+1]])
            [b, wSense] = self.isPointInEnvironment(pSense)
            self.computeDistanceJointsTOWalls(pSense, wSense, i)

            # both sides of the joint
            p1 = np.array([l[2*i] - w ,l[2*i+1]])
            p2 = np.array([l[2*i] + w ,l[2*i+1]])

            [b1, wall1] = self.isPointInEnvironment(p1)
            [b2, wall2] = self.isPointInEnvironment(p2)

            d = self.computeMinDistanceJointsTOWalls(p1, wall1)
            d = min(d, self.computeMinDistanceJointsTOWalls(p2, wall2))
            distance = min(distance, d)  # distance is defined as the minimum distance

            # if either point is not in the environment
            if (not b1 or not b2):
                return [False, 0]

        # check endeffector
        [b, wall] = self.isPointInEnvironment([x_ee, y_ee])
        if not b:
            return [False, 0]
        # check distance
        distance = min(distance, self.computeMinDistanceJointsTOWalls([x_ee, y_ee], wall))
        # TODO: check link connecting the small endeffector

        # check the links between joints for collisions
        l = [x_0, y_0, x_1,y_1,x_2,y_2,x_3,y_3,x_ee,y_ee]
        for j in range(0,3):
#            th = np.sum(self.robot.jointAngles[:j+1])
#            # test cum angle, to check which side should be checked for a collision
#            # TODO: check both lines.
#            if (th > 1.5707963267948966):
#                corners = self.findCornerPoint(self.envPoints, [l[2*j] - w, l[2*j+1]], [l[2*j+2] - w, l[2*j+3]])
#                [a,b] = self.computeLine([l[2*j] - w, l[2*j+1]], [l[2*j+2] - w, l[2*j+3]])
#            else:
#                corners = self.findCornerPoint(self.envPoints, [l[2*j] + w, l[2*j+1]], [l[2*j+2] + w, l[2*j+3]])
#                [a,b] = self.computeLine([l[2*j] + w, l[2*j+1]], [l[2*j+2] + w, l[2*j+3]])
#
#            if (corners.all() != 0):
#                for c in corners:
#                    [d, x, y]  = self.computeDistanceLineANDPoint([a,b], c)
#                    distance = min(distance, d)
#
#                    b = self.isPointInEnvironment([x, y])
#                    if not b:
#                        return [False, 0]

            [b1, d] = self.checkLine([l[2*j] - w, l[2*j+1]], [l[2*j+2] - w, l[2*j+3]])
            distance = min(distance, d)

            # check both sides of the link
            [b2, d] = self.checkLine([l[2*j] + w, l[2*j+1]], [l[2*j+2] + w, l[2*j+3]])
            distance = min(distance, d)

            if (not b1 or not b2):
                return [False, 0]

        return [True, distance]  # return true is no collission is found

    def checkLine(self, p1, p2):
        distance = 1000
        corners = self.findCornerPoint(self.envPoints, p1, p2)
        [a,b] = self.computeLine(p1, p2)

        if (corners.all() != 0):
            for c in corners:
                [d, x, y]  = self.computeDistanceLineANDPoint([a,b], c)
                distance = min(distance, d)

                b = self.isPointInEnvironment([x, y])
                if not b:
                    return [False, 0]

        return [True, distance]

class EnvironmentCreator:
    def __init__(self):
        self.WINDOW_WIDTH = 400
        self.WINDOW_HEIGHT = 400


    def createEnvironment(self):
        # type 1:
        leftWall = np.random.randint(0,self.WINDOW_WIDTH/2 - 60)
        middleWall = np.random.randint(leftWall,self.WINDOW_WIDTH/2 - 51)
        rightWall = np.random.randint(self.WINDOW_WIDTH/2 + 50, self.WINDOW_WIDTH)
        top = np.random.randint(30,self.WINDOW_HEIGHT/2)
        bottom = np.random.randint(self.WINDOW_HEIGHT/2 + 30, self.WINDOW_HEIGHT)

        self.walls = np.array([[(middleWall,self.WINDOW_HEIGHT), (middleWall,bottom)], [(middleWall,bottom), (leftWall,bottom)], [(leftWall,bottom),
                      (leftWall,top)], [(leftWall,top), (rightWall,top)], [(rightWall,top), (rightWall,self.WINDOW_HEIGHT)], [(rightWall,self.WINDOW_HEIGHT),(middleWall,self.WINDOW_HEIGHT)]])
        self.wallSide = ['l', 'b', 'l', 't', 'r', 'b']

        return [self.walls, self.wallSide]

#    def createVeryRandomEnvironment(self):
#        nrWalls = np.random.randint(3,10)
#        walls = np.array([[0,0],[0,0]])
#        wLabels = np.array('b')
#
#        wi = self.WINDOW_WIDTH / 2
#        hi = self.WINDOW_HEIGHT / 2
#        # create the walls
#        for i in range(1,nrWalls + 1):
#            p = walls[i-1, 1, :]
#            pNew = np.array([0,0])
#            if (wLabels[i-1] == 'b'):
#                pNew[0] = p[0]
#                pNew[1] = np.random.randint(p[1], hi*2)
#                # label is either 'l' or 'r'
#                if (p[0] < wi):
#
#                else:
#
#            elif(wLabels[i-1] == 'l'):
#                # label is either 't' or 'b'
#
#            elif(wLabels[i-1] == 't'):
#                # label is either 'l' or 'r'
#
#            elif(wLabels[i-1] == 'r'):
#                # label is either 't' or 'b'
#
#            w = np.random.randint(0,wi - 60)


