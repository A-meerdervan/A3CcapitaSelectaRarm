# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:07:02 2018

@author: arnold
"""
import pygame
from pygame.locals import *
import numpy as np
#import time
import math
import Robot
import time
#import threading
import gobalConst as cn
import moveRobotArm as cont
import MarkerDetector as cam
#import copy

#pygame.init()
#screen = pygame.display.set_mode((400, 400))
#def importPygame():
#    import pygame

class SimulationEnvironment:
    def __init__(self, realSetup = False):
        self.screen = 0 #pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.zeroPosition = ([cn.sim_WINDOW_WIDTH/2, cn.sim_WINDOW_HEIGHT])   # reference for drawing
        self.WINDOW_HEIGHT = cn.sim_WINDOW_HEIGHT
        self.WINDOW_WIDTH = cn.sim_WINDOW_WIDTH
        self.robot = Robot.Robot([100,100,80,20], np.radians(np.array([105,-90,120])))
        self.getEnv(cn.sim_defaultEnvNr)

        self.addNoise = cn.sim_AddNoise
        self.goal = cn.sim_Goal
        self.senseDistances = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        self.distanceEndEffector = np.array([0,0])
        self.randomGoal = cn.sim_RandomGoal
        self.ctr = 0 # current timestep
        self.wallHits = 0 #nr of times the agent wanted to hit the wall
        self.clock = pygame.time.Clock()
        self.actQueue = []

        self.image = pygame.image.load('armImage3.png')
        pygame.display.set_caption('Robot arm simulation')

        self.realSetup = realSetup
        if (realSetup):
            port = cn.REAL_comPort
            # start up controller
            self.controller = cont.RobotController(port, False)

            # start up webcam
            self.markerDetector = cam.MarkerDetector(False,False)

        # init previous states
        self.distEE = 0
        self.senseDist = 0
        self.stateAng = cn.rob_ResetAngles

    def step(self, action):
        # save previous state of the robot
        stateAng = self.robot.jointAngles
        senseDist = self.senseDistances
        distEE = self.distanceEndEffector

        # map the 6 outputs from the NN to the actions one can take
        act = self.actionToRoboAction(action)

        # move the robot
        self.robot.moveJoints(act, self.addNoise)

        [col, dist] = self.checkNOCollision()
        [r, reachGoal] = self.computeReward(dist, not col)

        # Prikkeldraad code!
        done = False
        # if a collision occured then set the robot angles back
        #print('Col detected after action ',not col)
        if not col:
            self.wallHits += 1
            # Restore the state
            self.robot.jointAngles = stateAng
            self.senseDistances = senseDist
            self.distanceEndEffector = distEE
        elif reachGoal:
            done = True

        # increase count
        self.ctr = self.ctr + 1

        state = self.getState()

        return [state, r, done, self.ctr]  # return [state, reward, done, timestep]

    # TODO: fix this function
    def stepRealWorld(self, action):
        # must set the zeroPosition
#        self.zeroPosition
        # remember two states back
#        s0 = self.stateAng
#        s1 = self.senseDist
#        s2 = self.distEE
#
#        self.stateAng = self.robot.jointAngles
#        self.senseDist = self.senseDistances
#        self.distEE = self.distanceEndEffector

        # map the 6 outputs from the NN to the actions one can take
        act = self.actionToRoboAction(action)
        self.actQueue.insert(0,list(act))

        # angles are stored in robot, computed here
#        self.robot.jointAngles = self.robot.jointAngles + cn.rob_StepSize*act

        # perform action in the real world
        if (self.controller.hasConnection):
            # find the joint to move
#            joint = np.argmax(np.abs(act))
#            self.controller.moveArm(joint, self.robot.jointAngles, True)
#            print(act)
            self.controller.moveArm2(act, self.robot.jointAngles, False)

        # use webcam to evaluate the angles
        self.robot.jointAngles, dummy = self.markerDetector.getAnglesFromWebcam(self.envWalls, self.goal)
        # check for any collision or if the robot has reached the goal
        [col, dist] = self.checkNOCollision()

        # if collision: Do something

#        if not col:
#            # check for collisison
#            self.wallHits += 1
#            self.robot.jointAngles = self.stateAng
#            if (self.controller.hasConnection):
#                self.controller.moveArm2(-1*act, self.robot.jointAngles, False)
#            # if after restoring the previous state, the robot is still hitting
#            # the walls, go back another state
#            print('collision')
#            [c, dist] = self.checkNOCollision()
#            if not c:
#                self.robot.jointAngles = s0
#                self.senseDistances = s1
#                self.distanceEndEffector = s2
#                if (self.controller.hasConnection):
#                    self.controller.moveArm2(-1*act, self.robot.jointAngles, False)
#                print('second collision')
        ctr = 0
        while not col:
            self.wallHits += 1
            if (self.controller.hasConnection):
                self.controller.moveArm2(-1*np.asarray(self.actQueue[ctr]), self.robot.jointAngles, False)
                time.sleep(0.5)

            # use webcam to evaluate the angles
            self.robot.jointAngles, dummy = self.markerDetector.getAnglesFromWebcam(self.envWalls, self.goal)

            [col, dist] = self.checkNOCollision()

            ctr += 1
            print('restore previous angle, try:  ', ctr)


        # only computes the reward if the collision has been solved.
        # this can only be done if you are not training
        [r, reachGoal] = self.computeReward(dist, not col)

        done = False
        if reachGoal:
            done = True

        # return state
        state = self.getState()

        # increase count
        self.ctr = self.ctr + 1
        print(self.ctr)
        # make sure the act queue is only 10 long always.
        if len(self.actQueue) > 10:
            self.actQueue.pop()

        return [state, r, done, self.ctr]

    def actionToRoboAction(self,action):
        a = np.zeros((6,1))   # added to convert the output of the agent to the simulation env
        a[action - 1] = 1
#        action = a
        a = a.reshape((3, 2))
        actionMap = [1,-1]
        act = a * actionMap
        return np.max(act, axis=1) + np.min(act, axis=1)

    def reset(self):
        """resets the robot and the environment, and also returns the state"""
#        if not pygame.display.get_init():
#            pygame.display.init()
#            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        self.robot.jointAngles = cn.rob_ResetAngles

        self.ctr = 0 # reset the number of timesteps
        self.wallHits = 0 # reset the number of times the wall was hit

        if cn.rob_RandomWalls:
            if cn.sim_FullyRandomWalls:
                self.setsRandomEnv()
            else:
                self.setRandomEnv()

        if (self.randomGoal):
            self.goal = self.createRandomGoal()

        if (cn.rob_RandomInit):
            self.createRandomInit()

        if (self.realSetup):
            self.controller.moveInitialPosition(self.robot.jointAngles)

        # check the environment for collisions (needed for the state)
        [col, dist] = self.checkNOCollision()
        if not col:
            print('colision on start, you fucked up the angles')
            raise NameError('A collission has occured before the robot '
                 + 'had the chance to move! you must redefine your'+
                 'starting angles/reset angles')
        # compute reward (as it computes distance to endeffector)
        [r, reachGoal] = self.computeReward(dist, not col)

        return self.getState()

    # This function is used to test configurations, sizes and environments
    def runTestMode(self,fromTestConsts):
        self.ctr = 0 # reset the number of timesteps
        self.wallHits = 0 # reset the number of times the wall was hit
        if fromTestConsts:
            self.robot.jointAngles = cn.TEST_rob_ResetAngles
            if cn.rob_RandomWalls:
                self.setRandomEnv()
            if (self.randomGoal):
                self.goal = self.createRandomGoal()
            if (cn.rob_RandomInit):
                self.createRandomInit()

        # use the stuff which is in the rest of the file
        else:
            self.robot.jointAngles = cn.rob_ResetAngles
            if cn.rob_RandomWalls:
                self.setRandomEnv()

            if (self.randomGoal):
                self.goal = self.createRandomGoal()

            if (cn.rob_RandomInit):
                self.createRandomInit()

        # check the environment for collisions (needed for the state)
        [col, dist] = self.checkNOCollision()
        if not col:
            print('collision on start, you fucked up the angles')
#                raise NameError('A collission has occured before the robot '
#                     + 'had the chance to move! you must redefine your'+
#                     'starting angles/reset angles')
#                        # compute reward (as it computes distance to endeffector)
        [r, reachGoal] = self.computeReward(dist, not col)
        # show the init screen
        self.render()
        print('Used angles ',np.degrees(self.robot.jointAngles),' minD ',dist)
        print(' r: ',r, ' col? ', not col)
        # now step using the user input
        # check whether the window was closed.
        continiue = True
        while continiue:
            Act = False
            for evt in pygame.event.get():
                if evt.type == pygame.KEYDOWN:
                    if evt.key == pygame.K_r:
                        # a bit of recurion
                        self.runTestMode(fromTestConsts)
                        continiue = False
                        break
                    elif evt.key == pygame.K_q or (evt.type == pygame.QUIT):
                        # this will quit the window
                        continiue = False
                        pygame.quit()
                        break
                    elif evt.key == pygame.K_1: action = 1; Act = True
                    elif evt.key == pygame.K_2: action = 2; Act = True
                    elif evt.key == pygame.K_3: action = 3; Act = True
                    elif evt.key == pygame.K_4: action = 4; Act = True
                    elif evt.key == pygame.K_5: action = 5; Act = True
                    elif evt.key == pygame.K_6: action = 6; Act = True
                    else: print('This key does nothing')
            # take a step
            if Act:
                print('1 step with a = ',action)
                act = self.actionToRoboAction(action)
                # move the robot
                self.robot.moveJoints(act, self.addNoise)
                [col, dist] = self.checkNOCollision()
                [r, reachGoal] = self.computeReward(dist, not col)
                print('Used angles ',np.degrees(self.robot.jointAngles),' minD ',dist)
                jointLocs = self.robot.computeJointLocations(self.zeroPosition)
                print('xee ',jointLocs[-2],' yee ',jointLocs[-1],' r: ',r, ' col? ', not col)
                # Print the current distance
                d = self.distanceEndEffector # x and y dist
                d = np.sqrt(d[0]**2 + d[1]**2) # absolute dist
                print('D relative: ',d)
                self.render()
#            FPS = 5 # check 5 times every second (Frames per s = 1)
#            self.clock.tick(FPS)
##            for event in pygame.event.get():
#                if event.type == pygame.QUIT:
#                    pygame.quit()
#                    raise NameError('Pygame screen was succesfully closed :)')
#


    def getState(self):
        s1 = np.resize(self.senseDistances/400, (12,1)) # normalized between 0 and 1
        s2 = np.resize(self.robot.jointAngles/3.141592653589793, (3,1)) # normalized between -1 and 1
        s3 = np.resize(self.distanceEndEffector/400, (2,1)) # normalized between 0 and 1
        s = np.reshape(np.vstack((s1,s2,s3)), (1,17))
        # get it into the shape the neural net wants
        s = np.reshape(s,[np.prod(s.shape)])
        return s

    def setGoal(self, goal):
        self.goal = goal

    # Take a set of connected points that go clockwise from left bottom to right bottom
    # and convert it to a valid wall construct. points = np.array([(x1,y1),(x2,y2),...])
    def getWalls(self,points):
        walls = []
        startPoint = points[0]; curEndPoint = startPoint
        for point in points[1:]:
            walls.append([curEndPoint,point])
            curEndPoint = point
        walls.append([points[-1],startPoint])
        return np.array(walls)

    # Switch between environments per reset of the environment.
    def setRandomEnv(self):
        envNr = np.random.randint(1, 7 + 1)
        self.getEnv(envNr)

    def getEnv(self,envNr):
        rC = 200 # robotCenter, the x pos of the arm bottom
        WH = self.WINDOW_HEIGHT
        WW = self.WINDOW_WIDTH
#        envNr = 5

        # LEFT corner (most used during training. Our first env.)
        if envNr == 1:
            pR = 60 # the width of the pipe
            tYs = 330 # rurnYstart This is the start of the right side of the pipe
            tYe = 250 #turn Y end, the top op the pipe
            pFe = 65 # pixelsFromEdge is the x distance to the side of the screen of the pipe.
            wallPoints = np.array([(rC-pR,WH), (rC-pR,tYs),(pFe,tYs),(pFe,tYe),(rC+pR,tYe),(rC+pR,WH)])
            self.envWallSide = ['l', 'b','l', 't', 'r', 'b']
        # A pipe witch is straight up and wide and high
        elif envNr == 2:
            pR = 60 # the width of the pipe
            tYe = 200 # #turnYend, the top op the pipe
            wallPoints = np.array([(rC-pR,WH),(rC-pR,tYe),(rC+pR,tYe),(rC+pR,WH)])
            self.envWallSide = ['l', 't','r', 'b']
        # A pipe witch is straight up and narrow
        elif envNr == 3:
            pR = 50 # the width of the pipe
            tYe = 250 # #turnYend, the top op the pipe
            wallPoints = np.array([(rC-pR,WH),(rC-pR,tYe),(rC+pR,tYe),(rC+pR,WH)])
            self.envWallSide = ['l', 't','r', 'b']
        # T - shaped pipe
        elif envNr == 4:
            pR = 50 # the width of the pipe
            tYs = 350 # rurnYstart This is the start of the turn of the pipe
            tYe = 250 #turn Y end, the top op the pipe
            pFe = 65 # pixelsFromEdge is the x distance to the side of the screen of the pipe.
            wallPoints = np.array([(rC-pR,WH),(rC-pR,tYs),(pFe,tYs),(pFe,tYe),(WW-pFe,tYe),(WW-pFe,tYs),(rC+pR,tYs),(rC+pR,WH)])
            self.envWallSide = ['l', 'b','l', 't', 'r', 'b','r','b']
        # CROSS - section
        elif envNr == 5:
            pR = 52 # the width of the pipe
            tYs = 370 # rurnYstart This is the start of the turn of the pipe
            tYe = 300 #turn Y end, the top op the pipe
            tYE = 170 # Even more the top of the pipe :P
            pFe = 50 # pixelsFromEdge is the x distance to the side of the screen of the pipe.
            wallPoints = np.array([(rC-pR,WH),(rC-pR,tYs),(pFe,tYs),(pFe,tYe),(rC-pR,tYe),(rC-pR,tYE),(rC+pR,tYE),(rC+pR,tYe),(WW-pFe,tYe),(WW-pFe,tYs),(rC+pR,tYs),(rC+pR,WH)])
            self.envWallSide = ['l', 'b','l', 't','l','t','r','t','r','b','r','b']
        # RIGHT corner.
        elif envNr == 6:
            pR = 60 # the width of the pipe
            tYs = 330 # rurnYstart This is the start of the turn of the pipe
            tYe = 250 #turn Y end, the top op the pipe
            pFe = 65 # pixelsFromEdge is the x distance to the side of the screen of the pipe.
            wallPoints = np.array([(rC-pR,WH), (rC-pR,tYe),(WW-pFe,tYe),(WW-pFe,tYs),(rC+pR,tYs),(rC+pR,WH)])
            self.envWallSide = ['l', 't','r', 'b', 'r', 'b']
        # This environment is to move freely
        elif envNr == 7:
            tYe = 30 #turn Y end, the top op the pipe
            pFe = 30 # pixelsFromEdge is the x distance to the side of the screen of the pipe.
            x2 = WW - pFe
            wallPoints = np.array([(pFe,WH), (pFe,tYe), (x2,tYe),(x2,WH)])
            self.envWallSide = ['l', 't','r', 'b']

        # For every environment:
        self.envWalls = self.getWalls(wallPoints)
        self.envPoints = self.wallsTOPoints(self.envWalls)
        return

    def getOldEnv(self,envNr):
        rC = 200 # robotCenter, the x pos of the arm bottom
        #envNr = 7

        # This is a pipe witch a corner to the left (most used during training. Our first env.)
        if envNr == 1:
            pR = 80 # the width of the pipe
            self.envWalls = np.array([[(rC-pR,400), (rC-pR,300)], [(rC-pR,300), (40,300)], [(40,300),
                      (40,140)], [(40,140), (rC+pR,140)], [(rC+pR,140), (rC+pR,400)], [(rC+pR,400),(rC-pR,400)]])
            self.envWallSide = ['l', 'b','l', 't', 'r', 'b']
        # A pipe witch is straight up and wide
        elif envNr == 2:
            pR = 90 # the width of the pipe
            self.envWalls = np.array([[(rC-pR,400), (rC-pR,100)], [(rC-pR,100), (rC+pR,100)], [(rC+pR,100),
                      (rC+pR,400)], [(rC+pR,400), (rC-pR,400)]])
            self.envWallSide = ['l', 't','r', 'b']
        # A pipe witch is straight up and narrower
        elif envNr == 3:
            pR = 70 # the width of the pipe
            self.envWalls = np.array([[(rC-pR,400), (rC-pR,100)], [(rC-pR,100), (rC+pR,100)], [(rC+pR,100),
                      (rC+pR,400)], [(rC+pR,400), (rC-pR,400)]])
            self.envWallSide = ['l', 't','r', 'b']
        # this is a T shaped pipe
        elif envNr == 4:
            pR = 72 # the width of the pipe
            tYe = 130
            self.envWalls = np.array([[(rC-pR,400), (rC-pR,300)], [(rC-pR,300), (45,300)], [(45,300),
                      (45,tYe)], [(45,tYe), (355,tYe)], [(355,tYe), (355,300)], [(355,300),(rC+pR,300)],[(rC+pR,300),(rC+pR,400)],[(rC+pR,400),(rC-pR,400)]])
            self.envWallSide = ['l', 'b','l', 't', 'r', 'b','r','b']#
        #
        # this is a turn to the right Which is at the top of the reach of the arm
        elif envNr == 5:
            tYs = 250 # rurnYstart This is the start of the right side of the pipe
            tYe = 130 #turn Y end, the top op the pipe
            pR = 70 # the width of the pipe
            self.envWalls = np.array([[(rC-pR,400), (rC-pR,tYe)], [(rC-pR,tYe), (355,tYe)], [(355,tYe),
                      (355,tYs)], [(355,tYs),(rC+pR,tYs)],[(rC+pR,tYs),(rC+pR,400)],[(rC+pR,400),(rC-pR,400)]])
            self.envWallSide = ['l', 't','r', 'b', 'r', 'b']
        # this is a turn to the right which is high but not that high
        elif envNr == 6:
            tYs = 270 # rurnYstart This is the start of the right side
            tYe = 110 #turn Y end, the top of the pipe
            pR = 73 # the width of the pipe
            self.envWalls = np.array([[(rC-pR,400), (rC-pR,tYe)], [(rC-pR,tYe), (355,tYe)], [(355,tYe),
                      (355,tYs)], [(355,tYs),(rC+pR,tYs)],[(rC+pR,tYs),(rC+pR,400)],[(rC+pR,400),(rC-pR,400)]])
            self.envWallSide = ['l', 't','r', 'b', 'r', 'b']
            # This is a turn to the right which is the same as the original pipe to the left
        elif envNr == 7:
            tYs = 300 # rurnYstart This is the start of the right side
            tYe = 140 #turn Y end, the top op the pipe
            pR = 60 # the width of the pipe
            self.envWalls = np.array([[(rC-pR,400), (rC-pR,tYe)], [(rC-pR,tYe), (355,tYe)], [(355,tYe),
                      (355,tYs)], [(355,tYs),(rC+pR,tYs)],[(rC+pR,tYs),(rC+pR,400)],[(rC+pR,400),(rC-pR,400)]])
            self.envWallSide = ['l', 't','r', 'b', 'r', 'b']
        # This environment is to move freely. TODO, get rid of maxY handicap
        elif envNr == 8:
            tYe = 30 #turn Y end, the top op the pipe
            x1 = 30
            x2 = 400 - x1
            self.envWalls = np.array([[(x1,400), (x1,tYe)], [(x1,tYe), (x2,tYe)], [(x2,tYe),
                      (x2,400)], [(x2,400),(x1,400)]])
            self.envWallSide = ['l', 't','r', 'b']
        else:
            raise NameError('envNr was out of range, no such environment defined')

        self.envPoints = self.wallsTOPoints(self.envWalls)

        return


    def setEnvironment(self, envWalls, envSides, setGoal):
        self.envWalls = envWalls
        self.envWallSide = envSides

        if (setGoal):
            self.goal = self.createRandomGoal()

        return

    def computeReward(self, minDistance, collision):
        # compute distance to goal
        # TODO: put variables in the globalConstants
        xy = self.robot.computeJointLocations(self.zeroPosition)
        xy_ee = np.array([xy[8], xy[9]])

        d = self.goal - xy_ee
        self.distanceEndEffector = d
        d = np.sqrt(d[0]**2 + d[1]**2)

        reachedGoal = False
        reward = 0
        if (d <= cn.sim_goalRadius):
            # reached goal
            reachedGoal = True
            reward = cn.sim_GoalReward
            return [reward/cn.sim_rewardNormalisation, reachedGoal]
        else:
            reachedGoal = False
#        if reachedGoal:
#            print('d ',d,'minD ',minDistance)
#            print('terminalReward 06 ',reward)
        # Compute the reward relative to the distance of the end effector
        # to the goal. This is caculated exponentially
        gamma = cn.sim_expRewardGamma # this sets the slope
        offset = cn.sim_expRewardOffset  # this determines the maximum negative reward
        # only calculate the relative punishment if the goal has not been reached
        if not cn.sim_SparseRewards:
            reward = reward + offset * math.exp(gamma * d) - offset
        # Calculate the linear punishment when being near to the wall
        thresholdWall = cn.sim_thresholdWall # the amount of pixels where the linear rewards starts
        wallReward = cn.sim_WallReward
        if collision:
            reward = reward - wallReward
        elif minDistance < thresholdWall:
            reward = reward - (thresholdWall - minDistance) * (wallReward/thresholdWall)

        return [reward/cn.sim_rewardNormalisation, reachedGoal]

    # TODO: Manage screen in A3C file --> return all objects to be drawn
    def render(self):
        if (self.screen == 0):
            self.screen = pygame.display.set_mode((cn.sim_WINDOW_WIDTH, cn.sim_WINDOW_HEIGHT))

        try:
            [wall, goal] = self.drawEnvironment()
            robot = self.drawRobot()
        except:
            [wall, robot, goal] = [0,0,0]

        return [wall, robot, goal]

    def drawEnvironment(self):
#        pygame.event.pump()

        self.screen.fill((0,0,0)) # black out screen
        time.sleep(0.1)

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
        pygame.draw.lines(self.screen, colour, False, endEffector, 2)

        imgs = self.rescaleArmImgs()
        for i in range(len(imgs)):
            middle = np.asarray(imgs[i].get_size()) / 2

            pos = np.add(pointlist[i], pointlist[i+1]) / 2 - middle
            self.screen.blit(imgs[i], pos)

            pygame.draw.circle(self.screen, colour, (int(pointlist[i][0]),
                                                     int(pointlist[i][1])), int(thickness/2), 0)

        pygame.display.update()

        return [(x_0, y_0), (x_1,y_1),(x_2,y_2),(x_3,y_3),(x_ee,y_ee)]

    def rescaleArmImgs(self):
        w = self.robot.width
        l = self.robot.jointLength
        th = self.robot.jointAngles

        ctr =  0
        ang = -90
        imgs = []
        for i in th:
            im = pygame.transform.scale(self.image, (w, l[ctr]))
            ang = ang + np.degrees(i)
            im = pygame.transform.rotate(im, ang)

            imgs = np.append(imgs, im)

            ctr += 1

        return imgs

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
        # the end effector location of a random body config is used as the
        # goal location.
        while(True):
            notUsed,xee,yee = self.getRandomAllowedBodyConfig()
            if yee < cn.sim_Y_threshold_goal :
                break
        return np.array([int(xee),int(yee)])


    def computeMinDistanceToCorner(self, p):
        if len(self.envPoints) == 1:
            self.envPoints = self.wallsTOPoints(self.envWalls)

        minDist = 100
        for w in self.envPoints:
            d = p - w
            d = np.sqrt(d[0]**2 + d[1]**2)

            minDist = min(minDist, d)

        return minDist

    def createRandomInit(self):
            decision = np.random.random_sample()
            if decision < cn.rob_resetAngles_Lchance:
                # Init for going left
                self.robot.jointAngles = cn.rob_ResetAngles_Left
            elif decision < (cn.rob_resetAngles_Lchance + cn.rob_resetAngles_Rchance):
                # Init for going left
                self.robot.jointAngles = cn.rob_ResetAngles_Right
            else: # Create an uniformly chosen robot configuration
                angles,xee,yee = self.getRandomAllowedBodyConfig()
                self.robot.jointAngles = angles
            return
    # This function gets a randomly chosen body configuration which fits in the
    # environment and repects its max angles.
    def getRandomAllowedBodyConfig(self):
        savedAngles = self.robot.jointAngles # save to restore at the end
        maxAngls = self.robot.maxJointAngle
        initCorrect = False
        while(not initCorrect):
            R = np.random.random_sample((3)) # get 3 values from 0 to 1
            th = np.zeros(3) # init the angles
            for i in range(3):
                # have the angles range from their min and max value
                th[i] = maxAngls[0+2*i] + R[i]*(maxAngls[1+2*i] - maxAngls[0+2*i])

            self.robot.jointAngles = th
            [initCorrect, minDtoWall] = self.checkNOCollision()
            # In case the goal is placed within the treshold distance
            # of a wall, than try again for a better one
            if minDtoWall < cn.sim_thresholdWall:
                initCorrect = False

        # now a succesful configuration has been found
        # get a point for the location of the end effector:
        jointLocs = self.robot.computeJointLocations(self.zeroPosition)
        # restore the robot's jointAngles:
        self.robot.jointAngles = savedAngles
        # return, jointAngles, endEffectorX, endEffectorY
        return th,jointLocs[-2],jointLocs[-1]

    def createHeatMap(self,envNr,totalDots):
        self.getEnv(envNr)
        # These will contain the possible end effector points
        Xpts = []#np.zeros(totalDots)
        Ypts = []#np.zeros(totalDots)
        for i in range(totalDots):
            notUsed,xee,yee = self.getRandomAllowedBodyConfig()
            Xpts.append(xee); Ypts.append(yee)
        return Xpts,Ypts

    def wallsTOPoints(self,walls):
        """environment is defined in walls. This can be converted into the corner
        points of the environment"""
        points = walls[0,0]  # init array with first
        for i in range(1, len(walls)):
            points = np.vstack((points, walls[i,0]))

        return points

    def pointsTOWalls(self, points):
        """environment defined in points. This can be converted into the walls
         of the environment"""
        # necessary or init of the array
#        walls = np.array([points[0], points[1]])

        walls = np.zeros((len(points), 2, 2))
        for i in range(0, len(points) - 1):
            w = np.array([points[i], points[i + 1]])
#            walls = np.hstack((walls, w))
            walls[i] = w

        # last one needs to be added manually
        x = np.array([points[len(points)-1], points[0]])
        walls[len(points)-1] = x

        return walls

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
            # TODO: this does not work
            [b1, d] = self.checkLine([l[2*j] - w, l[2*j+1]], [l[2*j+2] - w, l[2*j+3]])
            distance = min(distance, d)

            # check both sides of the link
            [b2, d] = self.checkLine([l[2*j] + w, l[2*j+1]], [l[2*j+2] + w, l[2*j+3]])
            distance = min(distance, d)
#            print(b1, b2)

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

                [b, x] = self.isPointInEnvironment([x, y])
                if not b:
                    return [False, 0]

        return [True, distance]

    def setsRandomEnv(self):
        # get the random envNr
        envNr = np.random.randint(1, 4 + 1)
#        envNr = 4
#        minWallSize = 50
        minW = 50
        # Switch between environments per reset of the environment.
        # This is a pipe which a corner to the left (most used during training. Our first env.)

        """ The points are chosen: all begin with w. Then either x or y, for x coordinate or y.
        Then wx0 is the left most point. Then wx1 is to the right of wx0 but to the left of wx2 etc.
        Same for y: wy0 is the bottom (so at y=400) coordinate , above it is wy1
        """

        # turn to the left
        if envNr == 1:
            wx0 = np.random.randint(0,self.WINDOW_WIDTH/2 - minW*2-1)
            wx1 = np.random.randint(wx0 + minW, self.WINDOW_WIDTH/2 - minW)
            wx2 = np.random.randint(self.WINDOW_WIDTH/2 + minW, self.WINDOW_WIDTH)
            wy0 = 400
            wy2 = np.random.randint(minW, self.WINDOW_HEIGHT/2)
            wy1 = np.random.randint(200, wy0 - minW/3)

            self.envPoints = np.array([(wx1, wy0), (wx1, wy1), (wx0, wy1), (wx0, wy2), (wx2, wy2), (wx2, wy0)])
            self.envWallSide = ['l', 'b','l', 't', 'r', 'b']
        # A pipe which is straight up
        elif envNr == 2:
            wx0 = np.random.randint(0, self.WINDOW_WIDTH/2 - 50)
            wx1 = np.random.randint(self.WINDOW_WIDTH/2 + 50, self.WINDOW_WIDTH)
            wy0 = self.WINDOW_HEIGHT
            wy1 = np.random.randint(100, self.WINDOW_HEIGHT / 2 + 40)

            self.envPoints = np.array([(wx0,wy0), (wx0, wy1), (wx1, wy1), (wx1, wy0)])
            self.envWallSide = ['l', 't','r', 'b']
        # this is a T shaped pipe
        elif envNr == 3:
            wx0 = np.random.randint(0,self.WINDOW_WIDTH/2 - minW*2-1)
            wx1 = np.random.randint(wx0 + minW, self.WINDOW_WIDTH/2 - minW)
            wx2 = np.random.randint(self.WINDOW_WIDTH/2 + minW, self.WINDOW_WIDTH - minW)
            wx3 = np.random.randint(wx2 + minW, self.WINDOW_WIDTH)
            wy0 = 400
            wy2 = np.random.randint(100, self.WINDOW_HEIGHT/2)
            wy1 = np.random.randint(200, wy0 - minW/3)

            self.envPoints = np.array([(wx1, wy0), (wx1, wy1), (wx0, wy1), (wx0, wy2), (wx3, wy2), (wx3, wy1), (wx2, wy1), (wx2, wy0)])
            self.envWallSide = ['l', 'b','l', 't', 'r', 'b','r','b']
        # this is a turn to the right
        elif envNr == 4:
            wx0 = np.random.randint(0,self.WINDOW_WIDTH/2 - minW)
            wx2 = np.random.randint(self.WINDOW_WIDTH/2 + 2*minW+1, self.WINDOW_WIDTH)
            wx1 = np.random.randint(self.WINDOW_WIDTH/2 + minW, wx2 - minW/2)

            wy0 = 400
            wy2 = np.random.randint(100, self.WINDOW_HEIGHT/2)
            wy1 = np.random.randint(200, wy0 - minW/3)

            self.envPoints = np.array([(wx0, wy0), (wx0, wy2), (wx2, wy2), (wx2, wy1), (wx1, wy1), (wx1, wy0)])
            self.envWallSide = ['l', 't','r', 'b', 'r', 'b']
        else:
            raise NameError('envNr was out of range, no such environment defined')

        self.envWalls = self.pointsTOWalls(self.envPoints)

        return
#
#sim = SimulationEnvironment(True)
#sim.reset()
#sim.markerDetector.getAnglesFromWebcam()