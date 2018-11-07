

import numpy as np
import SimulationEnvironment as sim
import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook
import cv2
import gobalConst as cn
import MarkerDetector as cam

def pltLine(p1,p2,color='r'):
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linestyle= (0, ()), linewidth=4, color=color)
def plotWalls(envWalls,color='r',realSetup=False):        
    for wall in envWalls:
        p1 = wall[0]
        p2 = wall[1]
        pltLine(p1,p2,color)
# Take a set of connected points that go clockwise from left bottom to right bottom
# and convert it to a valid wall construct. points = np.array([(x1,y1),(x2,y2),...])
def getWalls(points):
    walls = []
    startPoint = points[0]; curEndPoint = startPoint
    for point in points[1:]:
        walls.append([curEndPoint,point])
        curEndPoint = point
    walls.append([points[-1],startPoint])
    return np.array(walls)

def pltLineImg(img,p1,p2,color=(255,0,0),lineThicknes=5):   
    # The point must be a tuple: (x,y)
    cv2.line(img,tuple(p1.astype(int)),tuple(p2.astype(int)),color,lineThicknes)

def plotWallsImg(img,envWalls,color=(255,0,0),lineThicknes=5):        
    for wall in envWalls:
        p1 = wall[0]
        p2 = wall[1]
        pltLineImg(img,p1,p2,color,lineThicknes)
def plotGoalImg(img,goalLoc,goalRadius,color=(0,255,0)):
    # the -1 at the end means that it fills it up.
    cv2.circle(img,tuple(goalLoc.astype(int)), int(goalRadius), color, -1)
    
def fromSimToWebcamCoord(simCoord,factorSimToReal,Pwc00):
    offset = Pwc00 - Psim00*factorSimToReal
    # Do the transformation
    scaledCoord = simCoord * factorSimToReal
    webcamCoord = scaledCoord + offset
    return webcamCoord

def markersToArmLength(markerLocs):
    length1 = np.sqrt(np.sum((markerLocs[0] - markerLocs[1])**2))
    length2 = np.sqrt(np.sum((markerLocs[1] - markerLocs[2])**2))
    length3 = np.sqrt(np.sum((markerLocs[2] - markerLocs[3])**2))
    return length1 + length2 + length3

"""Completely invariant"""
# This is the base point in the simulation
Xsim00 = cn.sim_WINDOW_WIDTH/2 ; Ysim00 = cn.sim_WINDOW_HEIGHT
Psim00 = np.array([Xsim00,Ysim00])

# From the values used in the simulation on 6 november:
bodyFactor = 0.5 #0.5 after 5nov, Was at 0.9 on 3 nov; used to make the body verhoudingen match our envs.
rob_JointLenght = bodyFactor*np.array([71,112,141*0.6,141*0.4]) #[100,100,80,20] #100,100,80,20
rob_JointLenght = rob_JointLenght.astype(int)
simArmLength = np.sum(rob_JointLenght)

"""invariant within one session"""
def getImgWithWallOverlay(img,envWalls,simGoalLoc,markers):
    # this calculates the full arm length by taking a sum of the distances between markers
    webcamArmLength = markersToArmLength(markers)
    # This factor is used in the conversion from sim coordinates to webcam coordinates
    factorSimToReal = webcamArmLength/simArmLength
    ## Do the transformation
    Pwc00 = markers[0] # The base point in the webcam coordinates
    # Get the webcam coordinates for the wall points
    webcamAdjustedEnv = fromSimToWebcamCoord(envWalls,factorSimToReal,Pwc00)
    # Overlay the walls on the image
    colorWalls = (0,0,255) # RGB value (0,0,255)=red
    wallThickness = 5 # in pixels
    plotWallsImg(img,webcamAdjustedEnv,colorWalls,wallThickness)
    # Overlay the goal on the image
    webcamGoal = fromSimToWebcamCoord(simGoalLoc,factorSimToReal,Pwc00)
    goalRadius = cn.sim_goalRadius*factorSimToReal
    goalColor = (0,255,0) # this is green
    plotGoalImg(img,webcamGoal,goalRadius,goalColor)
    return img

#wcWH = 480 # webcam view width in pixels
#wcWW = 640 # webcam view height in pixels
# plot heatmap:
#figName = "Heatmap end effector"
#fig = plt.figure()
#plt.scatter(Xpts, Ypts,s=1)
# Plot the env walls before conversion
#plotWalls(envWalls)

# plot the webcam contour:
#webcamContour = getWalls(np.array([(0,wcWH),(0,0),(wcWW,0),(wcWW,wcWH)]))
#plotWalls(webcamContour,'b')
# plot the simulation contour:
#webcamContour = getWalls(np.array([(0,WH),(0,0),(WW,0),(WW,WH)]))
#plotWalls(webcamContour)
# plot the base point of the arm in webcamspace
#plt.plot(Xwc00,Ywc00,'bo')
# plot the base point of the arm in webcamspace
#plt.plot(WH/2,WH,'bo')


#plotWalls(scaledEnv,'b')


## Do the transformation
#scaledEnv = envWalls * factorSimToReal
#offset = Pwc00 - Psim00*factorSimToReal
#webcamAdjustedEnv = scaledEnv + offset


#for i in range(1):
#    # plot the result
#    plotWalls(webcamAdjustedEnv,'b')
#    
#    #datafile = cbook.get_sample_data('testImg1.png')
#    img = imread('testImg1.png')
#    plt.imshow(img)#, zorder=0, extent=[0.5, 8.0, 1.0, 7.0])
#    
#    # Plot a line which shows the lower Y goal threshold
#    #plt.plot([0,400],[280,280],linestyle= (0, ()), linewidth=4, color='black')
#    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
#    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
#    #plt.set_xlim(0, 400)
#    plt.xlim([0,400])
#    plt.ylim([400, 0])
#    plt.axis('equal')
#    plt.grid(True)
#    plt.ylabel("y in pixels"); plt.xlabel("x in pixels"); #plt.legend()
#    plt.title("end effector heatmap for en") ; plt.show() 


