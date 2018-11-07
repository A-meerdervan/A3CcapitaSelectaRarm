

import numpy as np
import SimulationEnvironment as sim
import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook
import cv2
import gobalConst as cn

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
def plotGoal(img,goalLoc,goalRadius,color=(0,255,0)):
    # the -1 at the end means that it fills it up.
    cv2.circle(img,tuple(goalLoc.astype(int)), int(goalRadius), color, -1)
    
def fromSimToWebcamCoord(simCoord,factorSimToReal,offset):
    # Do the transformation
    scaledCoord = simCoord * factorSimToReal
    webcamCoord = scaledCoord + offset
    return webcamCoord
    

WH = 400
WW = 400

# From inspecting a webcam image with the arm at angle 1 = 90 degrees a2,a3 = 0,0
# Done with testImg1
Xwc00 = 306
Ywc00 = 338
Pwc00 = np.array([Xwc00,Ywc00])
Xwcee = 291
Ywcee = 45
webcamArmLength = Ywc00 - Ywcee

# From the values used in the simulation on 6 november:
bodyFactor = 0.5 #0.5 after 5nov, Was at 0.9 on 3 nov; used to make the body verhoudingen match our envs.
rob_JointLenght = bodyFactor*np.array([71,112,141*0.6,141*0.4]) #[100,100,80,20] #100,100,80,20
rob_JointLenght = rob_JointLenght.astype(int)
#rob_JointWidth = int(bodyFactor*30) # 30 mm wide
simArmLength = np.sum(rob_JointLenght)
# This factor is used in the conversion from sim coordinates to webcam coordinates
factorSimToReal = webcamArmLength/simArmLength

# get an env
env = sim.SimulationEnvironment()
env.getEnv(5)
envWalls = env.envWalls
env.createRandomGoal()
# This is the base point in the simulation
Xsim00 = WW/2 ; Ysim00 = WH
Psim00 = np.array([Xsim00,Ysim00])

wcWH = 480 # webcam view width in pixels
wcWW = 640 # webcam view height in pixels
# plot heatmap:
figName = "Heatmap end effector"
fig = plt.figure()
#plt.scatter(Xpts, Ypts,s=1)
# Plot the env walls before conversion
plotWalls(envWalls)

# plot the webcam contour:
webcamContour = getWalls(np.array([(0,wcWH),(0,0),(wcWW,0),(wcWW,wcWH)]))
plotWalls(webcamContour,'b')
# plot the simulation contour:
webcamContour = getWalls(np.array([(0,WH),(0,0),(WW,0),(WW,WH)]))
plotWalls(webcamContour)
# plot the base point of the arm in webcamspace
plt.plot(Xwc00,Ywc00,'bo')
# plot the base point of the arm in webcamspace
plt.plot(WH/2,WH,'bo')

# Do the transformation
scaledEnv = envWalls * factorSimToReal
#plotWalls(scaledEnv,'b')
offset = Pwc00 - Psim00*factorSimToReal
webcamAdjustedEnv = scaledEnv + offset


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


# plot the result
img = imread('testImg1.png')
color = (0,0,255)
lineThicknes = 5
plotWallsImg(img,webcamAdjustedEnv,color,lineThicknes)
simGoalLoc = env.createRandomGoal()
print('goalLoc sim',simGoalLoc)
webcamGoal = fromSimToWebcamCoord(simGoalLoc,factorSimToReal,offset)
goalRadius = cn.sim_goalRadius*factorSimToReal
goalColor = (0,255,0)
print(webcamGoal,'webcamgoal')
plotGoal(img,webcamGoal,goalRadius,goalColor)
while (True):
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()