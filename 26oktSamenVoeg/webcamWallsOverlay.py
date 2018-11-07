

import numpy as np
import SimulationEnvironment as sim
import matplotlib.pyplot as plt
from scipy.misc import imread
import matplotlib.cbook as cbook

def pltLine(p1,p2,color='r'):
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linestyle= (0, ()), linewidth=4, color=color)
def plotWalls(envWalls,color='r'):        
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
# plot the result
plotWalls(webcamAdjustedEnv,'b')

#datafile = cbook.get_sample_data('testImg1.png')
img = imread('testImg1.png')
plt.imshow(img)#, zorder=0, extent=[0.5, 8.0, 1.0, 7.0])

# Plot a line which shows the lower Y goal threshold
#plt.plot([0,400],[280,280],linestyle= (0, ()), linewidth=4, color='black')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
#plt.set_xlim(0, 400)
plt.xlim([0,400])
plt.ylim([400, 0])
plt.axis('equal')
plt.grid(True)
plt.ylabel("y in pixels"); plt.xlabel("x in pixels"); #plt.legend()
plt.title("end effector heatmap for en") ; plt.show() 