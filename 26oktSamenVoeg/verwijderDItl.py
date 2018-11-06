import numpy as np

rC = 200 # robotCenter, the x pos of the arm bottom        
#envNr = 7

# This is a pipe witch a corner to the left (most used during training. Our first env.)
pR = 80 # the width of the pipe
envWalls = np.array([[(rC-pR,400), (rC-pR,300)], [(rC-pR,300), (40,300)], [(40,300),
          (40,140)], [(40,140), (rC+pR,140)], [(rC+pR,140), (rC+pR,400)], [(rC+pR,400),(rC-pR,400)]])
print(envWalls)
envWallSide = ['l', 'b','l', 't', 'r', 'b']

points = np.array([(rC-pR,400), (rC-pR,300),(40,300),
          (40,140), (rC+pR,140), (rC+pR,400)])
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
print(getWalls(points))
        
    
    