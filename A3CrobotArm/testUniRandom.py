# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:38:47 2018

@author: Alex
"""
import numpy as np
import scipy.signal

#d1 = np.random.randint(1, 6 + 1)
#print(d1)

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def getHalfLifeInSteps(gamma):
    return np.log(0.5)/np.log(gamma)

WallR = -100 # was -100
GoalR = 100  # was 200
distGR = -0.5 #-50# distance to goal reward rond het begin

#fast to wall
rewards1 = np.concatenate((np.repeat(distGR,3),np.array([distGR+0.5*WallR,distGR+WallR])),axis=None)
# slower to wall
rewards2 = np.concatenate((np.repeat(distGR,6),np.array([distGR+0.5*WallR,distGR+WallR])),axis=None) 
# a decent tracejtory to goal
p1 = np.repeat(distGR,13)
p2 = np.repeat(distGR*0.6,4)
p3 = np.repeat(distGR*0.2,4)
p4 = np.array([GoalR])
rewards3 = np.concatenate((p1,p2,p3,p4),axis=None) 
# a larger trajectory to goal with a near miss
p1 = np.repeat(distGR,13)
p2 = np.array([distGR*0.6+0.5*WallR,distGR*0.6+0.5*WallR,distGR*0.6+0.6*WallR]) # near miss
p3 = np.repeat(distGR*0.6,4)
p4 = np.repeat(distGR*0.2,4)
p5 = np.array([GoalR])
rewards4 = np.concatenate((p1,p2,p3,p4,p5),axis=None) 


#np.array([distGR,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-40,-40,-20,-20,-10, -10, 200])

#rewards4 = np.array([-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-40,-40,-80,-80,-40,-40,-40,-20,-20,-10, -10, 200])
gamma = 0.99
np.set_printoptions(precision=1)#, suppress=True)
#print(rewards1)
print(discount(rewards1,gamma)[0], "fast to wall") 
print(discount(rewards2,gamma)[0], "slower to wall") 
print(discount(rewards3,gamma)[0], "decent trajectory to goal")
print(discount(rewards4,gamma)[0], "near miss trajectory to goal")
    
    #Wat wil ik zien:
    #hoogste Reward, dus minste - op volgorde:
#    1. decent Traject
#    2. near miss Traject
#    3. slower to wall
#    4. fast to wall
    
#    Wat zie ik met de huidge pars?
#    Deze:
#    WallR = -100
#    GoalR = 200
#    distGR = -50# distance to goal reward rond het begin
#     1. fast to wall
#     2. slow to wall
#     3. decent traject
#     4. with near miss
    #Conclusie: Dit is precies wat je niet wilt
    
    #    Wat zie ik met distGR op 0? dus geen zetje om richting het doel te gaan
#    Deze:
#    WallR = -100
#    GoalR = 200
#    distGR = 0# distance to goal reward rond het begin
#     1. decent Traject #80
#     2. near miss # -60
#     3. slow to wall # -140
#     4. fast to wall # -144
#    Conclusie: Dit is in principe wat je wilt maar het is zonder een zetje
    
    #    Wat ik zie met een kleine distGR
#    Deze:
#    WallR = -100
#    GoalR = 200
#    distGR = -1# distance to goal reward rond het begin
#     1. decent Traject #65
#     2. near miss # -76
#     3. slow to wall # -148
#     4. fast to wall # -149
#    Conclusie: Dit is wat je wilt! Hopelijk gaat het goed
    
# This is to test wat the max time discounted reward is for standing still
p1 = np.repeat(distGR,10000) # stand still at max distance forever
p2 = np.repeat(distGR,1000)  # almost forever
p3 = np.repeat(distGR,160)   # stand still  for some steps
gamma = 0.99
#gamma = 0.99 has no effect after around 500 steps or so
# half life, 0.99^steps = 0.5 is at steps = 70
#gamma = 0.96 has no effect after around 100 steps or so
# half life, 0.96^steps = 0.5 is at steps = 17

# Distances list to goal in pixels: (perfect tracejtory)
ds = np.array(list(range(160,-1,-1)))
g = -0.01
offset = 0.6
#0.5*e^(0.015*x)-0.5
# returns an array of rewards with respect to the distances ds
def getGR(ds,g,offset):
        return offset * np.exp((g * ds)) - offset

with np.printoptions(precision=1, suppress=True):
    print("\n rewards for standing still")
    print(discount(p1,gamma)[0]) # = -50
    print(discount(p2,gamma)[0]) # = -50
    print(discount(p3,gamma)[0]) # = -25
    N = 20
    print("\n reward for hitting the wall after ",N," steps")
    rewards1 = np.concatenate((np.repeat(distGR,N),np.array([distGR+WallR])),axis=None)
    print(discount(rewards1,gamma)[0]) # fast to wall
    #print("Get half life of gamma:")
    #print(getHalfLifeInSteps(gamma))
    #Conclusion: Standing still forever, with a max reward of -1 gives a max time discounted reward of -100
    #So we need the negative reward of hitting a wall higher than 100 to motivate simply staying alive.
    print("the discounted reward for a nice tracejtory using the current exp function\n(Without a GoalR at the end!))")
    rs = getGR(ds,g,offset)
    print(discount(rs,gamma)[0])
    

