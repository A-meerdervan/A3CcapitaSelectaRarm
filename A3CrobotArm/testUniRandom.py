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

WallR = -1000 # was -100
GoalR = 200  # was 200
distGR = -50 #-50# distance to goal reward rond het begin

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
with np.printoptions(precision=1, suppress=True):
    #print(rewards1)
    print(discount(rewards1,gamma)[0]) #-322 fast to wall
    print(discount(rewards2,gamma)[0]) #-415 slower to wall
    print(discount(rewards3,gamma)[0]) #-568 decent traject
    print(discount(rewards4,gamma)[0]) #-810 near miss traject
    
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
#     1. decent Traject #160
#     2. near miss #18
#     3. slow to wall # -140
#     4. fast to wall # -144
#    Conclusie: Dit is in principe wat je wilt maar het is zonder een zetje
    
    #    Wat ik zie met een grote neg reward voor de wall = -1000
#    Deze:
#    WallR = -1000
#    GoalR = 200
#    distGR = -50# distance to goal reward rond het begin

    

