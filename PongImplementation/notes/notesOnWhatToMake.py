# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:08:02 2018

@author: Alex
"""

Just for notes

Inside the Worker class:
# init env at the top of the class
self.env = gym.make('Pong-v0')

    Inside the work() function
    # Reset the environment to the start state
    # This means this reset funct should return the start state
    s = self.env.reset()
        Inside the (while not done) loop at the start
        # Render the evironment
        if self.number == 0:
            self.env.render()
        Inside the same while loop
        # Now perform an action and obtain the results from the env.
        # d is a bool which is true only if a terminal state has been reached
        # i is not used I think. But it might be the current timestep in the
        # environment. 
        s1,r,d, i = self.env.step(self.actions[a])
    