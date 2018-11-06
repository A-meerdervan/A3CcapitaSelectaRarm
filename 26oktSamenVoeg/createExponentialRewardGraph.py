# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:19:34 2018

@author: Alex
"""
import numpy as np

import matplotlib.pyplot as plt

totalDots = 2000
d = np.linspace(0,270,totalDots)
#sim_expRewardGamma = -0.01
rGamma1 = -0.01
rGamma2 = -0.022
rGamma3 = -0.03
offset = 100
#sim_expRewardOffset = 100
r1 = offset * np.exp(rGamma1 * d) - offset
r2 = offset * np.exp(rGamma2 * d) - offset
r3 = offset * np.exp(rGamma3 * d) - offset
# plot heatmap:
figName = "The exponential relative reward set out against the relative distance."
fig = plt.figure()
plt.plot(d, r1, 'C7',label='gamma = 0.01')
plt.plot(d, r2, 'C5',label='gamma = 0.022')
plt.plot(d, r3, 'C6',label='gamma = 0.03')

# Plot a line which shows the lower Y goal threshold
#plt.plot([0,400],[280,280],linestyle= (0, ()), linewidth=4, color='black')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
#plt.set_xlim(0, 400)
#plt.xlim([0,400])
#plt.ylim([400, 0])
plt.axis('equal')
plt.grid(True)
plt.ylabel("Reward"); plt.xlabel("distance in pixels"); plt.legend()
plt.title(figName) ; plt.show() 