#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:47:10 2021

@author: sarah
"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.special import gamma

def Beta_func(alpha):
    B = gamma(alpha[0])*gamma(alpha[1])/gamma(alpha[0]+alpha[1])
    return B

def beta(x,alpha=[1,1]):
    y = x**(alpha[0]-1)*(1-x)**(alpha[1]-1) / Beta_func(alpha)
    return y

nb = 4
epsilon = 0.1
rew_probs = np.ones(nb) / 2
rew_probs[0] += epsilon
lamb = 0.2

choices = [1,1,3,2,3,0,2,1,3,0]
# First set up the figure, the axis, and the plot element we want to animate
fig, axes = plt.subplots(1, nb, figsize=(15,3), sharey=True)

line1, = axes[0].plot([], [], lw=3)
line2, = axes[1].plot([], [], lw=3)
line3, = axes[2].plot([], [], lw=3)
line4, = axes[3].plot([], [], lw=3)
lines = [line1, line2, line3, line4]

axes[0].set_ylabel("Beta distribution", fontsize=14, color="white")

fig.patch.set_alpha(0.0)

alpha = np.ones((nb,2))

#lines = []

for j,ax in enumerate(axes):
    #lines.append(ax.plot([], [], lw=2))
    
    ax.patch.set_alpha(0.0)

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,2.5)
    
    ax.set_xlabel("$p_r$ arm "+str(j+1), fontsize=14, color="white")
    ax.set_title("Give ME Space")
    
    plt.setp(ax.get_xticklabels(), Fontsize=12)
    plt.setp(ax.get_yticklabels(), Fontsize=12)
    plt.setp(ax.spines.values(), linewidth=2)
    

plt.tight_layout()

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    #lines[1].set_data([], [])
    return lines,

# animation function.  This is called sequentially
def animate(i):
    if i>0 and i%2==1:
        choice = np.random.choice(nb)
        c = np.random.rand()
        if c<= rew_probs[choice]:
            r = 0
        else:
            r = 1
        string = "choice: arm "+str(choice+1)+", reward: "+str(int(not r))
    else:
        string = ""
        choice = None
    
    if choice is not None:
        axes[choice].set_title(string, fontsize=14, color="white")
    x = np.linspace(0, 1, 1000)
    for j,line in enumerate(lines):
        y = beta(x,alpha=alpha[j])
        line.set_data(x, y)
        if choice is None:
            axes[j].set_title(" ")
    #lines[1].set_data(x, y)
    if i>0 and i%2==1:            
        alpha[:,:] = (1-lamb)*alpha[:,:] + 1 - (1-lamb)
        alpha[choice,r] += 1
        

    return lines,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=21, interval=1500)#, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
if lamb > 0:
    anim.save('forgetting_bandit.gif', fps=0.75, dpi=200)#, extra_args=['-vcodec', 'libx264'])
else:
    anim.save('stationary_bandit.gif', fps=0.75, dpi=200)#, extra_args=['-vcodec', 'libx264'])
#plt.show()
