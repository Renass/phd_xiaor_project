import h5py
import os 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

'''
Check one trajectory from dataset as a slide show
'''

FILENAME = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/tsa-trajs2024-03-10_20-39-48.h5'

def update(frame):
    plt.clf()
    plt.imshow(im[frame])
    plt.axis('off')
    # Check if the current frame has a corresponding action
    if frame < len(actions):
        action_text = f'Action: {actions[frame]}'
    else:
        action_text = 'Final State'
    plt.text(0.5, -0.1, action_text, ha='center', va='center', transform=ax.transAxes, fontsize=12)



with h5py.File(FILENAME, 'r') as file:
    im = file['states']['data_0'][:]
    actions = file['actions']['data_0'][:]

if im.dtype == np.float32 or im.dtype == np.float64:
    im = (im - im.min()) / (im.max() - im.min())

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update, frames= len(im), repeat=False, interval=1000)
plt.show()