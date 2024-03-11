import h5py
import os 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

'''
Check one trajectory from dataset as a slide show
camera_image-map-action slide show
'''

FILENAME = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/tsa-trajs2024-03-11_20-31-12.h5'

def update(frame):
    ax1.clear()
    ax2.clear()
    
    ax1.imshow(im[frame])
    ax1.axis('off')
    ax1.set_title('Camera Image')

    ax2.imshow(maps[frame], cmap='gray')
    ax2.axis('off')
    ax2.set_title('Map')
    
    new_action_text = f'Action: {actions[frame]}' if frame < len(actions) else 'Final State'
    action_text.set_text(new_action_text)



with h5py.File(FILENAME, 'r') as file:
    im = file['states']['data_0'][:]
    actions = file['actions']['data_0'][:]
    maps = file['maps']['data_0'][:]

if im.dtype == np.float32 or im.dtype == np.float64:
    im = (im - im.min()) / (im.max() - im.min())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
action_text = fig.text(0.5, 0.05, '', ha='center', va='center', fontsize=12, color='red')
ani = animation.FuncAnimation(fig, update, frames= len(im), repeat=False, interval=1000)
plt.show()