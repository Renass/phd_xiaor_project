import h5py
import os 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''
Check one trajectory from dataset as a video
'''

FILENAME = '/home/renas/pythonprogv2/phd_xiaor_project/sa-traj_dataset/real_pink_gates/sa-trajs2024-01-29_19-59-45.h5'

def update(frame):
    plt.clf()
    plt.imshow(im[frame].transpose(1, 2, 0))
    plt.axis('off')
    plt.text(0.5, -0.1, f'Action: {actions[frame]}', ha='center', va='center', transform=ax.transAxes, fontsize=12)

save_dir = 'sa-traj_dataset'
file = h5py.File(FILENAME, 'r')

im = file['states']['data'][0]
actions = file['actions']['data'][0]
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update, frames= len(im), repeat=False, interval=500)
plt.show()