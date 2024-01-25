import h5py
import os 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''
Check one trajectory from dataset as a video
'''

save_dir = 'sa-traj_dataset'
filename = 'C:\pythonprogv2\phd_xiaor_project\sa-traj_dataset\sa-trajs2023-11-08_14-31-15.h5'
file = h5py.File(filename, 'r')
#matplotlib.use('TkAgg') 

def update(frame):
    plt.clf()
    plt.imshow(im[frame].transpose(1, 2, 0))
    plt.axis('off')

im = file['states']['data'][67]
print(im.shape)
fig = plt.figure()
ani = animation.FuncAnimation(fig, update, frames= 100, repeat=False, interval=100)
plt.show()