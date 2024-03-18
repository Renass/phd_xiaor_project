import h5py
import os 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Arrow
from tf.transformations import euler_from_quaternion
import json

'''
Check one trajectory from dataset as a slide show
camera_image-map(costmap)-action slide show
'''

FILENAME = 'TSA_dataset/nav/tsa-trajs_2024-03-13_20-24-44.h5'

def update(frame):
    ax1.clear()
    ax2.clear()
    
    ax1.imshow(im[frame])
    ax1.axis('off')
    ax1.set_title('Camera Image')
 
    ax2.imshow(np.flipud(maps[frame]), cmap='gray_r')
    ax2.imshow(np.flipud(costmaps[frame]), cmap='gray_r', alpha=0.7) 
    ax2.axis('off')
    ax2.set_title('Map')


    map_pose = world_to_map(
        (pose[frame][0], pose[frame][1]), 
        mapinfo['resolution'], 
        (mapinfo['origin']['position']['x'], 
        mapinfo['origin']['position']['y'])
    )
    quaternion = [0, 0, pose[frame][2], pose[frame][3]]
    _, _, yaw = euler_from_quaternion(quaternion)
    yaw = -1*yaw
    arrow_length = 100
    dx = arrow_length * np.cos(yaw)
    dy = arrow_length * np.sin(yaw) 
    #arrow = Arrow(pose[frame][0], pose[frame][1], dx, dy, width=300, color='red')
    arrow = Arrow(map_pose[0], map_pose[1], dx, dy, width=100, color='red')
    ax2.add_patch(arrow)
    
    new_action_text = f'Action: {actions[frame]}' if frame < len(actions) else 'Final State'
    action_text.set_text(new_action_text)

def world_to_map(pose, resolution, origin):
    """
    Convert world coordinates to map pixel coordinates.
    
    :param pose: The pose in world coordinates (x, y).
    :param resolution: The map resolution (meters per pixel).
    :param origin: The origin of the map in world coordinates (x, y).
    :return: The pose in map pixel coordinates.
    """
    map_x =  int((pose[0] - origin[0]) / resolution)
    map_y = mapinfo['height'] - int((pose[1] - origin[1]) / resolution)
    return (map_x, map_y)



with h5py.File(FILENAME, 'r') as file:
    im = file['states']['data_0'][:]
    actions = file['actions']['data_0'][:]
    maps = file['maps']['data_0'][:]
    costmaps = file['costmaps']['data_0'][:]
    pose = file['pose']['data_0'][:]

if im.dtype == np.float32 or im.dtype == np.float64:
    im = (im - im.min()) / (im.max() - im.min())

mapinfo_filename = f"{os.path.splitext(FILENAME)[0]}_mapinfo.json"
with open(mapinfo_filename, 'r') as file:
    mapinfo = json.load(file)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
action_text = fig.text(0.5, 0.05, '', ha='center', va='center', fontsize=12, color='red')
ani = animation.FuncAnimation(fig, update, frames= len(im), repeat=False, interval=9000)
plt.show()