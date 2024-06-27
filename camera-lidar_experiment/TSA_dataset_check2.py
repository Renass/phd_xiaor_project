import h5py
import os 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Arrow
from tf.transformations import euler_from_quaternion
#from transforms3d.euler import euler_from_quaternion

import json

'''
Check one trajectory from dataset as a slide show
camera_image-map(costmap)-action slide show
'''

FILENAME = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/cola/tsa_combined.h5'
EPISODE_NUMBER = 0
#pause before slides
INTERVAL = 3000

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
    #print(map_pose)
    quaternion = [0, 0, pose[frame][2], pose[frame][3]]
    _, _, yaw = euler_from_quaternion(quaternion)
    yaw = -1*yaw
    arrow_length = 100
    dx = arrow_length * np.cos(yaw)
    dy = arrow_length * np.sin(yaw) 
    #arrow = Arrow(pose[frame][0], pose[frame][1], dx, dy, width=300, color='red')
    arrow = Arrow(map_pose[0], map_pose[1], dx, dy, width=100, color='blue')
    ax2.add_patch(arrow)
    
    new_action_text = f'Action: {actions[frame]}' if frame < len(actions) else 'Final State'
    action_text.set_text(new_action_text)

    if frame<len(actions):
        action_pose = world_to_map(
                    (actions[frame][0], actions[frame][1]), 
            mapinfo['resolution'], 
            (mapinfo['origin']['position']['x'], 
            mapinfo['origin']['position']['y'])
        )
        action_quaternion = [0, 0, actions[frame][2], actions[frame][3]]
        _, _, action_yaw = euler_from_quaternion(action_quaternion)
        action_yaw = -1*action_yaw
        a_dx = arrow_length * np.cos(action_yaw)
        a_dy = arrow_length * np.sin(action_yaw)
        a_arrow = Arrow(action_pose[0], action_pose[1], a_dx, a_dy, width=100, color='red')
        ax2.add_patch(a_arrow)

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
    im = file['states']['data_'+str(EPISODE_NUMBER)][:]
    actions = file['actions']['data_'+str(EPISODE_NUMBER)][:]
    maps = file['maps']['data_'+str(EPISODE_NUMBER)][:]
    costmaps = file['costmaps']['data_'+str(EPISODE_NUMBER)][:]
    pose = file['pose']['data_'+str(EPISODE_NUMBER)][:]

if im.dtype == np.float32 or im.dtype == np.float64:
    im = (im - im.min()) / (im.max() - im.min())

mapinfo_filename = f"{os.path.splitext(FILENAME)[0]}_mapinfo.json"
with open(mapinfo_filename, 'r') as file:
    mapinfo = json.load(file)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
action_text = fig.text(0.5, 0.05, '', ha='center', va='center', fontsize=12, color='red')
ani = animation.FuncAnimation(fig, update, frames= len(im), repeat=False, interval=INTERVAL)
plt.show()