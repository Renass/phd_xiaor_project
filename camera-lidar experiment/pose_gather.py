import trajectories_gather6
import rospy
from geometry_msgs.msg import PoseStamped
from diagnostic_msgs.msg import KeyValue
import threading
import time
import os
from datetime import datetime
import h5py
import numpy as np
import json

'''
Go through predifned poses on lidar map for future creating the
text-image token from text-image model backbone for future classification

import trajectories_gather6.py
'''

#IMAGE_TOPIC = '/camera/rgb/image_raw'
IMAGE_TOPIC = '/image_raw'
#MAP_SERVICE = '/dynamic_map'
MAP_SERVICE = '/static_map'
ACTION_ROSTOPIC = '/move_base_simple/goal'

SAVE_DIR = 'TSA_dataset/nav/sim'

#Sim 2A724_x3.yaml
POSES = [ 
    [14.9, 5.7, 0.14, 0.99],
    [15.30, 21.31, -0.53, 0.85]
]

DESCRIPTIONS = [
    'position facing fridge and rack',
    'position out of the 2A724 lab facing right side of the corridor'
]

def publish_pose(publisher, action):
    goal_msg = PoseStamped()
    #goal_msg.header.stamp = rospy.Time.now() 
    goal_msg.header.frame_id = 'map'
    goal_msg.pose.position.x = action[0]
    goal_msg.pose.position.y = action[1]
    goal_msg.pose.position.z = 0

    goal_msg.pose.orientation.x = 0
    goal_msg.pose.orientation.y = 0
    goal_msg.pose.orientation.z = action[2]
    goal_msg.pose.orientation.w = action[3]

    #print(goal_msg)
    publisher.publish(goal_msg)

def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass

def work_thread():
    for i, pose in enumerate(POSES):
        time.sleep(1)
        if traj_buffer.waiting == 'task':
            publish_pose(driv_pub, pose)
            time.sleep(1)
            print('going to the pose to explore')
            while traj_buffer.nav_status != 3:
                time.sleep(1)
            task_msg = KeyValue()
            task_msg.key = 'new_task'
            task_msg.value = DESCRIPTIONS[i]
            task_pub.publish(task_msg)
            while traj_buffer.waiting != 'action':
                time.sleep(1)
            publish_pose(driv_pub, pose)
            time.sleep(1)
            while traj_buffer.nav_status != 3:
                time.sleep(1)
            time.sleep(8)
            task_msg.key = 'end_task'
            task_msg.value = 'done'
            task_pub.publish(task_msg)
            print('Task done message sent')

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_filename = os.path.join(SAVE_DIR, f'poses_{current_datetime}.h5')
    txt_filename = os.path.join(SAVE_DIR, f'poses_{current_datetime}_tasks.txt')
    mapinfo_filename = os.path.join(SAVE_DIR, f'poses_{current_datetime}_mapinfo.json')


    with h5py.File(save_filename, 'w') as hf:
        # Create HDF5 datasets for states and actions
        states_group = hf.create_group('states')
        actions_group = hf.create_group('actions')
        map_group = hf.create_group('maps')
        costmap_group = hf.create_group('costmaps')
        pose_group = hf.create_group('pose')

        for i, states_trajectory in enumerate(traj_buffer.states_buffer):
            state = np.stack(states_trajectory, axis=0)
            states_group.create_dataset('data_'+str(i), data=state, dtype = np.float32, compression = 'gzip')

        for i, actions_trajectory in enumerate(traj_buffer.actions_buffer):
            action = np.stack(actions_trajectory, axis=0)
            actions_group.create_dataset('data_'+str(i), data=action, dtype = np.float32, compression = 'gzip')

        for i, maps_trajectory in enumerate(traj_buffer.map_buffer):
            map = np.stack(maps_trajectory, axis=0)
            map_group.create_dataset('data_'+str(i), data=map, dtype = np.float32, compression = 'gzip')

        for i, costmaps_trajectory in enumerate(traj_buffer.costmap_buffer):
            costmap = np.stack(costmaps_trajectory, axis=0)
            costmap_group.create_dataset('data_'+str(i), data=costmap, dtype = np.float32, compression = 'gzip')

        for i, pose_trajectory in enumerate(traj_buffer.pose_buffer):
            pose = np.stack(pose_trajectory, axis=0)
            pose_group.create_dataset('data_'+str(i), data=pose, dtype = np.float32, compression = 'gzip')

    with open(txt_filename, 'w') as txt_file:
        for task in traj_buffer.task_buffer:
            if 'new_task' in task:
                txt_file.write(f'{task["new_task"]}\n')


    with open(mapinfo_filename, 'w') as txt_file:
        json.dump(traj_buffer.map_info, txt_file, indent=4)
        #print(traj_buffer.map_info, file=txt_file)
    
    print('Buffer saved')

if __name__ == '__main__':
    traj_buffer = trajectories_gather6.TrajectoryBuffer(
        image_topic= IMAGE_TOPIC,
        map_service= MAP_SERVICE,
        buffer_size= len(POSES),
        always= False
    )

    driv_pub = rospy.Publisher(ACTION_ROSTOPIC, PoseStamped, queue_size=1)
    task_pub = rospy.Publisher('/task', KeyValue, queue_size=1)

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=work_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()

    t2.join()
    rospy.signal_shutdown('record finished')