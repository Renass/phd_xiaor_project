import trajectories_gather6
import threading
import rospy
import numpy as np
import time
import os
from datetime import datetime
import h5py 
import json
from geometry_msgs.msg import PoseStamped
from diagnostic_msgs.msg import KeyValue
import torch

'''
Send robot from starting point to target and
Write a Task-State-Action dataset as a HDF5 file with gzip 

imports: trajectories_gather6

waiting queue: 'task': ['state', 'state.costmap', 'action', 'status']

State: im, map, costmap, pose, mapinfo, prompt

Actions in ros: position(x,y) orientation quternions (z, w)

new task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'new_task', value: 'go left'}"
end task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'end_task', value: 'done'}"
'''

BUFFER_SIZE = 5
SAVE_DIR = 'TSA_dataset/nav/real/test'

IMAGE_TOPIC = '/camera/rgb/image_raw'
#IMAGE_TOPIC = '/image_raw'

#MAP_SERVICE = '/dynamic_map'
MAP_SERVICE = '/static_map'
ACTION_ROSTOPIC = '/move_base_simple/goal'

PROMPT = 'Go to the fridge'

#Sim 2A724_x3.yaml
#TARGET = [14.9, 5.7, 0.14, 0.99]
#STARTING_POINTS = [
#    [15.9, 22.2, -0.97, 0.26],
#    [19.0, 15.1, 0.85, 0.52],
#    [0.0, 11.3, 0.27, 0.96],
#    [10.56, 4.60, 0.03, 1.00],
#    [8.23, 17.02, 0.02, 1.0]
#]



# Real 2A724_april.yaml 
TARGET = [-1.83, 5.45, 0.83, 0.56]
#STARTING_POINTS = [
#    [-7.05, 6.51, -0.58, 0.82],
#    [-5.77, 4.33, 0.22, 0.98],
#    [-5.13, 2.73, 0.86, 0.51],
#    [-11.28, 4.76, 0.18, 0.98]
#]
#STARTING_POINTS = [
#    [-4.59, 1.45, 0.84, 0.55],
#    [-5.70, 4.34, 0.21, 0.98],
#    [-7.21, 6.54, -0.57, 0.82],
#    [-4.37, 5.01, 0.20, 0.98],
#    [-7.96, 5.91, 0.24, 0.97]
#]
STARTING_POINTS = [
    [-5.25, 2.71, 0.82, 0.57],
    [-2.94, 2.39, 0.17, 0.99],
    [-5.93, 4.29, 0.17, 0.99],
    [-3.28, 5.40, -0.98, 0.20],
    [-3.58, -0.28, 0.85, 0.52]
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


def iterable_index_circle(length):
    while True: 
        for i in range(length):
            yield i

def save_thread():
    starting_point_ind = iterable_index_circle(len(STARTING_POINTS))
    while len(traj_buffer.task_buffer)<BUFFER_SIZE*2:
        time.sleep(1)
        if traj_buffer.waiting == 'task':
            publish_pose(driv_pub, STARTING_POINTS[next(starting_point_ind)])
            time.sleep(1)
            print('going to the starting point')
            while traj_buffer.nav_status != 3:
                time.sleep(1)
            task_msg = KeyValue()
            task_msg.key = 'new_task'
            task_msg.value = PROMPT
            task_pub.publish(task_msg)
            while traj_buffer.waiting != 'action':
                time.sleep(1)
            publish_pose(driv_pub, TARGET)
            time.sleep(1)
            print('moving to the target')
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
    save_filename = os.path.join(SAVE_DIR, f'tsa-trajs_{current_datetime}.h5')
    txt_filename = os.path.join(SAVE_DIR, f'tsa-trajs_{current_datetime}_tasks.txt')
    mapinfo_filename = os.path.join(SAVE_DIR, f'tsa-trajs_{current_datetime}_mapinfo.json')


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
        buffer_size= BUFFER_SIZE,
        always= False
    )

    #rospy.init_node('behavior_goto_node', anonymous=True)
    driv_pub = rospy.Publisher(ACTION_ROSTOPIC, PoseStamped, queue_size=1)
    task_pub = rospy.Publisher('/task', KeyValue, queue_size=1)

    publish_pose(driv_pub, torch.zeros((4)))

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=save_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()

    t2.join()
    rospy.signal_shutdown('record finished')