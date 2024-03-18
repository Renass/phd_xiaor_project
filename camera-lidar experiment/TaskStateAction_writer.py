import trajectories_gather6
import rospy
import os
from datetime import datetime
import h5py 
import numpy as np 
import threading
import time
import json

'''
Write a Task-State-Action dataset as a HDF5 file with gzip 

imports: trajectories_gather6
'''

BUFFER_SIZE = 1
SAVE_DIR = 'TSA_dataset/nav'
IMAGE_TOPIC = '/camera/rgb/image_raw'
#IMAGE_TOPIC = '/image_raw'
MAP_SERVICE = '/dynamic_map'
#MAP_SERVICE = '/static_map'


def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass

def save_thread():
    while len(traj_buffer.task_buffer)<BUFFER_SIZE*2:
        time.sleep(1)


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

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=save_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()

    t2.join()
    rospy.signal_shutdown('record finished')