import trajectories_gather6
import rospy
import os
from datetime import datetime
import h5py 
import numpy as np 
import threading
import time

'''
Write a Task-State-Action dataset as a HDF5 file with gzip 

imports: trajectories_gather6
'''

BUFFER_SIZE = 2
SAVE_DIR = 'TSA_dataset/nav'
IMAGE_TOPIC = '/image_raw'


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
    save_filename = os.path.join(SAVE_DIR, f'tsa-trajs{current_datetime}.h5')



    with h5py.File(save_filename, 'w') as hf:
        # Create HDF5 datasets for states and actions
        states_group = hf.create_group('states')
        actions_group = hf.create_group('actions')

        for i, states_trajectory in enumerate(traj_buffer.states_buffer):
            state = np.stack(states_trajectory, axis=0)
            states_group.create_dataset('data_'+str(i), data=state, dtype = np.float32, compression = 'gzip')

        for i, actions_trajectory in enumerate(traj_buffer.actions_buffer):
            action = np.stack(actions_trajectory, axis=0)
            actions_group.create_dataset('data_'+str(i), data=action, dtype = np.float32, compression = 'gzip')


    print('Buffer saved')

if __name__ == '__main__':
    traj_buffer = trajectories_gather6.TrajectoryBuffer(
        image_topic= IMAGE_TOPIC,
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