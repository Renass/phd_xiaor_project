import trajectories_gather5
import threading
import rospy
import time
import os
from datetime import datetime
import h5py  # Import h5py for HDF5 support
import torch
import numpy as np

'''
Write a SA dataset as a HDF5 file with gzip 
imports: trajectories_gather5
'''

def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass

def behav_clon_thread():
    while traj_buffer.gather == True:
        time.sleep(1)

    save_dir = 'sa-traj_dataset'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_filename = os.path.join(save_dir, f'sa-trajs{current_datetime}.h5')  # Use .h5 extension for HDF5



    for i, states_trajectory in enumerate(traj_buffer.states_buffer):
        traj_buffer.states_buffer[i] = torch.stack(states_trajectory, dim=0)
    traj_buffer.states_buffer = torch.stack(list(traj_buffer.states_buffer), dim=0)

    traj_buffer.actions_buffer = torch.tensor(traj_buffer.actions_buffer)

    # Create an HDF5 file to store the data
    with h5py.File(save_filename, 'w') as hf:
        # Create HDF5 datasets for states and actions
        states_group = hf.create_group('states')
        actions_group = hf.create_group('actions')

        # Convert the states and actions data to NumPy arrays before saving
        states_data = traj_buffer.states_buffer.cpu().numpy()
        actions_data = traj_buffer.actions_buffer.cpu().numpy()

        # Create datasets for states and actions within their respective groups
        states_group.create_dataset('data', data=states_data, dtype = np.float32, compression = 'gzip')
        actions_group.create_dataset('data', data=actions_data, dtype = np.float32, compression = 'gzip')

    print('Buffer saved')

if __name__ == '__main__':
    BUFFER_SIZE = 2
    IM_RESOLUTION = (640, 480)
    NUM_TRANSITIONS = 30

    traj_buffer = trajectories_gather5.TrajectoryBuffer(
        buffer_size=BUFFER_SIZE, 
        im_resolution=IM_RESOLUTION, 
        num_transitions=NUM_TRANSITIONS, 
        always=False,
        image_topic= '/camera/rgb/image_raw',
        cmd_vel_topic= '/cmd_vel',
        reset_environment= False
    )

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('State-action record starts')

    t2.join()
    rospy.signal_shutdown('State-action record finished')

