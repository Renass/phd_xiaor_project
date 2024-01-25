import trajectories_gather5
#from torchvision import transforms
import threading
import rospy
import time
import os
from datetime import datetime
import torch

'''
Write a SA dataset as a Torch .pt file 
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
        save_filename = os.path.join(save_dir, f'sa-trajs{current_datetime}.pt')

        for i, states_trajectory in enumerate(traj_buffer.states_buffer):
             traj_buffer.states_buffer[i] = torch.stack(states_trajectory, dim=0)
        traj_buffer.states_buffer = torch.stack(list(traj_buffer.states_buffer), dim=0)

        traj_buffer.actions_buffer = torch.tensor(traj_buffer.actions_buffer)

        print(traj_buffer.states_buffer.shape)
        data = {
            'states': traj_buffer.states_buffer,
            'actions': traj_buffer.actions_buffer
            }
        torch.save(data, save_filename)
        print('Buffer saved')
        


if __name__ == '__main__':
    BUFFER_SIZE = 2
    IM_RESOLUTION = (224,224)
    NUM_TRANSITIONS = 100

    traj_buffer = trajectories_gather5.TrajectoryBuffer(
        buffer_size=BUFFER_SIZE, 
        im_resolution=IM_RESOLUTION, 
        num_transitions=NUM_TRANSITIONS, 
        always=False
        )
    


    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('State-action record starts')

    t2.join()
    rospy.signal_shutdown('State-action record finished')