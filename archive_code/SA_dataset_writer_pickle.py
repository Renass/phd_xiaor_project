import trajectories_gather4
from torchvision import transforms
import threading
import rospy
import time
import pickle
import os
from datetime import datetime

'''
Behavioral cloning transformer State-Action DATASET WRITER to a file
imports: trajectories_gather4
(used as a script)
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
        save_filename = os.path.join(save_dir, f'sa-trajs{current_datetime}.pkl')
        with open(save_filename, 'wb') as f:
             pickle.dump(traj_buffer.traj, f)
        print('Buffer ready')
        


if __name__ == '__main__':
    BUFFER_SIZE = 1
    IM_AMOUNT = 1
    IM_RESOLUTION = (224,224)
    IM_CHANNELS = 3
    NUM_TRANSITIONS = 100

    traj_buffer = trajectories_gather4.TrajectoryBuffer(
        buffer_size=BUFFER_SIZE, im_amount=IM_AMOUNT, im_resolution=IM_RESOLUTION, 
        im_channels=IM_CHANNELS, num_transitions=NUM_TRANSITIONS, always=False)
    


    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('Behavioral cloning record starts')

    t2.join()
    rospy.signal_shutdown('Behavioral cloning finished')