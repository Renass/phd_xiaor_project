import torch
from transformers import ViltProcessor, ViltModel, ViltImageProcessor
import h5py
import time
import os
import json
from tf.transformations import euler_from_quaternion
import numpy as np
import cv2

'''
rework(im_map)as_im
REWORK h5 dataset to:
1. image processor to image, resize 
2. Put an arrow to lidar map, resize 
3. Concatenate image and map to single image 

save as a new hdf
'''


DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/test/tsa-trajs_2024-04-30_17-15-25.h5'

def world_to_map(pose, resolution, origin):
    """
    Convert world coordinates to map pixel coordinates.
    
    :param pose: The pose in world coordinates (x, y).
    :param resolution: The map resolution (meters per pixel).
    :param origin: The origin of the map in world coordinates (x, y).
    :return: The pose in map pixel coordinates.
    """
    map_x =  int((pose[0] - origin[0]) / resolution)
    #map_y = mapinfo['height'] - int((pose[1] - origin[1]) / resolution)
    map_y = int((pose[1] - origin[1]) / resolution)
    return (map_x, map_y)

def draw_an_arrow_on_the_map(map, mapinfo, pose):
    '''
    unsqueeze a lidar map to 3 dimensions
    with 1st with map and second with pose arrow
    accept: numpy(batch_size, h, w)
    return: numpy(batch_size, 3, h, w)
    '''
    batch_size,h,w = map.shape
    empty_channel = np.zeros((batch_size, h, w))
    #map = np.expand_dims(map, axis=1)
    map = np.stack((map, empty_channel, empty_channel), axis=1)
    
    
    for i in range(batch_size):
        map_pose = world_to_map(
            (pose[i][0], pose[i][1]), 
            mapinfo['resolution'], 
            (mapinfo['origin']['position']['x'], 
            mapinfo['origin']['position']['y'])
        )
        quaternion = [0, 0, pose[i][2], pose[i][3]]
        _, _, yaw = euler_from_quaternion(quaternion)
        arrow_length = 50
        end_x = int(map_pose[0] + arrow_length * np.cos(yaw))
        end_y = int(map_pose[1] + arrow_length * np.sin(yaw))
        cv2.arrowedLine(map[i, 1, :, :], (map_pose[0], map_pose[1]), (end_x, end_y), 1.0, thickness=5)    
        
        # Visualization using matplotlib
        #plt.imshow(np.flipud(map[i].transpose(1,2,0)))
        #plt.show()
        return map

if __name__ == '__main__':
    new_dataset_path = DATASET[:-3]+'_reworked.h5'
    
    im_processor = ViltImageProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    im_processor.do_resize = True
    im_processor.do_rescale = False
    im_processor.do_normalize = False

    im = []
    action = []

    mapinfo_filename = f"{os.path.splitext(DATASET)[0]}_mapinfo.json"
    with open(mapinfo_filename, 'r') as file:
        mapinfo = json.load(file)

    with h5py.File(DATASET, 'r') as hdf:
        im_group = hdf['states']
        map_group =hdf['maps']
        pose_group = hdf['pose']
        action_group = hdf['actions']
        num_episodes = len(im_group)
        print('Dataset contains episodes: ', num_episodes)

        preprocess_timer_start = time.time()
        with h5py.File(new_dataset_path, 'w') as new_hdf:
            new_hdf_im_group = new_hdf.create_group('states')
            new_hdf_action_group = new_hdf.create_group('actions')
            for i in range(num_episodes):
                episode = 'data_'+str(i)
                pose_i = pose_group[episode][:]
                map_i = map_group[episode][:]/100
                map_i = draw_an_arrow_on_the_map(map_i, mapinfo, pose_i)
                map_i = torch.from_numpy(map_i).float()
                map_i = im_processor(images=map_i, return_tensors="pt")['pixel_values']
                im_i = torch.from_numpy(im_group[episode][:]).float()/255.0
                im_i = im_processor(images=im_i, return_tensors="pt")['pixel_values']
                im_i = torch.cat((im_i, map_i), dim=3).numpy()
                new_hdf_im_group.create_dataset(episode, data=im_i, dtype = np.float32, compression = 'gzip')
                a = action_group[episode][:]
                new_hdf_action_group.create_dataset(episode, data=a, dtype = np.float32, compression = 'gzip')

    
    print('preprocess full time: ',time.time()-preprocess_timer_start)
    