import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped 
from collections import deque
from diagnostic_msgs.msg import KeyValue
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from nav_msgs.srv import GetMap
import tf
from transformers import ViltImageProcessor
import torch
from tf.transformations import euler_from_quaternion
import time

'''
Node to gather sequences of:
{task, 
state:(camera-map concatenated image pose is drawed as an arrow on the map)
action}



waiting queue: 'task': ['state', 'action', 'status']
status is a variable that shows if robot:
* currently moving by navigation system
* success or fail of navigation

new task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'new_task', value: 'go left'}"
end task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'end_task', value: 'done'}"
'''



IMAGE_TOPIC = '/image_raw'

#For SLAM:
#MAP_SERVICE = '/dynamic_map'
#For AMCL:
MAP_SERVICE = '/static_map'

IM_PROCESSOR = ViltImageProcessor.from_pretrained("dandelin/vilt-b32-mlm")
IM_PROCESSOR.do_resize = True
IM_PROCESSOR.do_rescale = False
IM_PROCESSOR.do_normalize = False

class TrajectoryBuffer:
    def __init__(self, image_topic, map_service, buffer_size, always=True):
        self.waiting = 'task'
        self.always = always
        self.buffer_size = buffer_size

        self.task_buffer = deque([], maxlen=buffer_size*2)
        self.states_buffer = deque([[]], maxlen=buffer_size)
        self.actions_buffer = deque([[]], maxlen=buffer_size)
        self.map_info = None
        self.nav_status = None

        rospy.init_node('traj_gather_node', anonymous=True)
        rospy.Subscriber(image_topic, Image, self.callback_image)
        self.map_service = rospy.ServiceProxy(map_service, GetMap)
        self.pose_listener = tf.TransformListener()
        self.status_subscriber = rospy.Subscriber('/move_base/status', GoalStatusArray, self.goal_status_callback)
        self.goal_subscriber = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.task_subscriber = rospy.Subscriber('/task', KeyValue, self.task_callback)
        #self.global_costmap_subscriber = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.global_costmap_callback)

    def task_callback(self, task_msg):
        if self.waiting == 'task' and task_msg.key == 'new_task':
            task = {task_msg.key : task_msg.value}
            self.task_buffer.append(task)
            self.waiting = 'state'
            print('task start')
        
        if self.waiting == 'action' and task_msg.key == 'end_task':
            task = {task_msg.key : task_msg.value}
            self.task_buffer.append(task)
            print('task end')
            print('buffer length: ',len(self.states_buffer))
            if self.always or len(self.states_buffer)< self.buffer_size:
                self.states_buffer.append([])
                self.actions_buffer.append([])
                self.waiting = 'task'



    def  goal_status_callback(self, status_msg):
        if status_msg.status_list != []:
            self.nav_status = status_msg.status_list[-1].status
        #self.nav_status = status_msg.status_list[-1].status
        if self.waiting == 'status' and status_msg.status_list[-1].status in [3,4]:
            print('seems navigation ended')
            self.waiting = 'state'
        #if status_msg.status_list[-1].status == 1:
            #print('active goal')
        #elif status_msg.status_list[-1].status == 3:
        #    print('goal reached')
        #elif status_msg.status_list[-1].status == 4:
        #    print('failed')
    
    def goal_callback(self, goal_msg):
        if self.waiting == 'action':
            goal_info = {
                'position': {
                    'x': goal_msg.pose.position.x,
                    'y': goal_msg.pose.position.y
                },
                'orientation': {
                    'z': goal_msg.pose.orientation.z,
                    'w': goal_msg.pose.orientation.w
                }
            }
            action = np.array([
                goal_msg.pose.position.x,
                goal_msg.pose.position.y,
                goal_msg.pose.orientation.z,
                goal_msg.pose.orientation.w])
            self.actions_buffer[-1].append(action)
            print('action_add')
            self.waiting = 'status'



    def callback_image(self, msg):
        if self.waiting == 'state':
            state_start_time = time.time()
            #self.waiting = 'state.costmap'
            image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            image = image/255.0
            
            map = self.map_service().map
            self.map_info = {
                'resolution' : map.info.resolution,
                'width' : map.info.width,
                'height' : map.info.height,
                'origin' : {
                    'position' : {
                        'x' : map.info.origin.position.x,
                        'y' : map.info.origin.position.y,
                        'z' : map.info.origin.position.z },
                    'orientation' : {
                        'x' : map.info.origin.orientation.x,
                        'y' : map.info.origin.orientation.y,
                        'z' : map.info.origin.orientation.z,
                        'w' : map.info.origin.orientation.w}}}
            
            map = np.array(map.data, dtype=np.int8).reshape(map.info.height, map.info.width)
            map[map == -1] = 50
            map = map/100
            
            #string below may be not required
            self.pose_listener.waitForTransform('/map', '/base_link', rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.pose_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            pose = np.array([
                trans[0],
                trans[1],
                rot[2],
                rot[3]
                ])
            map = np.expand_dims(map, axis=0)
            pose = np.expand_dims(pose, axis=0)
            map = draw_an_arrow_on_the_map(map, self.map_info, pose)
            map = torch.from_numpy(map).float()
            map = IM_PROCESSOR(images=map, return_tensors="pt")['pixel_values']
            image = np.expand_dims(image, axis=0)
            image = IM_PROCESSOR(images=image, return_tensors="pt")['pixel_values']
            image = torch.cat((image, map), dim=3)[0].numpy()
            self.states_buffer[-1].append(image)
            print('state add')
            print('State gather time: ', time.time()-state_start_time)
            self.waiting = 'action'

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












if __name__ == '__main__':
    traj_buffer = TrajectoryBuffer(
        image_topic= IMAGE_TOPIC,
        map_service= MAP_SERVICE,
        always= True, 
        buffer_size=10)
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass