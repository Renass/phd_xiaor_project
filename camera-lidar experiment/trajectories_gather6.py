import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped 
from collections import deque
from diagnostic_msgs.msg import KeyValue
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
import tf

'''
Node to gather sequences of:
{task, 
state:(camera_image, map-costmap, pose), 
action}



waiting queue: 'task': ['state', 'state.costmap', 'action', 'status']

new task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'new_task', value: 'go left'}"
end task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'end_task', value: 'done'}"
'''



IMAGE_TOPIC = '/image_raw'
#MAP_SERVICE = '/dynamic_map'
MAP_SERVICE = '/static_map'

class TrajectoryBuffer:
    def __init__(self, image_topic, map_service, buffer_size, always=True):
        self.waiting = 'task'
        self.always = always
        self.buffer_size = buffer_size

        self.task_buffer = deque([], maxlen=buffer_size*2)
        self.states_buffer = deque([[]], maxlen=buffer_size)
        self.map_buffer = deque([[]], maxlen=buffer_size)
        self.costmap_buffer = deque([[]], maxlen=buffer_size)
        self.actions_buffer = deque([[]], maxlen=buffer_size)
        self.pose_buffer = deque([[]], maxlen=buffer_size)
        self.map_info = None
        self.nav_status = None

        rospy.init_node('traj_gather_node', anonymous=True)
        rospy.Subscriber(image_topic, Image, self.callback_image)
        self.map_service = rospy.ServiceProxy(map_service, GetMap)
        self.pose_listener = tf.TransformListener()
        self.status_subscriber = rospy.Subscriber('/move_base/status', GoalStatusArray, self.goal_status_callback)
        self.goal_subscriber = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.task_subscriber = rospy.Subscriber('/task', KeyValue, self.task_callback)
        self.global_costmap_subscriber = rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.global_costmap_callback)

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
                self.map_buffer.append([])
                self.costmap_buffer.append([])
                self.pose_buffer.append([])
                self.waiting = 'task'



    def  goal_status_callback(self, status_msg):
        if status_msg != []:
            self.nav_status = status_msg.status_list[-1].status
        #self.nav_status = status_msg.status_list[-1].status
        if self.waiting == 'status' and status_msg.status_list[-1].status in [3,4]:
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
            #self.waiting = 'state.costmap'
            image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = transforms.ToTensor()(image)
            image = np.array(image)
            self.states_buffer[-1].append(image)
            
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
            
            self.waiting = 'state.costmap'
            map = np.array(map.data, dtype=np.int8).reshape(map.info.height, map.info.width)
            map[map == -1] = 50
            self.map_buffer[-1].append(map)
            
            #string below may be not required
            self.pose_listener.waitForTransform('/map', '/base_link', rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.pose_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            pose = np.array([
                trans[0],
                trans[1],
                rot[2],
                rot[3]
                ])
            self.pose_buffer[-1].append(pose)
            print('image add')

    def global_costmap_callback(self, costmap_msg):
        if self.waiting == 'state.costmap':
            costmap = np.array(costmap_msg.data, dtype=np.int8).reshape(self.map_info['height'], self.map_info['width'])
            costmap[costmap == -1] = 50
            self.costmap_buffer[-1].append(costmap)
            print('state_add')
            self.waiting = 'action'

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