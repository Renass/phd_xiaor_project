import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped 
from collections import deque
from diagnostic_msgs.msg import KeyValue
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from torchvision import transforms
import numpy as np

'''
Node to gather sequences of:
task state action



waiting queue: 'task': ['state', 'action', 'status']

new task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'new_task', value: 'go left'}"
end task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'end_task', value: 'done'}"
'''



class TrajectoryBuffer:
    def __init__(self, image_topic, buffer_size, always=True):
        self.waiting = 'task'
        self.always = always
        self.buffer_size = buffer_size
        self.task_buffer = deque([], maxlen=buffer_size*2)
        self.states_buffer = deque([[]], maxlen=buffer_size)
        self.actions_buffer = deque([[]], maxlen=buffer_size)
        rospy.init_node('traj_gather_node', anonymous=True)
        rospy.Subscriber(image_topic, Image, self.callback_image)
        self.status_subscriber = rospy.Subscriber('/move_base/status', GoalStatusArray, self.goal_status_callback)
        self.goal_subscriber = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.task_subscriber = rospy.Subscriber('/task', KeyValue, self.task_callback)

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
            self.waiting = 'action'
            image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = transforms.ToTensor()(image)
            image = np.array(image)
            self.states_buffer[-1].append(image)
            print('state_add')

if __name__ == '__main__':
    traj_buffer = TrajectoryBuffer(image_topic= '/image_raw', buffer_size=10)
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass