import rospy
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped 
from collections import deque

'''
Node to gather sequences of:
camera_image, map_image 
'''

class TrajectoryBuffer:
    def __init__(self, buffer_size):
        self.goal_buffer = deque([], maxlen=buffer_size)
        rospy.init_node('traj_gather_node', anonymous=True)
        self.status_subscriber = rospy.Subscriber('/move_base/status', GoalStatusArray, self.goal_status_callback)
        self.goal_subscriber = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

    
    def  goal_status_callback(self, status_msg):
        #if status_msg.status_list[-1].status == 1:
            #print('active goal')
        #elif status_msg.status_list[-1].status == 3:
        #    print('goal reached')
        #elif status_msg.status_list[-1].status == 4:
        #    print('failed')
        pass
    
    def goal_callback(self, goal_msg):
        # Extract the target coordinates and orientation
    
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
        # Append the simplified goal info to the buffer
        self.goal_buffer.append(goal_info)
        print("New goal received and added to buffer.")
        # Optionally, print the current buffer size or contents
        print("Current buffer size:", len(self.goal_buffer))
        # For demonstration, you could print the last goal added
        print("Last goal added to buffer:", goal_info)


if __name__ == '__main__':
    traj_buffer = TrajectoryBuffer(buffer_size=10)
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass