import rospy
from gazebo_msgs.msg import ModelStates
import os
import tf
import math
import threading
from geometry_msgs.msg import Twist

'''
Driving to achieve target object with simple 'if' script using odometry
'''

def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass

def model_states_callback(msg):
    #os.system('clear')
    robot_base_index = msg.name.index("robot_base")
    robot_base_pose = msg.pose[robot_base_index]
    target_pose = msg.pose[msg.name.index(TARGET)]

    global robot_xyz 
    robot_xyz = robot_base_pose.position
    global robot_yaw
    robot_yaw = quaternion_to_yaw(robot_base_pose.orientation)
    global target_xyz
    target_xyz=target_pose.position
    global target_yaw
    target_yaw = math.atan2(target_xyz.y - robot_xyz.y, target_xyz.x - robot_xyz.x)

def quaternion_to_yaw(orientation):
    # Convert the quaternion to a tuple (x, y, z, w)
    quat = (orientation.x, orientation.y, orientation.z, orientation.w)

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    euler = tf.transformations.euler_from_quaternion(quat)

    # Extract the yaw angle (rotation around the Z-axis)
    yaw = euler[2]  # The third element is the yaw angle

    return yaw


def driving_thread():
    while not rospy.is_shutdown():
        if not (robot_yaw == None) and not (target_yaw==None):
            
            while abs(robot_yaw-target_yaw)>0.5:
                #print('strong')
                publish_twist(driv_pub, [0, 2])
            while abs(robot_yaw-target_yaw)>0.4:
                #os.system('clear')
                #print('robot_yaw: ',robot_yaw)
                #print('target_yaw: ',target_yaw)
                #print('low')
                publish_twist(driv_pub, [0, 1])
            
            if ((robot_xyz.x-target_xyz.x)**2+(robot_xyz.y-target_xyz.y)**2)> 20:
                publish_twist(driv_pub, [4, 0])
                #print('fast')
            elif ((robot_xyz.x-target_xyz.x)**2+(robot_xyz.y-target_xyz.y)**2)> 15:
                publish_twist(driv_pub, [2, 0])
                #print('slow')


def publish_twist(publisher, a):
    twist_msg = Twist()
    twist_msg.linear.x = a[0]
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = a[1]
    publisher.publish(twist_msg)

#TARGET = 'box_model'
TARGET = 'sphere_model'

if __name__ == '__main__':
    rospy.init_node('scripted_behav', anonymous=True)
    sub = rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback,queue_size=1) 
    driv_pub = rospy.Publisher('robot_base_velocity_controller/cmd_vel', Twist, queue_size=1)

    robot_xyz = None
    robot_yaw = None
    target_xyz = None
    target_yaw = None

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=driving_thread)
    t1.start()
    print('Odom thread starts')
    t2.start()
    print('Driving thread starts')
    t1.join()
    t2.join()

