import rospy
from std_srvs.srv import Empty
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import random
import math
from gazebo_msgs.msg import ModelState

'''
Running node to restart all objects in simulation to inital coordinates
(CUBE spawns at random place)
'''

def set_cube_coord():
    theta_cube = random.uniform(0, 2 * math.pi)
    theta_sphere = theta_cube + math.pi
    x_cube = 8 * math.cos(theta_cube)
    y_cube = 8 * math.sin(theta_cube)
    x_sphere = 8 * math.cos(theta_sphere)
    y_sphere = 8 * math.sin(theta_sphere)
    cube_pose = ModelState()
    cube_pose.model_name = "box_model"
    sphere_pose = ModelState()
    sphere_pose.model_name = "sphere_model"
    cube_pose.pose.position.x = x_cube
    cube_pose.pose.position.y = y_cube
    cube_pose.pose.position.z = 0.5 
    sphere_pose.pose.position.x = x_sphere
    sphere_pose.pose.position.y = y_sphere
    sphere_pose.pose.position.z = 0.5
    coord_pub.publish(sphere_pose)
    coord_pub.publish(cube_pose)

if __name__ == "__main__":
    rospy.init_node('world_resetter')

    # Wait for the Gazebo reset_world service to become available
    rospy.wait_for_service('/gazebo/reset_world')

    # Create a service proxy for the reset_world service
    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    twist_pub = rospy.Publisher('/robot_base_velocity_controller/cmd_vel', Twist, queue_size=1)
    coord_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=2)
    zero_twist = Twist()
    reset_pub = rospy.Publisher('/reset_signal', String, queue_size=1)

    # Call the reset_world service to reset the Gazebo world
    while not rospy.is_shutdown():
        try:
            reset_world()
            set_cube_coord()
            reset_pub.publish('reset')
            #twist_pub.publish(zero_twist)
            print("Gazebo world reset successful!")
            time.sleep(10)
        except:
            pass

