import rospy
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
import numpy as np

def costmap_callback(costmap_data):
    width = costmap_data.info.width
    height = costmap_data.info.height
    resolution = costmap_data.info.resolution
    origin_x = costmap_data.info.origin.position.x
    origin_y = costmap_data.info.origin.position.y

    data = np.array(costmap_data.data).reshape((height, width))
    
    plt.cla()
    plt.imshow(data, cmap='gray', origin='lower', extent=(origin_x, origin_x + width * resolution, origin_y, origin_y + height * resolution))
    plt.draw()
    plt.pause(0.001)

rospy.init_node('costmap_visualizer', anonymous=True)
costmap_subscriber = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, costmap_callback)
rospy.spin()
