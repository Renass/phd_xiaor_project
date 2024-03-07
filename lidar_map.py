import rospy
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
import numpy as np

def map_callback(map_data):
    # Extract the width, height, resolution, and origin from the map data
    width = map_data.info.width
    height = map_data.info.height
    resolution = map_data.info.resolution
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y

    # Convert the flat map data into a 2D numpy array for visualization
    data = np.array(map_data.data).reshape((height, width))
    #data = np.where(data == -1, -1, 100 - data)
    
    # Clear the current axes, plot the data, and refresh the plot
    plt.cla()  # Clear the current axes
    plt.imshow(data, cmap='gray', origin='lower', 
               extent=(origin_x, origin_x + width * resolution, 
                       origin_y, origin_y + height * resolution))
    plt.draw()
    plt.pause(0.001)  # Pause to update the plot

# Initialize the ROS node
rospy.init_node('map_visualizer', anonymous=True)

# Subscribe to the /map topic to receive the global map data
map_subscriber = rospy.Subscriber('/map', OccupancyGrid, map_callback)

# Keep the program alive until it's stopped manually
rospy.spin()
