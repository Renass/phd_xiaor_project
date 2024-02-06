import rospy
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np

def scan_callback(scan_data):
    # Calculate the number of scans
    num_scans = round((scan_data.angle_max - scan_data.angle_min) / scan_data.angle_increment) + 1
    
    # Generate the angles array using np.linspace
    angles = np.linspace(scan_data.angle_min, scan_data.angle_max, num_scans)
    
    ranges = np.array(scan_data.ranges)
    
    # Convert polar coordinates (angle, range) to Cartesian (x, y) for plotting
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    
    # Update the plot with the new scan data
    plt.cla()  # Clear the current plot
    plt.plot(x, y, 'bo', markersize=2)  # Plot new data as blue dots
    plt.axis('equal')  # Keep the aspect ratio of the plot square
    plt.pause(0.001)  # Pause briefly to allow the plot to update


# Initialize the ROS node
rospy.init_node('scan_visualizer', anonymous=True)

# Create a subscriber to the /scan topic with scan_callback as the callback function
scan_subscriber = rospy.Subscriber('/scan', LaserScan, scan_callback)

# Prevent the script from exiting until the node is shut down
rospy.spin()
