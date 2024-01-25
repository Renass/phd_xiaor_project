import random
import math

'''
Override sdf scripts with new coordinates of cube
(Used to be imported)
'''

def rewrite(distance):
    # Generate a random angle in radians
    theta = random.uniform(0, 2 * math.pi)

    # Calculate x and y coordinates using polar coordinates
    x = distance * math.cos(theta)
    y = distance * math.sin(theta)

    # Update the SDF file content with the new x and y coordinates
    sdf_content = f"""
    <sdf version="1.4">
    <model name="my_model">
    <pose>{x} {y} 0.5 0 0 0</pose>
    <static>true</static>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia> <!-- inertias are tricky to compute -->
          <!-- http://gazebosim.org/tutorials?tut=inertia&cat=build_robot -->
          <ixx>0.083</ixx>       <!-- for a box: ixx = 0.083 * mass * (y*y + z*z) -->
          <ixy>0.0</ixy>         <!-- for a box: ixy = 0 -->
          <ixz>0.0</ixz>         <!-- for a box: ixz = 0 -->
          <iyy>0.083</iyy>       <!-- for a box: iyy = 0.083 * mass * (x*x + z*z) -->
          <iyz>0.0</iyz>         <!-- for a box: iyz = 0 -->
          <izz>0.083</izz>       <!-- for a box: izz = 0.083 * mass * (x*x + y*y) -->
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
      </visual>
    </link>
    </model>
    </sdf>
    """

    # Update the SDF file with the new content
    with open("/home/renas/catkin_ws/src/mobile_manipulator_body/sdf/box.sdf", "w") as sdf_file:
        sdf_file.write(sdf_content)

