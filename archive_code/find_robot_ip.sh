#!/bin/bash

# Replace with your robot's MAC address
ROBOT_MAC="ac:82:47:32:cb:e3"

# Scan the network and find the IP corresponding to the MAC address
ROBOT_IP=$(sudo arp-scan --localnet | grep $ROBOT_MAC | awk '{print $1}')

if [ -n "$ROBOT_IP" ]; then
    echo "Robot IP: $ROBOT_IP"
else
    echo "Robot not found on the network."
fi


#Compile this code:
# chmod +x find_robot_ip.sh
# Execute this code:
# sudo ./find_robot_ip.sh
# sudo ~/pythonprogv2/phd_xiaor_project/find_robot_ip.sh
