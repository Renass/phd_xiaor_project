'''
Go through predifned poses on lidar map and create the
text-image token from text-image model backbone for future classification

import trajectories_gather6.py
'''

#Sim 2A724_x3.yaml
POSES = [ 
    [14.9, 5.7, 0.14, 0.99],
    [15.30, 21.31, -0.53, 0.85]
]

DESCRIPTIONS = [
    'position facing fridge and rack',
    'position out of the 2A724 lab facing right side of the corridor'

]