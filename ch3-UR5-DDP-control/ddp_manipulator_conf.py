# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os
from math import sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

q0 = np.array([ 0. , 0.,  0.,  0. ,  0. ,  0. ])  # initial configuration
qT = np.array([ 1. , 4.6,  0.0,  0. ,  0. ,  0. ])  # goal configuration
dt = 0.01*1                    # DDP time step
N = int(200/1)                     # horizon size

dt_sim = 1e-3
ndt = 1                          # number of integration steps for each control loop

simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'euler' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(6)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
simulate_real_time = 1          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]

#ERROR_MSG = 'You should set the environment variable UR5_MODEL_DIR to something like "$DEVEL_DIR/install/share"\n';
#path      = os.environ.get('UR5_MODEL_DIR', ERROR_MSG)
#urdf      = path + "/ur_description/urdf/ur5_robot.urdf";
#srdf      = path + '/ur_description/srdf/ur5_robot.srdf'
