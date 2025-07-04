#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 07:09:47 2020

@author: student
"""
import sys
import os
from os.path import dirname, exists, join

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from example_robot_data.robots_loader import getModelPath, readParamsFromSrdf


def loadUR(robotNum=5, limited=False, gripper=False, URDF_FILENAME='', path=''):
    assert (not (gripper and (robot == 10 or limited)))
    try:
        # first try to load model located in folder specified by env variable UR5_MODEL_DIR
        ERROR_MSG = 'You should set the environment variable UR5_MODEL_DIR to something like "$DEVEL_DIR/install/share"\n';
        path      = r"/home/he/Crocoddyl_bilibili/ch3-ddp/ch3-UR5-DDP-control"
        urdf      = path + "/ur_description/urdf/ur5_robot.urdf"
        robot = RobotWrapper.BuildFromURDF(urdf, [path, ])
        print("loaded robot:",robot)

        # collision setting
        try:
            srdf      = path + '/ur_description/srdf/ur5.srdf'
            pin.loadReferenceConfigurations(robot.model, srdf, False)
        except:
            srdf      = path + '/ur_description/srdf/ur5_robot.srdf'
            pin.loadReferenceConfigurations(robot.model, srdf, False)
        return robot
    
    except Exception as e:
        # if that did not work => use the one of example_robot_data
        from example_robot_data.robots_loader import loadUR
        return loadUR(robotNum, limited, gripper)
        
def loadUR_urdf(robot=5, limited=False, gripper=False):
    assert (not (gripper and (robot == 10 or limited)))
    URDF_FILENAME = "ur%i%s_%s.urdf" % (robot, "_joint_limited" if limited else '', 'gripper' if gripper else 'robot')
    URDF_SUBPATH = "/ur_description/urdf/" + URDF_FILENAME
    modelPath = getModelPath(URDF_SUBPATH)
    try:        
        path = '/opt/openrobots/share/'
        model = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [path])
        if robot == 5 or robot == 3 and gripper:
            SRDF_FILENAME = "ur%i%s.srdf" % (robot, '_gripper' if gripper else '')
            SRDF_SUBPATH = "/ur_description/srdf/" + SRDF_FILENAME
            readParamsFromSrdf(model, modelPath + SRDF_SUBPATH, False, False, None)
        return modelPath + URDF_SUBPATH, path
    except:
        return modelPath + URDF_SUBPATH, modelPath
        
# def loadPendulum():
#     URDF_FILENAME = "pendulum.urdf"
#     URDF_SUBPATH = "/pendulum_description/urdf/" + URDF_FILENAME
#     modelPath = getModelPath(URDF_SUBPATH)
#     robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath])
#     return robot
    
# def loadRomeo_urdf():
#     URDF_FILENAME = "romeo_small.urdf"
#     URDF_SUBPATH = "/romeo_description/urdf/" + URDF_FILENAME
#     modelPath = getModelPath(URDF_SUBPATH)
#     return modelPath + URDF_SUBPATH, modelPath