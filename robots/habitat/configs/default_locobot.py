# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from yacs.config import CfgNode as CN

_C = CN()

# whether the robot has an arm or not
_C.HAS_ARM = False
# whether the robot has a mobile base or not
_C.HAS_BASE = True
# whether the robot has a camera or not
_C.HAS_CAMERA = True
# whether the robot has a gripper or not
_C.HAS_GRIPPER = False
# whether the robot has a common shared class among all components
_C.HAS_COMMON = True


# Camera specific setting
_C.CAMERA = CN()
# CAMERA class name
_C.CAMERA.CLASS = "LoCoBotCamera"
# reset value for the pan
_C.CAMERA.RESET_PAN = 0.0
# reset value for the tilt
_C.CAMERA.RESET_TILT = 0.0

# Base specific settings
_C.BASE = CN()
# BASE class name
_C.BASE.CLASS = "LoCoBotBase"

_C.COMMON = CN()
# Name of the common class variable that will be shared in Robot class
_C.COMMON.NAME = "simulator"
# Class type to assign to 'simulator' variable
_C.COMMON.CLASS = "HabitatSim"

_C.COMMON.SIMULATOR = CN()

# Contains all of the simulator config
_SIMULATORC = _C.COMMON.SIMULATOR

_SIMULATORC.PHYSICS = False

_SIMULATORC.DEFAULT_AGENT_ID = 0

_SIMULATORC.SCENE_ID = "none"
_SIMULATORC.PHYSICS_CONFIG_FILE = "none"

# Contains the config of all the agents in simulation
_SIMULATORC.AGENT = CN()

_SIMULATORC.AGENT.NAME = ["locobot"]

# TODO: ADD agent- types, more agents, agent height, radius, actions etc

# Sensor config on the agent
_SIMULATORC.AGENT.SENSORS = CN()

# Set sensor names
_SIMULATORC.AGENT.SENSORS.NAMES = ["rgb", "depth"]

# Set sensor type (COLOR, DEPTH, SEMANTIC)
_SIMULATORC.AGENT.SENSORS.TYPES = ["COLOR", "DEPTH"]

# x, y, z and roll pitch yaw w.r.t to agent in meters and radians
# TODO: Check if these units are consistant with Habitat-sim
_SIMULATORC.AGENT.SENSORS.POSES = [
    [0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
]
# Height and Widhth in Pixels
_SIMULATORC.AGENT.SENSORS.RESOLUTIONS = [[512, 512], [512, 512]]


def get_cfg():
    return _C.clone()
