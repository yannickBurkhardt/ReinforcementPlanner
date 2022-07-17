import os

from yacs.config import CfgNode as Node

# -----------------------------------------------------------------------------
# Environment definition
# -----------------------------------------------------------------------------
_C = Node()
_C.ENV = Node()
_C.ENV.TOTAL_OBSTACLES = 10
_C.ENV.SEEN_OBSTACLES = 10
_C.ENV.MAX_VEL = 2.5
_C.ENV.DYNAMIC_OBSTACLES = True
_C.ENV.OBSTACLES_SPEED_FACTOR = 0.5
# Only if ENV_DIFFERENT_OBS_SIZES is True:
_C.ENV.OBS_MIN_SIZE = 5
_C.ENV.OBS_MAX_SIZE = 25
#Rewards
_C.ENV.GOAL_REWARD = 1.0
_C.ENV.CRASH_REWARD = 0.0

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
_C.MODEL = Node()
_C.MODEL.HIDDEN_SIZE = [256,256]
_C.MODEL.OUNOISE_SIGMA = 0.1
## -----------------------------------------------------------------------------
# Train definition
# -----------------------------------------------------------------------------
_C.TRAIN = Node()
_C.TRAIN.OUT_DIR = 'experiments/'
_C.TRAIN.TOTAL_STEPS = 4000000