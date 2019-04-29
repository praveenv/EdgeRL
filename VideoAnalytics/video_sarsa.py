import numpy as np
import itertools

from collections import defaultdict,OrderedDict
from video_env import Environment
from video_statedef import State
from tqdm import tqdm
from video_dqn import DQNAgent