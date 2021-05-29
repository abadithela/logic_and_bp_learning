# This script checks for the cost incurred from each of the value functions and switches to the lower cost 
import numpy as np
import h5py
from logic_config_lake import *
from pyvirtualdisplay import Display
import numpy as np
from six.moves import range
np.random.seed(3141592)
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from optimization_problem import Program
from fittedq import *
from exponentiated_gradient import ExponentiatedGradient
from fitted_off_policy_evaluation import *
from exact_policy_evaluation import ExactPolicyEvaluator
from stochastic_policy import StochasticPolicy
from DQN import DeepQLearning
from print_policy import PrintPolicy
from keras.models import load_model
from tensorflow.keras import backend as K
from env_dqns import *
import deepdish as dd
import time
import os
np.set_printoptions(suppress=True)
from config_lake import *
import pdb

model_dir = os.path.join(os.getcwd(), 'models')
if not os.path.exists(model_dir):
	os.makedirs(model_dir)
# opt = 1 # Different options for different versions of the high-level algorithm
# K1 and K2 are controllers for the OR constraints.
# Policy is a CNN
def choose_ctrl(K1, K2, st):
	a1 = K1(st) # Action chosen by controller 1
	a2 = K2(st) # Action chosen by controller 2
	_, costs1, done1, _ = env.step(action_space_map[a1])
	_, costs2, done2, _ = env.step(action_space_map[a1])
	if not done1 and not done2:
		print("Error: Neither action is feasible")
	elif done1 and not done2:
		return a1
	elif done2 and not done1:
		return a2
	else:
		if costs1[0] > costs2[0]:   # Comparing only the cost functions
			return a2
		else:
			return a1

