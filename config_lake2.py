
#### Setup Gym 
from __future__ import absolute_import
from frozen_lake2 import ExtendedFrozenLake
import numpy as np

map_size = 8
# register( id='FrozenLake-no-slip-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False, 'map_name':'{0}x{0}'.format(map_size)} )
# env = gym.make('FrozenLake-no-slip-v0')
max_time_spent_in_episode = 100
# env = ExtendedFrozenLake(max_time_spent_in_episode, map_name = '{0}x{0}'.format(map_size), is_slippery= False)
env = ExtendedFrozenLake(max_time_spent_in_episode, map_name = '{0}m2x{0}m2'.format(map_size), is_slippery= False)
position_of_holes = np.arange(env.desc.shape[0]*env.desc.shape[1]).reshape(env.desc.shape)[np.nonzero(env.desc == 'H')]
position_of_goals = np.arange(env.desc.shape[0]*env.desc.shape[1]).reshape(env.desc.shape)[np.nonzero(env.desc == 'G')]



#### Hyperparam
gamma = 0.9
max_epochs = 5000 # max number of epochs over which to collect data
max_Q_fitting_epochs = 30 #max number of epochs over which to converge to Q^\ast.   Fitted Q Iter
max_eval_fitting_epochs = 30 #max number of epochs over which to converge to Q^\pi. Off Policy Eval
lambda_bound = 30. # l1 bound on lagrange multipliers
epsilon = .01 #0.01 termination condition for two-player game
deviation_from_old_policy_eps = .95 #With what probabaility to deviate from the old policy
# convergence_epsilon = 1e-6 # termination condition for model convergence
action_space_dim = env.nA # action space dimension
state_space_dim = env.nS # state space dimension
eta = 50. # param for exponentiated gradient algorithm
initial_states = [[0]] #The only initial state is [1,0...,0]. In general, this should be a list of initial states
non_terminal_states = np.nonzero(np.reshape(((env.desc == 'S') + (env.desc == 'F')), -1))[0] # Used for dynamic programming. this is an optimization to make the algorithm run faster. In general, you may not have this
max_number_of_main_algo_iterations = 100 # After how many iterations to cut off the main algorithm
model_type = 'mlp'
old_policy_name = 'pi_old_map_size_{0}_{1}.h5'.format(map_size, model_type)
final_policy_name = 'final_policy_left_{0}_{1}.h5'.format(map_size, model_type)
# final_policy_name = 'test_{0}_{1}.h5'.format(map_size, model_type)
# constraints = [.1, 0]
constraints = [0, .1] + [0.4]
constraints_cared_about = [0, 1]
starting_lambda = 'uniform'

## DQN Param
num_iterations = 5000
sample_every_N_transitions = 10
batchsize = 1000
copy_over_target_every_M_training_iterations = 100
buffer_size = 10000
num_frame_stack=1
min_buffer_size_to_train=0
frame_skip = 1
pic_size = tuple()
min_epsilon = .02
initial_epsilon = .3
epsilon_decay_steps = 1000 #num_iterations
min_buffer_size_to_train = 2000 # min_buffer

# Other
stochastic_env = False
action_space_map = {
                0: 0,  
                1: 1,  
                2: 2,
                3: 3}

prob = [1/float(action_space_dim)]*action_space_dim # Probability with which to explore space when deviating from old policy

calculate_gap = True # Run Main algo. If False, it skips calc of primal-dual gap
infinite_loop = False # Stop script if reached primal-dual gap threshold
policy_improvement_name = 'lake_policy_improvement.h5'
results_name = 'lake_results.csv'
