# Apurva Badithela
# 5/27/21:  The python file for learning the high-level policy of the frozen lake example

# Example: We consider the FrozenLake example with the following or constraints:
# A frozen lake environment with arbitrary grid configurations with holes, two goals, and free space.
# The policy here is to constrain going to G1 with a restricted number of "west" steps OR going to
# G2 with a restricted number of "east" steps. 
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
opt = 1 # Different options for different versions of the high-level algorithm
# opt = 1 is for a direct high-level policy that doesn't have the safety constraint of falling in the hole

# ====================================== Basic functions ==================================== #
def learn_old_policy():
	HL_policy_old = None
	HL_old_policy_path = os.path.join(model_dir, HL_old_policy_name)
	print("HL_old_policy_path") # <--- set in config_lake.py
	HL_policy_old = LakeDQN(env, HL_gamma, 
		action_space_map = HL_action_space_map, 
		model_type=HL_model_type,
		position_of_holes=position_of_holes,
		position_of_goals=position_of_goals, 
		max_time_spent_in_episode=max_time_spent_in_episode,
		num_iterations = HL_num_iterations,
		sample_every_N_transitions = HL_sample_every_N_transitions,
		batchsize = HL_batchsize,
		min_epsilon = HL_min_epsilon,
		initial_epsilon = HL_initial_epsilon,
		epsilon_decay_steps = HL_epsilon_decay_steps,
		copy_over_target_every_M_training_iterations = HL_copy_over_target_every_M_training_iterations,
		buffer_size = HL_buffer_size,
		num_frame_stack=HL_num_frame_stack,
		min_buffer_size_to_train=HL_min_buffer_size_to_train,
		frame_skip = HL_frame_skip,	
		pic_size = HL_pic_size,
		models_path = os.path.join(model_dir,'weights.{epoch:02d}-{loss:.2f}.hdf5') ,
		)

	# Always learning a new policy at the high level:
	print('Learning a policy using DQN')
	HL_policy_old.learn()
	HL_policy_old.Q.model.save(HL_old_policy_path)
	policy_printer = PrintPolicy(size=[HL_map_size, HL_map_size], env=env)
	policy_printer.pprint(HL_policy_old)
	return HL_old_policy_path, HL_policy_old

# Setup of the initial problem:
def initial_setup():
	best_response_algorithm = LakeFittedQIteration(HL_state_space_dim + HL_action_space_dim, 
		[HL_map_size, HL_map_size], 
		HL_action_space_dim, 
		HL_max_Q_fitting_epochs, 
		HL_gamma, 
		model_type=HL_model_type, 
		position_of_goals=position_of_goals, 
		position_of_holes=position_of_holes,
		num_frame_stack=HL_num_frame_stack)

	fitted_off_policy_evaluation_algorithm = LakeFittedQEvaluation(HL_initial_states, 
		HL_state_space_dim + HL_action_space_dim, 
		[HL_map_size, HL_map_size], 
		HL_action_space_dim, 
		HL_max_eval_fitting_epochs, 
		HL_gamma, 
		model_type=HL_model_type, 
		position_of_goals=position_of_goals, 
		position_of_holes=position_of_holes,
		num_frame_stack=HL_num_frame_stack)
	exact_policy_algorithm = ExactPolicyEvaluator(HL_action_space_map, HL_gamma, env=env, frame_skip=HL_frame_skip, num_frame_stack=HL_num_frame_stack, pic_size = HL_pic_size)
	return best_response_algorithm, fitted_off_policy_evaluation_algorithm, exact_policy_algorithm

# Setup of the program:
def problem_setup(HL_policy_old, best_response_algorithm, online_convex_algorithm, fitted_off_policy_evaluation_algorithm, exact_policy_algorithm):
	online_convex_algorithm = ExponentiatedGradient(HL_lambda_bound, len(HL_constraints), HL_eta, starting_lambda=HL_starting_lambda)
	exploratory_policy_old = StochasticPolicy(HL_policy_old, 
		HL_action_space_dim, 
		exact_policy_algorithm, 
		epsilon=HL_deviation_from_old_policy_eps, 
		prob=HL_prob)
	problem = Program(HL_constraints, 
		HL_action_space_dim, 
		best_response_algorithm, 
		online_convex_algorithm, 
		fitted_off_policy_evaluation_algorithm, 
		exact_policy_algorithm, 
		lambda_bound, 
		epsilon, 
		env, 
		max_number_of_main_algo_iterations,
		num_frame_stack,
		pic_size,)    

	lambdas = []
	policies = []
	return online_convex_algorithm, exploratory_policy_old, problem, lambdas, policies

# Data Collection:
def data_collection(problem):
	num_goal = 0
	num_hole = 0
	dataset_size = 0 
	main_tic = time.time()
	# from layer_visualizer import LayerVisualizer; LV = LayerVisualizer(exploratory_policy_old.policy.Q.model)
	for i in range(max_epochs):
		tic = time.time()
		x = env.reset()
		problem.collect(x, start=True)
		dataset_size += 1
		# if env_name in ['car']:  env.render()
		done = False
		time_steps = 0
		episode_cost = 0
		while not done:
			time_steps += 1
			# if env_name in ['car']: 
			#     exploratory_policy_old.epsilon = 1.-np.exp(-3*(i/float(max_epochs)))
			
			#LV.display_activation([problem.dataset.current_state()[np.newaxis,...], np.atleast_2d(np.eye(12)[0])], 2, 2, 0)
			action = exploratory_policy_old([problem.dataset.current_state()], x_preprocessed=False)[0]
			cost = []
			for _ in range(frame_skip):
				# if env_name in ['car']: env.render()
				x_prime, costs, done, _ = env.step(HL_action_space_map[action])
				cost.append(costs)
				if done:
					break
					cost = np.vstack([np.hstack(x) for x in cost]).sum(axis=0)
					early_done, punishment = env.is_early_episode_termination(cost=cost[0], time_steps=time_steps, total_cost=episode_cost)
			# print cost, action_space_map[action] #env.car.fuel_spent/ENGINE_POWER, env.tile_visited_count, len(env.track), env.tile_visited_count/float(len(env.track))
			done = done or early_done

			# if done and reward: num_goal += 1
			# if done and not reward: num_hole += 1
			episode_cost += cost[0] + punishment
			c = (cost[0] + punishment).tolist()
			g = cost[1:].tolist()
			if len(g) < len(HL_constraints): g=np.hstack([g,0])
			problem.collect( action,
							 x_prime, #np.dot(x_prime/255. , [0.299, 0.587, 0.114]),
							 np.hstack([c,g]).reshape(-1).tolist(),
							 done
							 ) #{(x,a,x',c(x,a), g(x,a)^T, done)}
			dataset_size += 1
			x = x_prime
			if (i % 1) == 0:
				print('Epoch: %s. Exploration probability: %s' % (i, np.round(exploratory_policy_old.epsilon,5), )) 
				print('Dataset size: %s Time Elapsed: %s. Total time: %s' % (dataset_size, time.time() - tic, time.time()-main_tic))
			# if env_name in ['car']: 
			#     print('Performance: %s/%s = %s' %  (env.tile_visited_count, len(env.track), env.tile_visited_count/float(len(env.track))))
			print('*'*20) 
			problem.finish_collection(env_name)

			if env_name in ['lake']:
				problem.dataset['x'] = problem.dataset['frames'][problem.dataset['prev_states']]
				problem.dataset['x_prime'] = problem.dataset['frames'][problem.dataset['next_states']]
				problem.dataset['g'] = problem.dataset['g'][:,0:1]
				print('x Distribution:') 
				print(np.histogram(problem.dataset['x'], bins=np.arange(HL_map_size**2+1)-.5)[0].reshape(HL_map_size,HL_map_size))

				print('x_prime Distribution:') 
				print(np.histogram(problem.dataset['x_prime'], bins=np.arange(HL_map_size**2+1)-.5)[0].reshape(HL_map_size,HL_map_size))

				print('Number episodes achieved goal: %s. Number episodes fell in hole: %s' % (-problem.dataset['c'].sum(axis=0), problem.dataset['g'].sum(axis=0)[0]))

				number_of_total_state_action_pairs = (HL_state_space_dim-np.sum(env.desc=='H')-np.sum(env.desc=='G'))*HL_action_space_dim
				number_of_state_action_pairs_seen = len(np.unique(np.hstack([problem.dataset['x'].reshape(1,-1).T, problem.dataset['a'].reshape(1,-1).T]),axis=0))
				print('Percentage of State/Action space seen: %s' % (number_of_state_action_pairs_seen/float(number_of_total_state_action_pairs)))
				return problem, number_of_state_action_pairs, number_of_total_state_action_pairs

# ======================= Batch Constrained Policy Learning ================================= #
def batch_policy_learn(problem, policies, lambdas):
	iteration = 0
	while not problem.is_over(policies, lambdas, infinite_loop=infinite_loop, calculate_gap=calculate_gap, results_name=HL_results_name, policy_improvement_name=HL_policy_improvement_name):
		iteration += 1
		K.clear_session()
		for i in range(1):

			policy_printer.pprint(policies)
			print('*'*20)
			print('Iteration %s, %s' % (iteration, i))
			print()
			if len(lambdas) == 0:
				# first iteration
				lambdas.append(online_convex_algorithm.get())
				print('lambda_{0}_{2} = {1}'.format(iteration, lambdas[-1], i))
			else:
				# all other iterations
				lambda_t = problem.online_algo()
				lambdas.append(lambda_t)
				print('lambda_{0}_{3} = online-algo(pi_{1}_{3}) = {2}'.format(iteration, iteration-1, lambdas[-1], i))

				lambda_t = lambdas[-1]
				pi_t, values = problem.best_response(lambda_t, desc='FQI pi_{0}_{1}'.format(iteration, i), exact=exact_policy_algorithm)

			policies.append(pi_t) # Updating policies
			problem.update(pi_t, values, iteration) #Evaluate C(pi_t), G(pi_t) and save
	return problem, policies, lambdas

# =========================================================================================== #
def unconstrained_hlp():
	HL_old_policy_path, HL_old_policy = learn_old_policy()
	best_response_algorithm, fitted_off_policy_evaluation_algorithm, exact_policy_algorithm = initial_setup()
	online_convex_algorithm, exploratory_policy_old, problem, lambdas, policies = problem_setup(HL_policy_old, best_response_algorithm, online_convex_algorithm, fitted_off_policy_evaluation_algorithm, exact_policy_algorithm)
	problem, number_of_state_action_pairs, number_of_total_state_action_pairs = data_collection(problem, policies, lambdas)
	pdb.set_trace()
	batch_policy_learn(problem, policies, lambdas)

# =========================================================================================== #
def constrained_hlp():
	pass

if opt==1:
	unconstrained_hlp()	
elif opt == 2:
	constrained_hlp()