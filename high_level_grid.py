# High level grid policy for MCTS
# Apurva Badithela

#############################################################################
#                                                                           #
# Basic class for grid world traffic simulation, separate from MCTS for now #
# Anushri Dixit, Apurva Badithela                                      #
# Caltech, May 2021                                                       #
#                                                                           #
#############################################################################
from frozen_lake import *
import random 
from env_dqns import *
from mcts import MCTS, Node
from print_policy import PrintPolicy
from keras.models import load_model
from copy import deepcopy
import os
import pickle as pkl
from print_policy import PrintPolicy
from keras.models import Sequential, Model as KerasModel

class MCTS_Lake(ExtendedFrozenLake):
	def __init__(self, current_state, policy1, policy2, c_history, g_history, g1_history, g2_history, depth, isDone= False, early_termination=100, desc=None, map_name="8x8",is_slippery=False):
		super(MCTS_Lake, self).__init__(desc=desc, early_termination=early_termination, map_name=map_name, is_slippery=is_slippery)

		self.policy1 = policy1
		self.policy2 = policy2
		self.g = g_history
		self.g1 = g1_history
		self.g2 = g2_history
		self.c = c_history
		self.s = current_state
		self.isDone = isDone
		self.depth = depth

	def get_policies(self):
		return self.policy1, self.policy2

	def find_random_child(self):
		rand = randrange(0,2)
		if (rand == 0):
			pi = self.policy1
		else:
			pi = self.policy2


		if len(pi) == 0: return

		action = int(pi[self.s][1])
		transitions = self.P[self.s][action]
		i = self.categorical_sample([t[0] for t in transitions], self.np_random)
		p, s, r, d= transitions[i]
		
		c = (-r + self.c*self.depth)/(self.depth+1)
		g = ((self.g*self.depth) + int(d and not r))/(self.depth+1)
		g1 = ((self.g*self.depth) + int(action == 0))/(self.depth+1)
		g2 = ((self.g*self.depth) + int(action == 2))/(self.depth+1)

		child = MCTS_Lake(s, policy1, policy2, c, g, g1, g2, (self.depth+1), isDone=d)
		return child

	def print_lake_status(self):
		print("Position: ") 
		print(self.s)
		
	def find_children(self):
		pi1 = self.policy1
		pi2 = self.policy2


		if len(pi1) != 0:
			pi = pi1
			pdb.set_trace()
			action = int(pi[self.s][1])
			transitions = self.P[self.s][action]
			i = self.categorical_sample([t[0] for t in transitions], self.np_random)
			p, s, r, d= transitions[i]

			c = (-r + self.c*self.depth)/(self.depth+1)
			g = ((self.g*self.depth) + int(d and not r))/(self.depth+1)
			g1 = ((self.g*self.depth) + int(action == 0))/(self.depth+1)
			g2 = ((self.g*self.depth) + int(action == 2))/(self.depth+1)

			child1 = MCTS_Lake(s, policy1, policy2, c, g, g1, g2, (self.depth+1), isDone=d)
		else:
			child1 = None

		if len(pi2) != 0:
			pi = pi2
			action = int(pi[self.s][1])
			transitions = self.P[self.s][action]
			i = self.categorical_sample([t[0] for t in transitions], self.np_random)
			p, s, r, d= transitions[i]

			c = (-r + self.c*self.depth)/(self.depth+1)
			g = ((self.g*self.depth) + int(d and not r))/(self.depth+1)
			g1 = ((self.g*self.depth) + int(action == 0))/(self.depth+1)
			g2 = ((self.g*self.depth) + int(action == 2))/(self.depth+1)

			child2 = MCTS_Lake(s, policy1, policy2, c, g, g1, g2, (self.depth+1), isDone=d)
		else:
			child2 = None

		return child1, child2

	def reward(self):
		# return(self.c, [self.g, self.g1, self.g2])
		return self.c 

	def is_terminal(self, tau1=0.4, tau2=0.4, tau_s=0.1):
		if self.is_constraint1_violated(tau1) and self.is_constraint2_violated(tau2):
			return True
		elif self.is_safety_constraint_violated(tau_s):
			return True
		elif self.isDone:
			return True
		else: 
			return False
	
	def is_safety_constraint_violated(self,tau_s):
		if self.g - tau_s > 0:
			return True
		else:
			return False

	def is_constraint1_violated(self,tau1):
		if self.g1 - tau1 > 0:
			return True
		else:
			return False

	def is_constraint2_violated(self,tau2):
		if self.g2 - tau2 > 0:
			return True
		else:
			return False
def save_trace(filename,trace):
	print('Saving trace in pkl file')
	#import pdb; pdb.set_trace()
	with open(filename, 'wb') as pckl_file:
		pickle.dump(trace, pckl_file)

def play_game():
	tree = MCTS()
	# save the trace
	output_dir = os.getcwd()+'/saved_traces/'
	policy_dir = os.getcwd()+'/models/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	filename = 'sim_trace.p'
	filepath = output_dir + filename

	# initializing:
	col_pos= 3
	ncol = 8
	row_pos = 0
	start = (row_pos -1)*ncol+ col_pos # Check fromat of states

	policy1_path = "saved_pol1.csv"
	policy2_path = "saved_pol2.csv"
	policy1 = np.loadtxt(policy1_path, delimiter=',')
	policy2 = np.loadtxt(policy2_path, delimiter=',')

	c_history = 0
	g_history = 0
	g1_history = 0
	g2_history = 0
	root_node = MCTS_Lake(start, policy1, policy2, c_history, g_history, g1_history, g2_history, depth=0, early_termination=100, desc=None, map_name="8x8",is_slippery=False)
	# trace = save_scene(gridworld,trace) # save initial scene

	k = 0 #  Time stamp
	
	while True:
		trace=[root_node]
		# root_node.ego_take_input('mergeR')  # Ego action
		root_term = root_node.is_terminal()
		if root_term:
			if k==0:
				print("Poor initial choices; no MCTS rollouts yet")
			else:
				print("No. of iterations are {0}.format", k)
				save_trace(filepath, trace)
			break
		else:
			k = k+1
		root_new = deepcopy(root_node)
		for ki in range(50):
			#print("Rollout: ", str(k+1))
			tree.do_rollout(root_new)
		root_new = tree.choose(root_new) # Env action
		root_node = deepcopy(root_new) # Copying root_new to root_node
		trace.append(root_node)
		root_term = root_node.is_terminal()
	return trace

if __name__ == '__main__':
	#run_random_sim(10)
	ego_trace = play_game()
	print("Robot Trajectory")
	# print(ego_trace)
	for ei in ego_trace:
		ei.print_lake_status()

