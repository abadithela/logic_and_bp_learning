# High level grid policy for MCTS
# Apurva Badithela

#############################################################################
#                                                                           #
# Basic class for grid world traffic simulation, separate from MCTS for now #
# Anushri Dixit, Apurva Badithela                                      #
# Caltech, May 2021                                                       #
#                                                                           #
#############################################################################
import frozen_lake
import random 
class MCTS_Lake(ExtendedFrozenLake):
	def __init__(self, early_termination, desc=None, map_name="8x8",is_slippery=False):
        super(MCTS_Lake, self).__init__(desc=desc, map_name=map_name, is_slippery=is_slippery)
        self.deterministic = True
        self.policy1 = None
        self.policy2 = None
        self.g1 = None
        self.g2 = None
        self.c = None
        self.safety = None

    def get_policies(self):
        return self.policy1, self.policy2
    def find_random_child(self):

    def find_children(self, policy):

    def reward():

    def is_terminal():

    def is_constraint1_violated():
        if self.g1 > 0:
            return True
        else
            return False
            
    def is_constraint2_violated():
        if self.g2 > 0:
            return True
        else
            return False
    	


if __name__ == '__main__':
    #run_random_sim(10)
    output_dir = os.getcwd()+'/saved_traces/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = 'sim_trace.p'
    filepath = output_dir + filename
    ego_trace = play_game()
    print("Robot Trajectory")
    # print(ego_trace)
    print("")
