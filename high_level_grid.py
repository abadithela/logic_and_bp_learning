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
from env_dqns import *
class MCTS_Lake(ExtendedFrozenLake):
	def __init__(self, current_state, policy1, policy2, c_history, g_history, g1_history, g2_history, isDone= False, depth, early_termination, desc=None, map_name="8x8",is_slippery=False):
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
            pi = policy1
        else:
            pi = policy2

        if not isinstance(pi,(list,)):
            pi = [pi]

        if len(pi) == 0: return

        action = pi(self.s)[0]
        transitions = self.P[self.s][action]
        i = self.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        
        c = (-r + self.c*self.depth)/(self.depth+1)
        g = ((self.g*self.depth) + int(d and not r))/(self.depth+1)
        g1 = ((self.g*self.depth) + int(action == 0))/(self.depth+1)
        g2 = ((self.g*self.depth) + int(action == 2))/(self.depth+1)

        child = MCTS_Lake(s, policy1, policy2, c, g, g1, g2, d, (self.depth+1), early_termination)
        return child


    def find_children(self):
        pi1 = policy1
        pi2 = policy2

        if not isinstance(pi1,(list,)):
            pi1 = [pi1]

        if not isinstance(pi2,(list,)):
            pi2 = [pi2]

        if len(pi1) != 0:
            pi = pi1
            action = pi(self.s)[0]
            transitions = self.P[self.s][action]
            i = self.categorical_sample([t[0] for t in transitions], self.np_random)
            p, s, r, d= transitions[i]

            c = (-r + self.c*self.depth)/(self.depth+1)
            g = ((self.g*self.depth) + int(d and not r))/(self.depth+1)
            g1 = ((self.g*self.depth) + int(action == 0))/(self.depth+1)
            g2 = ((self.g*self.depth) + int(action == 2))/(self.depth+1)

            child1 = MCTS_Lake(s, policy1, policy2, c, g, g1, g2, d, (self.depth+1), early_termination)
        else:
            child1 = None

        if len(pi2) != 0:
            pi = pi2
            action = pi(self.s)[0]
            transitions = self.P[self.s][action]
            i = self.categorical_sample([t[0] for t in transitions], self.np_random)
            p, s, r, d= transitions[i]

            c = (-r + self.c*self.depth)/(self.depth+1)
            g = ((self.g*self.depth) + int(d and not r))/(self.depth+1)
            g1 = ((self.g*self.depth) + int(action == 0))/(self.depth+1)
            g2 = ((self.g*self.depth) + int(action == 2))/(self.depth+1)

            child2 = MCTS_Lake(s, policy1, policy2, c, g, g1, g2, d, (self.depth+1), early_termination)
        else:
            child2 = None

        return child1, child2

    def reward():
        return(self.c, [self.g, self.g1, self.g2])

    def is_terminal(tau1=0.1, tau2=0.1):
        if self.is_constraint1_violated(tau1) and self.is_constraint2_violated(tau2):
            return True
        elif self.isDone:
            return True
        else: 
            return False
            
    def is_constraint1_violated(tau1):
        if self.g1 - tau1 > 0:
            return True
        else
            return False

    def is_constraint2_violated(tau2):
        if self.g2 - tau2 > 0:
            return True
        else
            return False
    	
def play_game():
    def play_game():
    trace=[]
    tree = MCTS()
    # save the trace
    output_dir = os.getcwd()+'/saved_traces/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = 'sim_trace.p'
    filepath = output_dir + filename
    gridworld = new_World()
    gridworld.setup_world()
    trace = save_scene(gridworld,trace) # save initial scene
    #print(gridworld.print_state())
    # gridworld = new_World()
    # gridworld.setup_world()
    acts = ['mergeL','stay','move', 'mergeR']
    ego_trace = {"x": [], "y": [], "v": []}
    env_trace = {"x": [], "y": [], "v": []}
    for agent in gridworld.env_agents:
        append_trace(env_trace, agent)
    for agent in gridworld.ego_agents:
        append_trace(ego_trace, agent)

    game_trace = [] # Same as ego_trace and env_trace condensed into one step with env going first
    k = 0 #  Time stamp
    # Initial step by environment:
    for agent in gridworld.env_agents:
        gridworld.env_take_step(agent,'move')
    for agent in gridworld.env_agents:
        append_trace(env_trace, agent)
        trace = save_scene(gridworld,trace) # save first env action
    gridworld.print_state()
    while True:
        gridworld.ego_take_input('mergeR')  # Ego action
        for agent in gridworld.ego_agents:
            append_trace(ego_trace, agent)
        game_trace.append(deepcopy(gridworld))
        grid_term = gridworld.is_terminal()
        trace = save_scene(gridworld,trace)
        gridworld.print_state()
        if grid_term:
            if k==0:
                print("Poor initial choices; no MCTS rollouts yet")
            for agent in gridworld.ego_agents:
                if gridworld.width == agent.x and agent.y == 1:
                    print('Did not merge; end of road')
            else:
                print("Goal reached; ego successfully merged!")
            break
        else:
            k = k+1
        gridworldnew = deepcopy(gridworld)
        for k in range(50):
            #print("Rollout: ", str(k+1))
            tree.do_rollout(gridworldnew)
        gridworldnew = tree.choose(gridworldnew) # Env action
        #import pdb; pdb.set_trace()
        # sanity_chk_ego_same(grid_new, gridworld)
        newx = gridworldnew.env_agents[0].x
        oldx = gridworld.env_agents[0].x
        newy = gridworldnew.env_agents[0].y
        oldy = gridworld.env_agents[0].y
        if newx == oldx:
            action = 'stay'
        elif newy != oldy:
            action = 'mergeR'
        else:
            action = 'move'
        for agent in gridworld.env_agents:
            gridworld.env_take_step(agent,action)
        for agent in gridworld.env_agents:
            append_trace(env_trace, agent)
        trace = save_scene(gridworld,trace)
        gridworld.print_state()
        grid_term = gridworld.is_terminal()
    save_trace(filepath,trace)
    return ego_trace, env_trace, game_trace

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
