# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 1
# Code authors: [Luis Santos - lmpss@kth.se]
#               [Simon Mello - smello@kth.se]

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, minotaur_stay=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.minotaur_stay            = minotaur_stay
        self.actions                  = self.__actions()
        self.states, self.state_ids   = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1, 0)
        actions[self.MOVE_DOWN]  = (1, 0)
        return actions

    def __states(self):
        states = dict()
        state_ids = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                # check if the current grid cell is an obstacle
                if self.maze[i,j] != 1:
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            states[s] = (i, j, k, l)
                            state_ids[(i, j, k, l)] = s
                            s += 1
        return states, state_ids

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future player position given current (state, action)
        row_p = self.states[state][0] + self.actions[action][0]
        col_p = self.states[state][1] + self.actions[action][1]

        # Is the future player position an impossible one ?
        hitting_maze_walls =  (row_p == -1) or (row_p == self.maze.shape[0]) or \
                              (col_p == -1) or (col_p == self.maze.shape[1]) or \
                              (self.maze[row_p,col_p] == 1)


        # Based on the impossible position check reset the player's position
        if hitting_maze_walls:
            row_p, col_p = self.states[state][0], self.states[state][1]

        loop_start = 0 if self.minotaur_stay else 1
        prob_states = []

        for action_m in range(loop_start, self.n_actions):
            row_m = self.states[state][2] + self.actions[action_m][0]
            col_m = self.states[state][3] + self.actions[action_m][1]

            hitting_maze_walls = (row_m == -1) or (row_m == self.maze.shape[0]) or \
                                 (col_m == -1) or (col_m == self.maze.shape[1])

            if not hitting_maze_walls:
                state_id = self.state_ids[(row_p, col_p, row_m, col_m)]
                prob_states.append(state_id)

        return prob_states

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                prob_states = self.__move(s,a)
                prob_next_s = 1 / len(prob_states)
                for next_s in prob_states:
                    transition_probabilities[next_s, s, a] = prob_next_s

        return transition_probabilities

    def __rewards(self):
        """ Computes the reward function for every state action pair.
        """
        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                prob_states = self.__move(s, a)
                # Compute rewards for each probable next state
                for next_s in prob_states:
                    # Reward for hitting a wall
                    if self.states[s][:2] == self.states[next_s][:2] and a != self.STAY:
                        rewards[s,a] += self.IMPOSSIBLE_REWARD
                    # Reward for reaching the exit
                    elif self.states[s][:2] == self.states[next_s][:2] and self.maze[self.states[next_s][:2]] == 2:
                        rewards[s,a] += self.GOAL_REWARD
                    # Reward for encountering the minotaur
                    elif self.states[next_s][:2] == self.states[next_s][2:]:
                        rewards[s,a] += self.IMPOSSIBLE_REWARD
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] += self.STEP_REWARD

                rewards[s,a] /= len(prob_states)

        return rewards

    def simulate(self, start, policy, method, gamma=29/30):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.state_ids[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = random.choice(self.__move(s, policy[s, t]))
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.state_ids[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = random.choice(self.__move(s, policy[s]))
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while self.maze[self.states[next_s][:2]] != 2 \
                  and self.states[next_s][:2] != self.states[next_s][2:] \
                  and random.random() < gamma:  # geometrically distributed life
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = random.choice(self.__move(s, policy[s]))
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.state_ids)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic programming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q      = np.zeros((n_states, n_actions))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1)
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1)
    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    BV  = np.zeros(n_states)
    # Iteration counter
    n   = 0
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy

def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN}

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN}

    # Size of the maze
    rows,cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[path[i][:2]].get_text().set_text('Player')
        grid.get_celld()[path[i][2:]].get_text().set_text('Minotaur')

        if i > 0:
            # update the color of the previous player position
            if path[i-1][:2] != path[i][:2]:
                grid.get_celld()[path[i-1][:2]].set_facecolor(col_map[maze[path[i-1][:2]]])
                grid.get_celld()[path[i-1][:2]].get_text().set_text('')

            # check if the player has arrived at the goal
            if maze[path[i][:2]] == 2:
                grid.get_celld()[path[i][:2]].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[path[i][:2]].get_text().set_text('Player is out')

            # update the color of the previous minotaur position
            if path[i-1][2:] != path[i][2:]:
                grid.get_celld()[path[i-1][2:]].set_facecolor(col_map[maze[path[i-1][2:]]])
                grid.get_celld()[path[i-1][2:]].get_text().set_text('')

        grid.get_celld()[path[i][:2]].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[path[i][2:]].set_facecolor(LIGHT_PURPLE)

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)

def survival_probability(maze, minotaur_stay=False, start=(0, 0, 6, 5), goal=(6, 5),
                         num_simulations=10000, method='DynProg', **kwargs):
    # create the Maze environment
    env = Maze(maze, minotaur_stay)

    if method == 'DynProg':
        sp = []
        T = kwargs.get('T', 20)
        for horizon in range(1, T+1):
            win_counter = 0
            V, policy = dynamic_programming(env, horizon)

            for simulation_idx in range(num_simulations):
                path = env.simulate(start, policy, method)
                reached_goal = path[-1][:2] == goal
                if reached_goal:
                    win_counter += 1
            sp.append(win_counter / num_simulations)

        plt.figure()
        plt.plot(range(1, T+1), sp, 'o')
        plt.title('Survival probability for horizon T')
        plt.xlabel('T')
        plt.ylabel('Probability')
        plt.grid()
        plt.xticks(np.arange(0, T+5, 5))
        plt.yticks(np.arange(0, 1.1, 0.1))

    if method == 'ValIter':
        win_counter = 0
        gamma = kwargs.get('gamma', 29/30)
        epsilon = kwargs.get('epsilon', 0.0001)
        V, policy = value_iteration(env, gamma, epsilon)

        for simulation_idx in range(num_simulations):
            path = env.simulate(start, policy, method, gamma)
            reached_goal = path[-1][:2] == goal
            if reached_goal:
                win_counter += 1

        sp = win_counter / num_simulations

    return sp

def draw_arrows(env, policy, t, minotaur_pos):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN}

    # Size of the maze
    maze = env.maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title(f'Policy map at t={t}')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]
    colored_maze[minotaur_pos[0]][minotaur_pos[1]] = LIGHT_RED

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # draw the grid before drawing the arrows
    plt.draw()

    for state_id, state in env.states.items():
        # Keep only the states matching the given minotaur position
        if state[2:] != minotaur_pos:
            continue

        player_x, player_y = state[:2]
        action_y, action_x = env.actions[policy[state_id, t]]

        cell = grid[(player_x, player_y)]
        arrow_size = 0.33*cell.get_width()

        cell_mid_x = cell.get_x() + 0.5*cell.get_width()
        cell_mid_y = cell.get_y() + 0.5*cell.get_height()

        arrow_dir_x, arrow_dir_y = arrow_size*action_x, -arrow_size*action_y
        plt.arrow(cell_mid_x - arrow_dir_x/2, cell_mid_y - arrow_dir_y/2,
                  arrow_dir_x, arrow_dir_y, width=0.005)

    return grid


if __name__ == '__main__':
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0],
    ])

    # draw_arrows() and animate_solution() should be called inside Jupyter notebook
    # env = Maze(maze, minotaur_stay=False)
    # V, policy = dynamic_programming(env, 20)
    # path = env.simulate((0, 0, 6, 5), policy, method='DynProg')
    # animate_solution(maze, path)
    # draw_arrows(env, policy, t=0, minotaur_pos=(4, 5))

    sr_nonstay = survival_probability(maze, False, method='DynProg', T=20)

    # draw_arrows() and animate_solution() should be called inside Jupyter notebook
    # env = Maze(maze, minotaur_stay=True)
    # V, policy = dynamic_programming(env, 20)
    # path = env.simulate((0, 0, 6, 5), policy, method='DynProg')
    # animate_solution(maze, path)
    # draw_arrows(env, policy, t=0, minotaur_pos=(4, 5))

    sr_stay = survival_probability(maze, True, method='DynProg', T=20)
    plt.show()

    sr_nonstay = survival_probability(maze, False, method='ValIter', gamma=29/30, epsilon=0.0001)
    print(f'Survival probability (geometric life): {sr_nonstay}')
