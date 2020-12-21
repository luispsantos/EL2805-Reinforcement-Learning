# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 2
# Code authors: [Luis Santos - lmpss@kth.se]
#               [Simon Mello - smello@kth.se]

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter']

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class City:

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
    STEP_REWARD = 0
    BANK_REWARD = 10
    POLICE_REWARD = -50


    def __init__(self, city, start_state=(0, 0, 1, 2)):
        """ Constructor of the environment City.
        """
        self.city                     = city
        self.start_state              = start_state
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
        for i in range(self.city.shape[0]):
            for j in range(self.city.shape[1]):
                for k in range(self.city.shape[0]):
                    for l in range(self.city.shape[1]):
                        states[s] = (i, j, k, l)
                        state_ids[(i, j, k, l)] = s
                        s += 1
        return states, state_ids

    def police_moves(self, state):
        """ Compute possible police moves from a state.
        """
        possible_moves = []
        y_offset = clamp(state[0] - state[2], -1, 1)
        x_offset = clamp(state[1] - state[3], -1, 1)

        if x_offset == 0 and y_offset == 0:
            possible_moves.append((x_offset, y_offset))
        elif x_offset == 0 and y_offset != 0:
            possible_moves.extend([(0, 1), (0, -1), (y_offset, 0)])
        elif x_offset != 0 and y_offset == 0:
            possible_moves.extend([(0, x_offset), (1, 0), (-1, 0)])
        else:
            possible_moves.extend([(0, x_offset), (y_offset, 0)])

        return possible_moves

    def __move(self, state, action):
        """ Makes a step in the city, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the city that agent transitions to.
        """
        # reset the game if the robber has been caught
        if self.states[state][:2] == self.states[state][2:]:
            return [self.state_ids[self.start_state]]

        # Compute the future robber position given current (state, action)
        row_r = self.states[state][0] + self.actions[action][0]
        col_r = self.states[state][1] + self.actions[action][1]

        # Is the future robber position an impossible one ?
        hitting_city_walls =  (row_r == -1) or (row_r == self.city.shape[0]) or \
                              (col_r == -1) or (col_r == self.city.shape[1])

        # Based on the impossible position check reset the robber's position
        if hitting_city_walls:
            row_r, col_r = self.states[state][0], self.states[state][1]

        prob_states = []

        for action_p in self.police_moves(self.states[state]):
            row_p = self.states[state][2] + action_p[0]
            col_p = self.states[state][3] + action_p[1]
            
            hitting_city_walls = (row_p == -1) or (row_p == self.city.shape[0]) or \
                                 (col_p == -1) or (col_p == self.city.shape[1])

            if not hitting_city_walls:
                state_id = self.state_ids[(row_r, col_r, row_p, col_p)]
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
                prob_states = self.__move(s, a)
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
                    # Reward for encountering the police
                    if self.states[next_s][:2] == self.states[next_s][2:]:
                        rewards[s,a] += self.POLICE_REWARD
                    # Reward for being at a bank
                    elif self.states[s][:2] == self.states[next_s][:2] and self.city[self.states[next_s][:2]] == 1:
                        rewards[s,a] += self.BANK_REWARD
                    # Reward for taking a step to an empty cell
                    else:
                        rewards[s,a] += self.STEP_REWARD

                rewards[s,a] /= len(prob_states)

        return rewards

    def simulate(self, start, policy, method, max_timesteps=20):
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
            # Add the starting position in the city to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = random.choice(self.__move(s, policy[s, t]))
                # Add the position in the city corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.state_ids[start]
            # Add the starting position in the city to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = random.choice(self.__move(s, policy[s]))
            # Add the position in the city corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while t < max_timesteps:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = random.choice(self.__move(s, policy[s]))
                # Add the position in the city corresponding to the next state
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
        :input city env           : The city environment in which we seek to
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
        :input city env           : The city environment in which we seek to
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

def draw_city(city):

    # Map a color to each cell in the city
    col_map = {0: WHITE, 1: LIGHT_GREEN}

    # Give a color to each cell
    rows,cols    = city.shape
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The city')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = city.shape
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_city,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def animate_solution(city, path):

    # Map a color to each cell in the city
    col_map = {0: WHITE, 1: LIGHT_GREEN}

    # Size of the city
    rows,cols = city.shape

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_city,
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
        grid.get_celld()[path[i][:2]].get_text().set_text('Robber')
        grid.get_celld()[path[i][2:]].get_text().set_text('Police')

        if i > 0:
            # update the color of the previous player position
            if path[i-1][:2] != path[i][:2]:
                grid.get_celld()[path[i-1][:2]].set_facecolor(col_map[city[path[i-1][:2]]])
                grid.get_celld()[path[i-1][:2]].get_text().set_text('')

            # check if the robber has arrived at the bank
            if city[path[i][:2]] == 1:
                grid.get_celld()[path[i][:2]].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[path[i][:2]].get_text().set_text('Looting')

            # update the color of the previous police position
            if path[i-1][2:] != path[i][2:]:
                grid.get_celld()[path[i-1][2:]].set_facecolor(col_map[city[path[i-1][2:]]])
                grid.get_celld()[path[i-1][2:]].get_text().set_text('')

        grid.get_celld()[path[i][:2]].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[path[i][2:]].set_facecolor(LIGHT_PURPLE)

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)

def display_initial_value(env, lambda_range=(0.01, 1.0, 0.01), epsilon=0.0001):
    initial_v = []
    start_state_id = env.state_ids[env.start_state]

    for gamma in np.arange(*lambda_range):
        V, policy = value_iteration(env, gamma, epsilon)
        initial_v.append(V[start_state_id])

    plt.figure()
    plt.plot(np.arange(*lambda_range), initial_v)
    plt.title('Value function at the initial state')
    plt.xlabel('$\lambda$')
    plt.ylabel('$V^*$')
    plt.grid(True)

def display_policy(env, police_pos, lambda_range=(0.1, 1.0, 0.1), epsilon=0.0001):
    for gamma in np.arange(*lambda_range):
        V, policy = value_iteration(env, gamma, epsilon)
        draw_arrows(env, policy, gamma, police_pos)

def draw_arrows(env, policy, gamma, police_pos, ax=None):
    # Map a color to each cell in the city
    col_map = {0: WHITE, 1: LIGHT_GREEN}

    # Size of the city
    city = env.city
    rows, cols = city.shape

    # Create figure of the size of the city
    if ax is None:
        fig = plt.figure(figsize=(cols,rows))
        ax = plt.gca()

    # Remove the axis ticks and add title title
    ax.set_title(f'Policy map, $\lambda$={gamma:.2f}')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]
    colored_city[police_pos[0]][police_pos[1]] = LIGHT_RED

    # Create a table to color
    grid = ax.table(cellText=None,
                    cellColours=colored_city,
                    cellLoc='center',
                    loc=(0,0),
                    edges='closed')

    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Draw the grid before drawing the arrows
    plt.draw()

    for state_id, state in env.states.items():
        # Keep only the states matching the given police position
        if state[2:] != police_pos:
            continue

        player_x, player_y = state[:2]
        action_y, action_x = env.actions[policy[state_id]]

        cell = grid[(player_x, player_y)]
        arrow_size = 0.33*cell.get_width()

        cell_mid_x = cell.get_x() + 0.5*cell.get_width()
        cell_mid_y = cell.get_y() + 0.5*cell.get_height()

        arrow_dir_x, arrow_dir_y = arrow_size*action_x, -arrow_size*action_y
        ax.arrow(cell_mid_x - arrow_dir_x/2, cell_mid_y - arrow_dir_y/2,
                 arrow_dir_x, arrow_dir_y, width=0.005)

    return grid


if __name__ == '__main__':
    # Description of the city as a numpy array
    city = np.array([
        [1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
    ])

    env = City(city, start_state=(0, 0, 1, 2))

    display_initial_value(env, lambda_range=(0.01, 1.0, 0.01))
    #display_policy(env, police_pos=(1, 0))  # must be run inside Jupyter notebook
    plt.show()
