import random
import numpy as np
from itertools import product

def create_obstacles(width, height, env_name=None):
    #return [(4,6),(9,6),(14,6),(4,12),(9,12),(14,12)] # 19 x 19
    #return [(3,5),(7,5),(11,5),(3,10),(7,10),(11,10)] # 17 x 17
    #return [(3,4),(6,4),(9,4),(3,9),(6,9),(9,9)] # 15 x 15
    #return [(4,4),(7,4),(4,8),(7,8)] # 13 x 13
    #return [(3,3),(6,3),(3,6),(6,6)] # 12 x 12
    if env_name == 'diverse':
        
        obstacles = []
        obs_starts = [1,15]
        obs_ends = [5,19]
        assert(len(obs_starts) == len(obs_ends))
        for i in range(len(obs_starts)):
            for j in range(len(obs_starts)):
                product_iter = product(range(obs_starts[i], obs_ends[i]+1), 
                                        range(obs_starts[j], obs_ends[j]+1))
                for k in product_iter:
                    obstacles.append(k)

        return obstacles

    else:
        return []

def obstacle_movement(t):
#    if t % 6 == 0:
#        return (0,1) # move up
#    elif t % 6 == 1:
#        return (1,0) # move right
#    elif t % 6 == 2:
#        return (1,0) # move right
#    elif t % 6 == 3:
#        return (0,-1) # move down
#    elif t % 6 == 4:
#        return (-1,0) # move left
#    elif t % 6 == 5:
#        return (-1, 0) # move left

    return (0,0)

def sample_start(set_diff):
    return random.choice(set_diff)

class State():
    def __init__(self, coordinates, list_of_obstacles, feat_type='view', view_size=3):
        #coordinates - tuple, list_of_obstacles - list of tuples
        assert(len(coordinates) == 2)
        self.coordinates = coordinates
        self.n_obs = 0
        for obs in list_of_obstacles:
            assert(len(obs) == 2)
            self.n_obs += 1

        self.list_of_obstacles = list_of_obstacles
        self.set_features(feat_type=feat_type,view_size=view_size)

    def set_features(self, feat_type='view', view_size=3):
        
        if feat_type == 'view':

            view_size = int(view_size)
            self.state = np.zeros(2 + view_size*view_size - 1)
            self.state[0] = self.coordinates[0]
            self.state[1] = self.coordinates[1]
            count = 0
            for i in range(-(view_size/2), view_size/2):
                for j in range(-(view_size/2), view_size/2):
                    if i == 0 and j == 0:
                        continue
                    count += 1
                    if (self.state[0]+i, self.state[1]+j) in self.list_of_obstacles:
                        self.state[2+count] = 1

        elif feat_type == 'all':

            self.state = np.zeros(2*(self.n_obs+1))
            self.state[0] = self.coordinates[0]
            self.state[1] = self.coordinates[1]
            for i in range(1,len(self.list_of_obstacles)+1):
                self.state[2*i] = self.list_of_obstacles[i-1][0]
                self.state[2*i+1] = self.list_of_obstacles[i-1][1]
        



class Action():
    def __init__(self, delta):
        #delta - number (integer)
        #assert(delta in (0,1,2,3,4))
        #assert(delta in (0,1,2,3))
        self.num_actions = 8
        self.delta = delta
        assert(delta in tuple(range(self.num_actions)))

    @staticmethod
    def oned_to_twod(delta, diagonal_allowed=True):
        #assert(delta in (0,1,2,3,4))
        #assert(delta in (0,1,2,3))
        assert delta in tuple(range(self.num_actions)), "Invalid action index."

        #if delta == 0:
            #return (0,0) # no movement
        if delta == 0:
            return (0,1) # up
        elif delta == 1:
            return (0,-1) # down
        elif delta == 2:
            return (-1,0) # left
        elif delta == 3:
            return (1,0) # right
        elif delta == 4:
            return (1,1) # north-east
        elif delta == 5:
            return (-1,1) # north-west
        elif delta == 6:
            return (-1,-1) # south-west
        elif delta == 7:
            return (1,-1) # south-east

class TransitionFunction():
    def __init__(self, width, height, obs_func):
        # height - number (integer), width - number (integer),
        # list_of_obstacles - list of tuples
        #assert(height >= 16)
        #assert(width >= 16)
        self.height = height
        self.width = width
        self.obs_func = obs_func

    def __call__(self, state, action, t):
        delta = Action.oned_to_twod(action.delta)
        # reward is computed later, t+1 is the correct time to compute
        # new obstacles
        t = t+1
        new_list_of_obstacles = []
        obs_delta = self.obs_func(t)
        for obs in state.list_of_obstacles:
            new_obs = (obs[0] + obs_delta[0], obs[1]+obs_delta[1])
            if new_obs[0] >= self.width or new_obs[0] < 0 \
                    or new_obs[1] >= self.height or new_obs[1] < 0:
                raise ValueError('Obstacle moved outside of the grid!!!')
            new_list_of_obstacles.append(new_obs)

        # Compute new coordinates here.
        # Stay within boundary and don't move over obstacles (new).
        new_coordinates = (max(min(state.coordinates[0] + delta[0],
                                   self.width-1), 0),
                               max(min(state.coordinates[1] + delta[1],
                                       self.height-1), 0))
        if new_coordinates in new_list_of_obstacles:
            # do stuff here - option 1. Remain where you are.
            # This should be sufficient. If not, then try moving right,
            # left down or up.
            if state.coordinates not in new_list_of_obstacles:
                # best case scenario ... stay where you are
                new_coordinates = state.coordinates
            else:
                # right
                if (max(min(state.coordinates[0]+1, self.width-1), 0),
                            state.coordinates[1]) not in new_list_of_obstacles:
                    new_coordinates = (max(min(state.coordinates[0] + 1,
                                               self.width-1),
                                           0), state.coordinates[1])
                    #print 'Warning at transition 1'
                elif (max(min(state.coordinates[0]-1, self.width-1), 0),
                              state.coordinates[1]) not in \
                                      new_list_of_obstacles: # left
                    new_coordinates = (max(min(state.coordinates[0] - 1,
                                               self.width - 1), 0),
                                       state.coordinates[1])
                    #print 'Warning at transition 2'
                elif (state.coordinates[0],
                      max(min(state.coordinates[1]-1, self.height-1),0)) \
                              not in new_list_of_obstacles: # down
                    new_coordinates = (state.coordinates[0],
                                       max(min(state.coordinates[1] - 1,
                                               self.height - 1), 0))
                    #print 'Warning at transition 3'
                elif (state.coordinates[0],
                      max(min(state.coordinates[1] + 1, self.height - 1), 0)) \
                              not in new_list_of_obstacles: # up
                    #print 'Warning at transition 4'
                    new_coordinates = (state.coordinates[0],
                                       max(min(state.coordinates[1] + 1,
                                               self.height - 1), 0))
                else:
                    raise ValueError('There is an obstacle for every transition')

        new_state = State(new_coordinates, new_list_of_obstacles)
        return new_state

class RewardFunction():
    def __init__(self, penalty, reward):
        # penalty - number (float), reward - number (float)
        self.terminal = False
        self.penalty = penalty
        self.reward = reward
        self.t = 0 # timer

    def __call__(self, state, action, c):
        self.t += 1
        if action.delta != np.argmax(c):
            return self.penalty
        else:
            return self.reward

    def reset(self, goal_1_func=None, goal_2_func=None):
        self.terminal = False
        self.t = 0


class RewardFunction_SR2():
    def __init__(self, penalty, reward, grid_width=12):
        # penalty - number (float), reward - number (float)
        self.terminal = False
        self.penalty = penalty
        self.reward = reward
        self.grid_width = grid_width
        self.action_deltas = [[3,3,1,1,2,2,0,0], [3,1,1,1,2,0,0,0]]
        self.t = 0 # timer

    def __call__(self, state, action, c):
        self.t += 1
        if state.state[0] >= self.grid_width/2:
            if action.delta != self.action_deltas[0][self.t-1]:
                return self.penalty
            else:
                return self.reward
        else:
            if action.delta != self.action_deltas[1][self.t-1]:
                return self.penalty
            else:
                return self.reward

    def reset(self, goal_1_func=None, goal_2_func=None):
        self.terminal = False
        self.t = 0
