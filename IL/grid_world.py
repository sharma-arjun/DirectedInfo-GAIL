import random
import numpy as np
from itertools import product

import pdb

def create_obstacles(width, height, env_name=None, room_size=None):
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

  elif env_name == 'room':
    assert(room_size != None)

    room_cells = []
    room_starts = [(1, 1), (1, height-room_size-1),
                   (width-room_size-1, 1), (width-room_size-1, height-room_size-1)]
    room_centres = [((room_size+1)//2, (room_size+1)//2),
                    (width-1-(room_size+1)//2, (room_size+1)//2),
                    ((room_size+1)//2, height-1-(room_size+1)//2),
                    (width-1-(room_size+1)//2, height-1-(room_size+1)//2)]

    for s in room_starts:
      for i in range(room_size):
        for j in range(room_size):
          room_cells.append((s[0] + i, s[1] + j))

    all_cells = set(product(range(0, width), range(0, height)))

    corridor_cells = []
    for i in range(room_size+1, height-room_size-1):
      corridor_cells.append((room_size, i))
      corridor_cells.append((width-room_size-1, i))

    for i in range(room_size+1, width-room_size-1):
      corridor_cells.append((i, (height-1) // 2))

    obstacles = list(all_cells - set(room_cells) - set(corridor_cells))

    return obstacles, room_cells, room_centres

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

class State(object):
    def __init__(self, coordinates, list_of_obstacles,
                feat_type='view', view_size=3):
        #coordinates - tuple, list_of_obstacles - list of tuples
        assert(len(coordinates) == 2)
        self.coordinates = coordinates
        self.n_obs = 0
        for obs in list_of_obstacles:
            assert len(obs) == 2, \
                    'Incorrect observation length {}'.format(len(obs))
            self.n_obs += 1

        self.list_of_obstacles = list_of_obstacles
        self.state = self.set_features(coordinates,
                                       feat_type=feat_type,
                                       view_size=view_size)
        self.feat_type = feat_type
        self.view_size = view_size

    def get_features(self):
        assert self.state is not None, 'Feature not set'
        return self.state

    def feature_size(self, feat_type, view_size):
        if feat_type == 'view':
            return 2 + view_size*view_size - 1
        elif feat_type == 'all':
            return 2*(self.n_obs+1)
        else:
            raise ValueError("Incorrect feat_type: {}".format(feat_type))

    def set_features(self, coordinates, feat_type='view', view_size=3):
        state = np.zeros(self.feature_size(feat_type, view_size))
        state[0], state[1] = coordinates
        if feat_type == 'view':
            view_size, count = int(view_size), 0
            for i in range(-(view_size//2), view_size//2):
                for j in range(-(view_size//2), view_size//2):
                    if i == 0 and j == 0:
                        continue
                    count += 1
                    if (state[0]+i, state[1]+j) in self.list_of_obstacles:
                        state[2+count] = 1

        elif feat_type == 'all':
            for i in range(1,len(self.list_of_obstacles)+1):
                state[2*i] = self.list_of_obstacles[i-1][0]
                state[2*i+1] = self.list_of_obstacles[i-1][1]
        else:
            raise ValueError('Incorrect feature type {}'.format(feat_type))

        return state

class StateVector(State):
    def __init__(self, coordinates_arr, list_of_obstacles,
                 feat_type='view', view_size=3):
        super(StateVector, self).__init__((1, 1),  # Fake coordinates not used
                                          list_of_obstacles,
                                          feat_type=feat_type,
                                          view_size=view_size)
        self.coordinates_arr = coordinates_arr
        self.coordinates = coordinates_arr
        self.state_arr = self.set_features_array(coordinates_arr,
                                                 feat_type=feat_type,
                                                 view_size=view_size)


    def set_features_array(self, coordinates_arr,
                           feat_type='view', view_size=3):
        batch_size = coordinates_arr.shape[0]
        state_arr = np.zeros((batch_size,
                              self.feature_size(feat_type, view_size)))
        for b in range(batch_size):
            state_arr[b, :] = self.set_features(
                    coordinates_arr[b, :],
                    feat_type=feat_type,
                    view_size=view_size)
        return state_arr

    def get_features(self):
        assert self.state_arr is not None, 'Feature not set'
        return self.state_arr

class Action(object):
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
        num_actions = 8
        assert delta in tuple(range(num_actions)), "Invalid action index."

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

class ActionVector(Action):
    def __init__(self, delta_arr):
        # Pass in a dummy action
        super(ActionVector, self).__init__(delta_arr[0])
        self.delta_arr = delta_arr

class TransitionFunction():
    def __init__(self, width, height, obs_func):
        '''Transition function for the grid world.

        height: Integer
        width: Integer
        list_of_obstacles: list of tuples
        '''
        self.height = height
        self.width = width
        self.obs_func = obs_func

    def __call__(self, state, action, t):
        if type(state) is StateVector:
            assert state.coordinates_arr.shape[0] == action.delta_arr.shape[0], \
                    "(State, Action) batch sizes do not match"
            batch_size = state.coordinates_arr.shape[0]
            new_coord_arr = np.zeros(state.coordinates.shape)
            for b in range(batch_size):
                state_coord = state.coordinates_arr[b, :]
                if type(state_coord) is not type([]):
                    state_coord = state_coord.tolist()
                action_delta = action.delta_arr[b]
                new_coord, new_list_of_obstacle = self.next_state(
                        state_coord,
                        action_delta,
                        t,
                        state.list_of_obstacles)
                new_coord_arr[b, :] = new_coord

            new_state = StateVector(new_coord_arr,
                                    new_list_of_obstacle,
                                    feat_type=state.feat_type,
                                    view_size=state.view_size)
        elif type(state) is State:
            new_coord, new_list_of_obstacle = self.next_state(
                    state.coordinates,
                    action.delta,
                    t,
                    state.list_of_obstacles)
            new_state = State(new_coord, new_list_of_obstacles,
                              feat_type=state.feat_type,
                              view_size=state.view_size)
        else:
            raise ValueError('Incorrect state type: {}'.format(type(state)))

        return new_state

    def next_state(self, state_coord, action_delta, t, list_of_obstacles):
        delta = Action.oned_to_twod(action_delta)
        # reward is computed later, t+1 is the correct time to compute
        # new obstacles
        t = t+1
        new_list_of_obstacles = []
        obs_delta = self.obs_func(t)
        for obs in list_of_obstacles:
            new_obs = (obs[0] + obs_delta[0], obs[1]+obs_delta[1])
            if new_obs[0] >= self.width or new_obs[0] < 0 \
                    or new_obs[1] >= self.height or new_obs[1] < 0:
                raise ValueError('Obstacle moved outside of the grid!!!')
            new_list_of_obstacles.append(new_obs)

        # Compute new coordinates here.
        # Stay within boundary and don't move over obstacles (new).
        new_coordinates = (max(
            min(state_coord[0] + delta[0], self.width-1), 0),
            max(min(state_coord[1] + delta[1], self.height-1), 0))

        if new_coordinates in new_list_of_obstacles:
            # do stuff here - option 1. Remain where you are.
            # This should be sufficient. If not, then try moving right,
            # left down or up.
            if state_coord not in new_list_of_obstacles:
                # best case scenario ... stay where you are
                new_coordinates = state_coord
            else:
                # right
                if (max(min(state_coord[0]+1, self.width-1), 0),
                        state_coord[1]) not in new_list_of_obstacles:
                    new_coordinates = (max(min(state_coord[0] + 1,
                        self.width-1), 0), state_coord[1])
                  #print 'Warning at transition 1'
                elif (max(min(state_coord[0]-1, self.width-1), 0),
                        state_coord[1]) not in new_list_of_obstacles: # left
                    new_coordinates = (max(min(state_coord[0] - 1,
                        self.width - 1), 0), state_coord[1])
                  #print 'Warning at transition 2'
                elif (state_coord[0], max(min(state_coord[1]-1,
                    self.height-1),0)) not in new_list_of_obstacles: # down
                    new_coordinates = (
                            state_coord[0],
                            max(min(state_coord[1]-1, self.height-1), 0)
                            )
                  #print 'Warning at transition 3'
                elif (state_coord[0], max(min(state_coord[1] + 1,
                    self.height - 1), 0)) not in new_list_of_obstacles: # up
                    #print 'Warning at transition 4'
                    new_coordinates = (
                            state_coord[0],
                            max(min(state_coord[1]+1, self.height-1), 0))
                else:
                    raise ValueError('There is an obstacle for every transition')

        return new_coordinates, new_list_of_obstacles


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

class GridWorldReward(object):
  def __init__(self, width, height, goals, obstacles):
    self.width = width
    self.height = height
    self.goals = goals
    self.obstacles = [(x[0], x[1]) for x in obstacles.tolist()]
    self.goal_reward = 100
    self.obstacle_reward = -100
    self.other_reward = -1

  def reward_at_location(self, pos, goals=None):
    pos = [int(pos[0]), int(pos[1])]
    goals = self.goals if goals is None else goals
    if pos in goals:
      return self.goal_reward
    if pos in self.obstacles:
      return self.obstacle_reward
    return self.other_reward

class ActionBasedGridWorldReward(GridWorldReward):
  def __init__(self, width, height, goals, obstacles):
    super(ActionBasedGridWorldReward, self).__init__(width, height,
                                                     goals, obstacles)
    self.corret_action_reward = 1.0
    self.incorrect_action_reward = -1.0

  def reward_at_location(self, expert_action, gen_action):
    if expert_action == gen_action:
      return self.corret_action_reward
    else:
      return self.incorrect_action_reward
