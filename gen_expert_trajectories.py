from grid_world import State, Action, TransitionFunction
from grid_world import create_obstacles, obstacle_movement, sample_start
from itertools import product
import numpy as np
import argparse
import h5py
import pdb
import pickle
import random
import sys
import os

from load_expert_traj import recursively_save_dict_contents_to_group

def oned_to_onehot(action_delta, n):
    action_onehot = np.zeros(n,)
    action_onehot[action_delta] = 1.0

    return action_onehot

def save_expert_traj_dict_to_txt(traj_data_dict, save_dir,
                                 num_actions, num_goals,
                                 pickle_data=None):
    for path_key, traj in traj_data_dict.items():
        # File to save trajectory in.
        filename = os.path.join(save_dir, path_key)
        with open(filename, 'w') as f:
            traj_len = len(traj['state'])
            for i in range(traj_len):
                # Write state
                f.write(' '.join([str(e) for e in traj['state'][i]]) + '\n')

                # Write action
                f.write(' '.join([str(e)
                    for e in oned_to_onehot(traj['action'][i],
                                            num_actions)])+'\n')

                # Write goal
                f.write(' '.join([str(g)
                    for g in oned_to_onehot(traj['goal'][i], num_goals)])+'\n')

        print("Did save results to: {}".format(filename))

    if pickle_data is not None:
        pickle_filepath = os.path.join(save_dir, 'env_data.pkl')
        with open(pickle_filepath, 'wb') as pickle_f:
            pickle.dump(pickle_data, pickle_f, protocol=2)
        print("Did save pickle data {}".format(pickle_filepath))

def save_expert_traj_dict_to_h5(traj_data_dict, save_dir,
                                h5_filename='expert_traj.h5'):
    h5_f = h5py.File(os.path.join(save_dir, h5_filename), 'w')
    recursively_save_dict_contents_to_group(h5_f, '/', traj_data_dict)
    h5_f.flush()
    h5_f.close()
    print("Did save data to {}".format(os.path.join(save_dir, h5_filename)))

def gen_L(grid_width, grid_height, path='L_expert_trajectories'):
    ''' Generates trajectories of shape L, with right turn '''
    t = 3
    n = 2
    N = 200

    obstacles = create_obstacles(grid_width, grid_height)
    set_diff = list(set(product(tuple(range(3, grid_width-3)),
                                tuple(range(3, grid_height-3)))) \
                                        - set(obstacles))

    T = TransitionFunction(grid_width, grid_height, obstacle_movement)
    expert_data_dict = {}
    # Number of goals is the same as number of actions
    num_actions, num_goals = 4, 4
    env_data_dict = {'num_actions': num_actions, 'num_goals': num_goals}

    for i in range(N):
        path_key = str(i)
        expert_data_dict[path_key] = {'state': [], 'action': [], 'goal': []}

        for j in range(n):
            if j == 0:
                action = Action(random.choice(range(0, num_actions)))
                state = State(sample_start(set_diff), obstacles)
            else: # take right turn
                if action.delta == 0:
                    action = Action(3)
                elif action.delta == 1:
                    action = Action(2)
                elif action.delta == 2:
                    action = Action(0)
                elif action.delta == 3:
                    action = Action(1)
                else:
                    raise ValueError("Invalid action delta {}".format(
                        action.delta))
            for k in range(t):
                expert_data_dict[path_key]['state'].append(state.state)
                expert_data_dict[path_key]['action'].append(action.delta)
                expert_data_dict[path_key]['goal'].append(action.delta)
                state = T(state, action, j)

    return env_data_dict, expert_data_dict, obstacles, set_diff

def gen_sq_rec(grid_width, grid_height, path='SR_expert_trajectories'):
    ''' Generates squares if starting in quadrants 1 and 4, and rectangles if
    starting in quadransts 2 and 3 '''
    N = 200

    obstacles = create_obstacles(grid_width, grid_height)

    if not os.path.exists(path):
        os.makedirs(path)

    T = TransitionFunction(grid_width, grid_height, obstacle_movement)

    for i in range(N):
        filename = os.path.join(path, str(i) + '.txt')
        f = open(filename, 'w')
        half = random.choice(range(0,2))
        if half == 0: # left half
            set_diff = list(set(product(tuple(range(0, (grid_width/2)-3)),
                                        tuple(range(1, grid_height)))) \
                                                - set(obstacles))
            start_loc = sample_start(set_diff)
        elif half == 1: # right half
            set_diff = list(set(product(
                tuple(range(grid_width/2, grid_width-2)),
                tuple(range(2, grid_height)))) - set(obstacles))
            start_loc = sample_start(set_diff)

        state = State(start_loc, obstacles)

        if start_loc[0] >= grid_width/2: # quadrants 1 and 4
            # generate 2x2 square clockwise
            t = 2
            n = 4
            delta = 3

            for j in range(n):
                for k in range(t):
                    action = Action(delta)
                    # Write state
                    f.write(' '.join([str(e) for e in state.state]) + '\n')
                    # Write action
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 4)]) + '\n')
                    # Write c[t]s
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 4)]) + '\n')
                    state = T(state, action, j*2 + k)

                if delta == 3:
                    delta = 1
                elif delta == 1:
                    delta = 2
                elif delta == 2:
                    delta = 0

        else: # quadrants 2 and 3
            # generate 3x1 rectangle anti-clockwise
            t = [1,3,1,3]
            delta = 1

            for j in range(len(t)):
                for k in range(t[j]):
                    action = Action(delta)
                    # Write state
                    f.write(' '.join([str(e) for e in state.state]) + '\n')
                    # Write action
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 4)]) + '\n')
                    # Write c[t]s
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 4)]) + '\n')
                    state = T(state, action, sum(t[0:j]) + k)

                if delta == 1:
                    delta = 3
                elif delta == 3:
                    delta = 0
                elif delta == 0:
                    delta = 2


def gen_sq_rec_2(grid_width, grid_height, path='SR2_expert_trajectories'):
    ''' Generates squares if starting in quadrants 1 and 4, and rectangles if
    starting in quadransts 2 and 3 '''
    N = 200

    obstacles = create_obstacles(grid_width, grid_height)

    if not os.path.exists(path):
        os.makedirs(path)

    T = TransitionFunction(grid_width, grid_height, obstacle_movement)

    for i in range(N):
        filename = os.path.join(path, str(i) + '.txt')
        f = open(filename, 'w')
        half = random.choice(range(0,2))
        if half == 0: # left half
            set_diff = list(set(product(tuple(range(0, (grid_width/2)-3)),
                                        tuple(range(1, grid_height)))) \
                                                - set(obstacles))
            start_loc = sample_start(set_diff)
        elif half == 1: # right half
            set_diff = list(set(product(
                tuple(range(grid_width/2, grid_width-2)),
                tuple(range(2, grid_height)))) - set(obstacles))
            start_loc = sample_start(set_diff)

        state = State(start_loc, obstacles)

        if start_loc[0] >= grid_width/2: # quadrants 1 and 4
            # generate 2x2 square clockwise
            t = 2
            n = 4
            delta = 3
            cs = [0,0,1,1]

            for j in range(n):
                for k in range(t):
                    action = Action(delta)
                    # Write state
                    f.write(' '.join([str(e)
                        for e in state.state]) + '\n')
                    # Write action
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 4)]) + '\n')
                    # Write c[t]s
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(cs[j], 2)]) + '\n')
                    state = T(state, action, j*2 + k)

                if delta == 3:
                    delta = 1
                elif delta == 1:
                    delta = 2
                elif delta == 2:
                    delta = 0

        else: # quadrants 2 and 3
            # generate 3x1 rectangle clockwise
            t = [1,3,1,3]
            delta = 3
            cs = [0,0,1,1]

            for j in range(len(t)):
                for k in range(t[j]):
                    action = Action(delta)
                    # Write state
                    f.write(' '.join([str(e) for e in state.state]) + '\n')
                    # Write action
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 4)]) + '\n')
                    # Write c[t]s
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(cs[j], 2)]) + '\n')
                    state = T(state, action, sum(t[0:j]) + k)

                if delta == 3:
                    delta = 1
                elif delta == 1:
                    delta = 2
                elif delta == 2:
                    delta = 0


def gen_diverse_trajs(grid_width, grid_height):
    '''Generate diverse trajectories in a 21x21 grid with 4 goals.

    Return: Dictionary with keys as text filenames and values as dictionary.
        Each value dictionary contains two keys, 'states' with a list of states
        as value, and 'actions' with list of actions as value.
    '''

    assert grid_width == 21 and grid_height == 21, "Incorrect grid width height"
    N = 20
    goals = [(0,0), (20,20), (20,0), (0,20)]
    n_goals = len(goals)

    obstacles = create_obstacles(21, 21, 'diverse')

    T = TransitionFunction(grid_width, grid_height, obstacle_movement)

    set_diff = list(set(product(tuple(range(7,13)),tuple(range(7,13)))) \
            - set(obstacles))
    expert_data_dict = {}
    env_data_dict = {'num_actions': 8, 'num_goals': n_goals}

    for n in range(N):

        start_state = State(sample_start(set_diff), obstacles)

        for g in range(n_goals): # loop over goals
            # path 1 - go up/down till boundary and then move right/left

            if g == 0 or g == 2: # do path 1 only for goal 0 and goal 2

                state = start_state
                path_key = str(n) + '_' + str(g) + '_' + str(1)  + '.txt'
                expert_data_dict[path_key] = {
                        'state': [], 'action': [], 'goal': []}

                delta = 0 if g < 2 else 1
                action = Action(delta)

                while state.state[1] != grid_height-1 and state.state[1] != 0:
                    expert_data_dict[path_key]['state'].append(state.state)
                    expert_data_dict[path_key]['action'].append(action.delta)
                    expert_data_dict[path_key]['goal'].append(g)
                    state = T(state, action, 0)

                delta = 3 if g == 0 or g == 3 else 2
                action = Action(delta)

                while state.state[0] != grid_width-1 and state.state[0] != 0:
                    expert_data_dict[path_key]['state'].append(state.state)
                    expert_data_dict[path_key]['action'].append(action.delta)
                    expert_data_dict[path_key]['goal'].append(g)
                    state = T(state, action, 0)

                assert(state.coordinates in goals)

            # path 2 - go right/left till boundary and then move up/down

            if g == 1: # do path 2 only for goal 1

                state = start_state
                path_key = str(n) + '_' + str(g) + '_' + str(2)  + '.txt'
                expert_data_dict[path_key] = {'state': [], 'action': [], 'goal': []}

                delta = 3 if g == 0 or g == 3 else 2
                action = Action(delta)

                while state.state[0] != grid_width-1 and state.state[0] != 0:
                    expert_data_dict[path_key]['state'].append(state.state)
                    expert_data_dict[path_key]['action'].append(action.delta)
                    expert_data_dict[path_key]['goal'].append(g)
                    state = T(state, action, 0)

                delta = 0 if g < 2 else 1
                action = Action(delta)

                while state.state[1] != grid_height-1 and state.state[1] != 0:
                    expert_data_dict[path_key]['state'].append(state.state)
                    expert_data_dict[path_key]['action'].append(action.delta)
                    expert_data_dict[path_key]['goal'].append(g)
                    state = T(state, action, 0)

                assert(state.coordinates in goals)

            # path 3 - go diagonally till obstacle and then
            #          move up/down if x > 10 or right/left if y > 10
            #          and then move right/left or up/down till goal

            if g == 3: # do path 3 only for goal 3

                state = start_state
                path_key = str(n) + '_' + str(g) + '_' + str(3)  + '.txt'
                expert_data_dict[path_key] = {'state': [], 'action': [], 'goal': []}

                delta = g + 4
                action = Action(delta)

                while True:
                    new_state = T(state, action, 0)
                    if new_state.coordinates == state.coordinates:
                        break
                    expert_data_dict[path_key]['state'].append(state.state)
                    expert_data_dict[path_key]['action'].append(action.delta)
                    expert_data_dict[path_key]['goal'].append(g)
                    state = new_state

                if T(state, Action(2), 0).coordinates == state.coordinates \
                    or T(state, Action(3), 0).coordinates == state.coordinates:

                    delta = 0 if g < 2 else 1
                    action = Action(delta)

                    while state.state[1] != grid_height-1 and state.state[1] != 0:
                        expert_data_dict[path_key]['state'].append(state.state)
                        expert_data_dict[path_key]['action'].append(action.delta)
                        expert_data_dict[path_key]['goal'].append(g)
                        state = T(state, action, 0)

                    delta = 3 if g == 0 or g == 3 else 2
                    action = Action(delta)

                    while state.state[0] != grid_width-1 and state.state[0] != 0:
                        expert_data_dict[path_key]['state'].append(state.state)
                        expert_data_dict[path_key]['action'].append(action.delta)
                        expert_data_dict[path_key]['goal'].append(g)
                        state = T(state, action, 0)

                else:

                    delta = 3 if g == 0 or g == 3 else 2
                    action = Action(delta)

                    while state.state[0] != grid_width-1 and state.state[0] != 0:
                        expert_data_dict[path_key]['state'].append(state.state)
                        expert_data_dict[path_key]['action'].append(action.delta)
                        expert_data_dict[path_key]['goal'].append(g)
                        state = T(state, action, 0)

                    delta = 0 if g < 2 else 1
                    action = Action(delta)

                    while state.state[1] != grid_height-1 and state.state[1] != 0:
                        expert_data_dict[path_key]['state'].append(state.state)
                        expert_data_dict[path_key]['action'].append(action.delta)
                        expert_data_dict[path_key]['goal'].append(g)
                        state = T(state, action, 0)

                assert(state.coordinates in goals)

    return env_data_dict, expert_data_dict, obstacles, set_diff

def main(args):

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    expert_data_dict = None

    if args.data_type == 'gen_L':
        env_data_dict, expert_data_dict, obstacles, set_diff = gen_L(
                args.width, args.height, path=args.save_dir)
    elif args.data_type == 'gen_square_rect':
        gen_sq_rec(args.width, args.height, path=args.save_dir)
    elif args.data_type == 'gen_square_rect_2':
        gen_sq_rec_2(args.width, args.height, path=args.save_dir)
    elif args.data_type == 'gen_diverse_trajs':
        env_data_dict, expert_data_dict, obstacles, set_diff = gen_diverse_trajs(
                args.width, args.height)
    else:
        raise ValueError("Undefined value")

    if args.save_format == 'h5':
        data_to_save = {
            'expert_traj': expert_data_dict,
            'obstacles': obstacles,
            'set_diff': set_diff,
            'env_data': env_data_dict,
        }

    if expert_data_dict is not None:
        if args.save_format == 'text':
            save_expert_traj_dict_to_txt(expert_data_dict,
                                         env_data_dict['num_actions'],
                                         env_data_dict['num_goals'],
                                         args.save_dir,
                                         pickle_data=(obstacles, set_diff))
        elif args.save_format == 'h5':
            save_expert_traj_dict_to_h5(data_to_save, args.save_dir)
        else:
            raise ValueError("Incorrect save format {}".format(args.save_format))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate expert data")
    parser.add_argument('--data_type', type=str, default='gen_L',
                        choices=['gen_L',
                                 'gen_square_rect',
                                 'gen_square_rect_2',
                                 'gen_diverse_trajs'],
                        help='Type of data to be generated.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save expert data in.')
    parser.add_argument('--width', type=int, default=11,
                        help='Gridworld environment width.')
    parser.add_argument('--height', type=int, default=11,
                        help='Gridworld environment height.')
    parser.add_argument('--save_format', type=str, default='text',
                        choices=['text', 'h5'],
                        help='Format to save expert data in.')

    args = parser.parse_args()
    main(args)
