from grid_world import State, Action, TransitionFunction
from grid_world import create_obstacles, obstacle_movement, sample_start
from itertools import product
import numpy as np
import random
import sys
import os

def oned_to_onehot(action_delta, n):
    action_onehot = np.zeros(n,)
    action_onehot[action_delta] = 1.0

    return action_onehot

def gen_L(grid_width, grid_height, path='L_expert_trajectories'):
    ''' Generates trajectories of shape L, with right turn '''
    t = 3
    n = 2
    N = 200

    obstacles = create_obstacles(grid_width, grid_height)
    set_diff = list(set(product(tuple(range(3, grid_width-3)),
                                tuple(range(3, grid_height-3)))) \
                                        - set(obstacles))


    if not os.path.exists(path):
        os.makedirs(path)

    T = TransitionFunction(grid_width, grid_height, obstacle_movement)

    for i in range(N):
        filename = os.path.join(path, str(i) + '.txt')
        f = open(filename, 'w')
        for j in range(n):
            if j == 0:
                action = Action(random.choice(range(0,4)))
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
            for k in range(t):
                # Write state
                f.write(' '.join([str(e)
                    for e in state.state]) + '\n')
                # Write action
                f.write(' '.join([str(e)
                    for e in oned_to_onehot(action.delta, 4)]) + '\n')
                # Write c[t]
                f.write(' '.join([str(e)
                    for e in oned_to_onehot(action.delta, 4)]) + '\n')
                state = T(state, action, j)

        f.close()

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


def gen_diverse_trajs(grid_width, grid_height, path='diverse_path_trajs'):
    ''' Generate diverse trajectories in a 21x21 grid with 4 goals '''

    N = 20
    goals = [(0,0), (20,20), (20,0), (0,20)]
    n_goals = len(goals)

    obstacles = create_obstacles(21, 21, 'diverse')

    if not os.path.exists(path):
        os.makedirs(path)

    T = TransitionFunction(grid_width, grid_height, obstacle_movement)

    set_diff = list(set(product(tuple(range(7,13)),tuple(range(7,13)))) - set(obstacles))

    for n in range(N):

        start_state = State(sample_start(set_diff), obstacles)

        for g in range(n_goals): # loop over goals
            # path 1 - go up/down till boundary and then move right/left

            state = start_state

            filename = os.path.join(
                    path, str(n) + '_' + str(g) + '_' + str(1)  + '.txt')
            f = open(filename,'w')

            if g < 2:
                delta = 0
            else:
                delta = 1

            action = Action(delta)

            while state.state[1] != grid_height-1 and state.state[1] != 0:
                # Write state
                f.write(' '.join([str(e) for e in state.state]) + '\n')
                # Write action
                f.write(' '.join([str(e)
                    for e in oned_to_onehot(action.delta, 8)]) + '\n')
                state = T(state, action, 0)

            if g == 0 or g == 3:
                delta = 3
            else:
                delta = 2

            action = Action(delta)

            while state.state[0] != grid_width-1 and state.state[0] != 0:
                # Write state
                f.write(' '.join([str(e) for e in state.state]) + '\n')
                # Write action
                f.write(' '.join([str(e)
                    for e in oned_to_onehot(action.delta, 8)]) + '\n')
                state = T(state, action, 0)


            f.close()

            assert(state.coordinates in goals)


            # path 2 - go right/left till boundary and then move up/down

            state = start_state

            filename = os.path.join(
                    path, str(n) + '_' + str(g) + '_' + str(2)  + '.txt')
            f = open(filename,'w')

            if g == 0 or g == 3:
                delta = 3
            else:
                delta = 2

            action = Action(delta)

            while state.state[0] != grid_width-1 and state.state[0] != 0:
                # Write state
                f.write(' '.join([str(e)
                    for e in state.state]) + '\n')
                # Write action
                f.write(' '.join([str(e)
                    for e in oned_to_onehot(action.delta, 8)]) + '\n')
                state = T(state, action, 0)

            if g < 2:
                delta = 0
            else:
                delta = 1

            action = Action(delta)

            while state.state[1] != grid_height-1 and state.state[1] != 0:
                # Write state
                f.write(' '.join([str(e) for e in state.state]) + '\n')
                # Write action
                f.write(' '.join([str(e)
                    for e in oned_to_onehot(action.delta, 8)]) + '\n')
                state = T(state, action, 0)

            assert(state.coordinates in goals)

            f.close()

            # path 3 - go diagonally till obstacle and then
            #          move up/down if x > 10 or right/left if y > 10
            #          and then move right/left or up/down till goal


            state = start_state

            filename = os.path.join(
                    path, str(n) + '_' + str(g) + '_' + str(3)  + '.txt')
            f = open(filename,'w')

            delta = g + 4
            action = Action(delta)

            while True:
                new_state = T(state, action, 0)
                if new_state.coordinates == state.coordinates:
                    break
                # Write state
                f.write(' '.join([str(e) for e in state.state]) + '\n')
                # Write action
                f.write(' '.join([str(e)
                    for e in oned_to_onehot(action.delta, 8)]) + '\n')
                state = new_state

            if T(state, Action(2), 0).coordinates == state.coordinates \
                or T(state, Action(3), 0).coordinates == state.coordinates:

                if g < 2:
                    delta = 0
                else:
                    delta = 1

                action = Action(delta)

                while state.state[1] != grid_height-1 and state.state[1] != 0:
                    # Write state
                    f.write(' '.join([str(e) for e in state.state]) + '\n')
                    # Write action
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 8)]) + '\n')
                    state = T(state, action, 0)

                if g == 0 or g == 3:
                    delta = 3
                else:
                    delta = 2

                action = Action(delta)

                while state.state[0] != grid_width-1 and state.state[0] != 0:
                    # Write state
                    f.write(' '.join([str(e) for e in state.state]) + '\n')
                    # Write action
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 8)]) + '\n')
                    state = T(state, action, 0)

            else:

                if g == 0 or g == 3:
                    delta = 3
                else:
                    delta = 2

                action = Action(delta)

                while state.state[0] != grid_width-1 and state.state[0] != 0:
                    # Write state
                    f.write(' '.join([str(e) for e in state.state]) + '\n')
                    # Write action
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 8)]) + '\n')
                    state = T(state, action, 0)

                if g < 2:
                    delta = 0
                else:
                    delta = 1

                action = Action(delta)

                while state.state[1] != grid_height-1 and state.state[1] != 0:
                    # Write state
                    f.write(' '.join([str(e) for e in state.state]) + '\n')
                    # Write action
                    f.write(' '.join([str(e)
                        for e in oned_to_onehot(action.delta, 8)]) + '\n')
                    state = T(state, action, 0)


            assert(state.coordinates in goals)

            f.close()


def main():
    if int(sys.argv[1]) == 0:
        gen_L(12,12)
    elif int(sys.argv[1]) == 1:
        gen_sq_rec(12,12)
    elif int(sys.argv[1]) == 2:
        gen_sq_rec_2(12,12)
    elif int(sys.argv[1]) == 3:
        gen_diverse_trajs(21,21)
    else:
        print 'Undefined arguement!'

if __name__ == '__main__':
    main()
