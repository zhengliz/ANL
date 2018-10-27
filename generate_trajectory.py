import argparse
import copy
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import signal
import torch
import torch.multiprocessing as mp
from collections import deque
from maze2d import *
from model import *
from a3c_train import train
from a3c_test import test
from utils.visualization import *

import logging


def generate_trajectory(model, args, env, map_id):
    env.test_maze_idx = map_id
    state, depth = env.reset()

    map_design = env.map_design
    positions = [env.position]
    orientations = [env.orientation]

    states = [state]
    state = torch.from_numpy(state).float()

    episode_length = 0
    done = False
    action_hist = deque([3] * args.hist_size, maxlen=args.hist_size)
    actions = []
    rewards = []

    while not done:
        episode_length += 1

        ax = Variable(torch.from_numpy(np.array(action_hist)), volatile=True)
        dx = Variable(torch.from_numpy(np.array([depth])).long(), volatile=True)
        tx = Variable(torch.from_numpy(np.array([episode_length])).long(), volatile=True)

        value, logit = model((Variable(state.unsqueeze(0), volatile=True), (ax, dx, tx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.numpy()[0]

        state, reward, done, depth = env.step(action)
        states.append(state)
        positions.append(env.position)
        orientations.append(env.orientation)
        action_hist.append(action)
        actions.append(action)
        rewards.append(reward)

        done = done or episode_length >= args.max_episode_length

        idx = np.unravel_index(np.argmax(state[:4], axis=None), state[:4].shape)
        if idx == (env.orientation, env.position[0], env.position[1]):
            done = True

        state = torch.from_numpy(state).float()

    return map_design, positions, orientations, states, actions, rewards


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Active Neural Localization')

    # Environment arguments
    parser.add_argument('-l', '--max-episode-length', type=int,
                        default=30, metavar='L',
                        help='maximum length of an episode (default: 30)')
    parser.add_argument('-m', '--map-size', type=int, default=7,
                        help='''m: Size of the maze m x m (default: 7),
                                must be an odd natural number''')

    # A3C and model arguments
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num-iters', type=int, default=1000000, metavar='NS',
                        help='''number of training iterations per training thread
                                (default: 10000000)''')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-n', '--num-processes', type=int, default=8, metavar='N',
                        help='how many training processes to use (default: 8)')
    parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                        help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--hist-size', type=int, default=5,
                        help='action history size (default: 5)')
    parser.add_argument('--load', type=str, default="0",
                        help='model path to load, 0 to not reload (default: 0)')
    parser.add_argument('-e', '--evaluate', type=int, default=0,
                        help='0:Train, 1:Evaluate on test data (default: 0)')
    parser.add_argument('-d', '--dump-location', type=str, default="./saved/",
                        help='path to dump models and log (default: ./saved/)')
    parser.add_argument('-td', '--test-data', type=str,
                        default="./test_data/m7_n1000.npy",
                        help='''Test data filepath
                                (default: ./test_data/m7_n1000.npy)''')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.dump_location):
        os.makedirs(args.dump_location)

    logging.basicConfig(
        filename=args.dump_location + 'generate.log',
        level=logging.INFO)

    env = Maze2D(args)

    model = Localization_2D_A3C(args)
    model.load_state_dict(torch.load(args.load))
    model.eval()

    map_design, positions, orientations, states, actions, rewards = generate_trajectory(model, args, env, 0)

    print map_design
    clear_folder('./tmp')

    count = len(positions)

    # for i in range(count):
    #     visualize(map_design, belief=states[i], position=positions[i], orientation=orientations[i],
    #               fn='./tmp/frame_{0:03d}.png'.format(i), idx=i)
    #     if i > 0:
    #         print('frame {} to {}: {}'.format(i - 1, i, c_actions[actions[i - 1]]))
    #
    # images2video('./tmp', './output/video.mp4')

    for i in range(count):
        visualize_comparison(map_design,
                             belief_pair=(states[i], states[count - 1 - i]),
                             position_pair=(positions[i], positions[count - 1 - i]),
                             orientation_pair=(orientations[i], orientations[count - 1 - i]),
                             fn='./tmp/frame_{0:03d}.png'.format(i), idx=i)
        if i > 0:
            print('frame {} to {}: {}'.format(i - 1, i, c_actions[actions[i - 1]]))

    images2video('./tmp', './output/video.mp4')
