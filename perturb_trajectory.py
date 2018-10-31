import copy
from generate_trajectory import *
from maze2d import *
from model import *
from utils.visualization import *


def perturb_trajectory(idx, args, env, map_design, positions, orientations, states, actions,
                       model=None):

    position_pairs = [(positions[0], positions[0])]
    orientation_pairs = [(orientations[0], orientations[0])]
    state_pairs = [(states[0], states[0])]
    action_pairs = []

    state, depth = env.overwrite(map_design, positions[0], orientations[0])

    assert(np.array_equal(states[0], state))

    episode_length = len(actions)
    action_hist = deque([3] * args.hist_size, maxlen=args.hist_size)

    for i in range(idx):

        state, reward, done, depth = env.step(actions[i])
        action_hist.append(actions[i])

        assert(np.array_equal(states[i + 1], state))
        assert(positions[i + 1] == env.position)
        assert(orientations[i + 1] == env.orientation)

        position_pairs.append((positions[i + 1], positions[i + 1]))
        orientation_pairs.append((orientations[i + 1], orientations[i + 1]))
        state_pairs.append((states[i + 1], states[i + 1]))
        action_pairs.append((actions[i], actions[i]))

    state, reward, done, depth = env.step(actions[idx])
    env.posterior = env.prior
    action_hist.append(actions[idx])

    position_pairs.append((positions[idx + 1], positions[idx + 1]))
    orientation_pairs.append((orientations[idx + 1], orientations[idx + 1]))
    state_pairs.append((states[idx + 1], states[idx]))
    action_pairs.append((actions[idx], actions[idx]))

    for i in range(idx + 1, episode_length):

        if model is None:
            state, reward, done, depth = env.step(actions[i])
            action_hist.append(actions[i])

            position_pairs.append((positions[i + 1], env.position))
            orientation_pairs.append((orientations[i + 1], env.orientation))
            state_pairs.append((states[i + 1], state))
            action_pairs.append((actions[i], actions[i]))

        else:
            state = torch.from_numpy(state).float()
            ax = Variable(torch.from_numpy(np.array(action_hist)), volatile=True)
            dx = Variable(torch.from_numpy(np.array([depth])).long(), volatile=True)
            tx = Variable(torch.from_numpy(np.array([i + 1])).long(), volatile=True)

            value, logit = model((Variable(state.unsqueeze(0), volatile=True), (ax, dx, tx)))
            prob = F.softmax(logit, dim=1)
            action = prob.max(1)[1].data.numpy()[0]

            state, reward, done, depth = env.step(action)
            action_hist.append(action)

            position_pairs.append((positions[i + 1], env.position))
            orientation_pairs.append((orientations[i + 1], env.orientation))
            state_pairs.append((states[i + 1], state))
            action_pairs.append((actions[i], action))

    return position_pairs, orientation_pairs, state_pairs, action_pairs


def perturb_trajectory_observation(idx, args, env, map_design, positions, orientations, states,
                                   actions, model=None):

    position_pairs = [(positions[0], positions[0])]
    orientation_pairs = [(orientations[0], orientations[0])]
    state_pairs = [(states[0], states[0])]
    action_pairs = []

    state, depth = env.overwrite(map_design, positions[0], orientations[0])

    assert(np.array_equal(states[0], state))

    episode_length = len(actions)
    action_hist = deque([3] * args.hist_size, maxlen=args.hist_size)

    for i in range(idx):

        state, reward, done, depth = env.step(actions[i])
        action_hist.append(actions[i])

        assert(np.array_equal(states[i + 1], state))
        assert(positions[i + 1] == env.position)
        assert(orientations[i + 1] == env.orientation)

        position_pairs.append((positions[i + 1], positions[i + 1]))
        orientation_pairs.append((orientations[i + 1], orientations[i + 1]))
        state_pairs.append((states[i + 1], states[i + 1]))
        action_pairs.append((actions[i], actions[i]))

    state, reward, done, depth = env.step(actions[idx], perturb_depth=True)
    action_hist.append(actions[idx])

    position_pairs.append((positions[idx + 1], positions[idx + 1]))
    orientation_pairs.append((orientations[idx + 1], orientations[idx + 1]))
    state_pairs.append((states[idx + 1], state))
    action_pairs.append((actions[idx], actions[idx]))

    for i in range(idx + 1, episode_length):

        if model is None:
            state, reward, done, depth = env.step(actions[i])
            action_hist.append(actions[i])

            position_pairs.append((positions[i + 1], env.position))
            orientation_pairs.append((orientations[i + 1], env.orientation))
            state_pairs.append((states[i + 1], state))
            action_pairs.append((actions[i], actions[i]))

        else:
            state = torch.from_numpy(state).float()
            ax = Variable(torch.from_numpy(np.array(action_hist)), volatile=True)
            dx = Variable(torch.from_numpy(np.array([depth])).long(), volatile=True)
            tx = Variable(torch.from_numpy(np.array([i + 1])).long(), volatile=True)

            value, logit = model((Variable(state.unsqueeze(0), volatile=True), (ax, dx, tx)))
            prob = F.softmax(logit, dim=1)
            action = prob.max(1)[1].data.numpy()[0]

            state, reward, done, depth = env.step(action)
            action_hist.append(action)

            position_pairs.append((positions[i + 1], env.position))
            orientation_pairs.append((orientations[i + 1], env.orientation))
            state_pairs.append((states[i + 1], state))
            action_pairs.append((actions[i], action))

    return position_pairs, orientation_pairs, state_pairs, action_pairs


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

    # # # # # # # # # # # # # # # # # # # # # #

    flog = open('./output/{}_log_continue.txt'.format(args.load.split('/')[-1]), 'w+')
    acc_list = []
    act_list = []

    for maze_id in range(100):

        map_design, positions, orientations, states, actions, rewards = generate_trajectory(
            model, args, env, maze_id)

        for frame_id in range(0, len(actions) - 1):

            # blind at frame_id, but follow the same sequence of actions

            position_pairs, orientation_pairs, state_pairs, action_pairs = perturb_trajectory(
                frame_id, args, env, map_design, positions, orientations, states, actions)

            count = len(position_pairs)
            clear_folder('./tmp')
            for i in range(count):
                visualize_comparison(map_design,
                                     belief_pair=state_pairs[i],
                                     position_pair=position_pairs[i],
                                     orientation_pair=orientation_pairs[i],
                                     fn='./tmp/frame_{0:03d}.png'.format(i), idx=i)
                if i > 0:
                    print('frame {} to {}: before {}, after {}'.format(
                        i - 1, i, c_actions[action_pairs[i - 1][0]], c_actions[action_pairs[i - 1][1]]))
                    flog.write('frame {} to {}: before {}, after {}\n'.format(
                        i - 1, i, c_actions[action_pairs[i - 1][0]], c_actions[action_pairs[i - 1][1]]))

            images2video('./tmp', './output/{}_maze{}_frame{}_continue.mp4'.format(
                args.load.split('/')[-1], maze_id, frame_id))
            flog.write('./output/{}_maze{}_frame{}_continue.mp4\n'.format(
                args.load.split('/')[-1], maze_id, frame_id))

            # # blind at frame_id, let the agent rollout new sequence of actions from there
            #
            # position_pairs, orientation_pairs, state_pairs, action_pairs = perturb_trajectory(
            #     frame_id, args, env, map_design, positions, orientations, states, actions, model)
            #
            # count = len(position_pairs)
            # clear_folder('./tmp')
            # for i in range(count):
            #     visualize_comparison(map_design,
            #                          belief_pair=state_pairs[i],
            #                          position_pair=position_pairs[i],
            #                          orientation_pair=orientation_pairs[i],
            #                          fn='./tmp/frame_{0:03d}.png'.format(i), idx=i)
            #     if i > 0:
            #         print('frame {} to {}: before {}, after {}'.format(
            #             i - 1, i, c_actions[action_pairs[i - 1][0]], c_actions[action_pairs[i - 1][1]]))
            #         flog.write('frame {} to {}: before {}, after {}\n'.format(
            #             i - 1, i, c_actions[action_pairs[i - 1][0]], c_actions[action_pairs[i - 1][1]]))
            #
            # images2video('./tmp', './output/{}_maze{}_frame{}_rollout.mp4'.format(
            #     args.load.split('/')[-1], maze_id, frame_id))
            # flog.write('./output/{}_maze{}_frame{}_rollout.mp4\n'.format(
            #     args.load.split('/')[-1], maze_id, frame_id))

            # # perturb at frame_id, but follow the same sequence of actions
            #
            # position_pairs, orientation_pairs, state_pairs, action_pairs = perturb_trajectory_observation(
            #     frame_id, args, env, map_design, positions, orientations, states, actions)
            #
            # count = len(position_pairs)
            # clear_folder('./tmp')
            # for i in range(count):
            #     visualize_comparison(map_design,
            #                          belief_pair=state_pairs[i],
            #                          position_pair=position_pairs[i],
            #                          orientation_pair=orientation_pairs[i],
            #                          fn='./tmp/frame_{0:03d}.png'.format(i), idx=i)
            #     if i > 0:
            #         print('frame {} to {}: before {}, after {}'.format(
            #             i - 1, i, c_actions[action_pairs[i - 1][0]], c_actions[action_pairs[i - 1][1]]))
            #         flog.write('frame {} to {}: before {}, after {}\n'.format(
            #             i - 1, i, c_actions[action_pairs[i - 1][0]], c_actions[action_pairs[i - 1][1]]))
            #
            # images2video('./tmp', './output/{}_maze{}_frame{}_perturb_continue.mp4'.format(
            #     args.load.split('/')[-1], maze_id, frame_id))
            # flog.write('./output/{}_maze{}_frame{}_perturb_continue.mp4\n'.format(
            #     args.load.split('/')[-1], maze_id, frame_id))

            # # perturb at frame_id, let the agent rollout new sequence of actions from there
            #
            # position_pairs, orientation_pairs, state_pairs, action_pairs = perturb_trajectory_observation(
            #     frame_id, args, env, map_design, positions, orientations, states, actions, model)
            #
            # count = len(position_pairs)
            # clear_folder('./tmp')
            # for i in range(count):
            #     visualize_comparison(map_design,
            #                          belief_pair=state_pairs[i],
            #                          position_pair=position_pairs[i],
            #                          orientation_pair=orientation_pairs[i],
            #                          fn='./tmp/frame_{0:03d}.png'.format(i), idx=i)
            #     if i > 0:
            #         print('frame {} to {}: before {}, after {}'.format(
            #             i - 1, i, c_actions[action_pairs[i - 1][0]], c_actions[action_pairs[i - 1][1]]))
            #         flog.write('frame {} to {}: before {}, after {}\n'.format(
            #             i - 1, i, c_actions[action_pairs[i - 1][0]], c_actions[action_pairs[i - 1][1]]))
            #
            # images2video('./tmp', './output/{}_maze{}_frame{}_perturb_rollout.mp4'.format(
            #     args.load.split('/')[-1], maze_id, frame_id))
            # flog.write('./output/{}_maze{}_frame{}_perturb_rollout.mp4\n'.format(
            #     args.load.split('/')[-1], maze_id, frame_id))

            for action_before, action_after in action_pairs[frame_id+1:]:
                if action_before == action_after:
                    act_list.append(1)
                else:
                    act_list.append(0)

            position_after = np.unravel_index(np.argmax(state_pairs[-1][1][:4], axis=None),
                                              state_pairs[-1][1][:4].shape)
            position_truth = (orientation_pairs[-1][1], position_pairs[-1][1][0], position_pairs[-1][1][1])
            if position_after == position_truth:
                acc_list.append(1)
                print position_truth, position_after
                print("correct position")
                flog.write("correct position\n")
            else:
                acc_list.append(0)
                print position_truth, position_after
                print("incorrect position")
                flog.write("incorrect position\n")

    print "accuracy: {}, changed actions: {}".format(np.mean(acc_list), 1-np.mean(act_list))
    flog.write("accuracy: {}, changed actions: {}\n".format(np.mean(acc_list), 1-np.mean(act_list)))
