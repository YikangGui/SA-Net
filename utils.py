import numpy as np
import pandas as pd
from collections import Counter
from os import listdir
from os.path import isfile, join


def generate_csv(pwd='./data/RealSense/train', csv_file='train_transition.csv',save_pwd=None):
    df = pd.read_csv(f'{pwd}/{csv_file}', names=['images', 'onion', 'eef', 'yolo'])
    result = pd.DataFrame(columns=['image-1', 'image-2', 'image-3', 'image-4', 'image-5', 'onion', 'eef', 'yolo'])
    i = 0
    for j in range(len(df) - 5):
        name_first = df.loc[j]['images'].split('_')
        name_last = df.loc[j + 4]['images'].split('_')
        if name_first[0] == name_last[0]:
            result.loc[i] = [df.loc[j]['images'], df.loc[j + 1]['images'], df.loc[j + 2]['images'],
                             df.loc[j + 3]['images'], df.loc[j + 4]['images'], df.loc[j]['onion'],
                             df.loc[j]['eef'], df.loc[j]['yolo']]
            i += 1
    if save_pwd:
        result.to_csv(save_pwd, index=False)
    return result


def generate_new_eef(csv_file, save_pwd='./dist/data_new_mdp.csv'):
    df = pd.read_csv(csv_file)

    eef_new = df['eef'].replace('atHome', 'onConveyor')
    df['eef_new'] = eef_new

    action_new = df['action'].replace('Claim', 'Pick')
    df['action_new'] = action_new

    if save_pwd:
        df.to_csv(save_pwd, index=False)


def get_transition(csv_pwd, transition_interval=15, count_interval=5):
    assert transition_interval % count_interval == 0
    df = pd.read_csv(csv_pwd)
    transition = {}

    memory = {}

    for i in range(0, len(df), count_interval):
        state_action = tuple(df.loc[i][['onion', 'eef', 'yolo', 'action']])
        state = state_action[:-1]
        frame = df.loc[i]['image-1']
        episode = int(frame.split('_')[0])
        frame = int(frame.split('_')[1][5:-4])
        prev_frame = frame - transition_interval

        if prev_frame in memory:
            prev_episode = memory[prev_frame]['episode']
            if episode == prev_episode:
                prev_state_action = memory[prev_frame]['state_action']
                transition_next_state = transition.setdefault(prev_state_action, {})
                transition_next_state.setdefault(tuple(state), 0)
                transition[prev_state_action][state] += 1

        memory[frame] = {}
        memory[frame]['episode'] = episode
        memory[frame]['state_action'] = state_action
    return transition


def get_transition_new(csv_pwd, transition_interval=10, count_interval=2, npy_pwd='./dist/transition_new.npy', pd_pwd='./dist/transition_new.csv'):
    assert transition_interval % count_interval == 0
    df = pd.read_csv(csv_pwd)
    transition = {}

    memory = {}

    for i in range(0, len(df), count_interval):
        state_action = tuple(df.loc[i][['eef_new', 'yolo_new', 'action_new']])
        state = state_action[:-1]
        frame = df.loc[i]['image-1']
        episode = int(frame.split('_')[0])
        frame = int(frame.split('_')[1][5:-4])
        prev_frame = frame - transition_interval

        if prev_frame in memory:
            prev_episode = memory[prev_frame]['episode']
            if episode == prev_episode:
                prev_state_action = memory[prev_frame]['state_action']
                # if prev_state_action == ('inFront', 'unknown', 'Inspect') and state == ('inFront', 'good'):
                #     print(frame, episode)
                transition_next_state = transition.setdefault(prev_state_action, {})
                transition_next_state.setdefault(state, 0)
                transition[prev_state_action][state] += 1

        memory[frame] = {}
        memory[frame]['episode'] = episode
        memory[frame]['state_action'] = state_action

    if npy_pwd and pd_pwd:
        np.save(npy_pwd, transition)
        pd.DataFrame.from_dict(transition).to_csv(pd_pwd)
    return transition


def get_transition_prob(transition, np_pwd=None, pd_pwd=None):
    for state_action in transition:
        total_count = sum(transition[state_action].values())
        for state in transition[state_action]:
            transition[state_action][state] /= total_count
    if np_pwd and pd_pwd:
        np.save(np_pwd, transition)
        pd.DataFrame.from_dict(transition).to_csv(pd_pwd)


def get_trajectory(onion_dist, eef_dist, yolo_dist, state_action_csv_file, state_csv_file, transition_interval=10):
    state_action_records = pd.read_csv(state_action_csv_file)
    state_records = pd.read_csv(state_csv_file, names=['image', 'onion', 'eef'])
    onion_dist = np.load(onion_dist)
    eef_dist = np.load(eef_dist)
    yolo_dist = np.load(yolo_dist)

    trajectory = dict()

    onion_window = []
    eef_window = []
    yolo_window = []
    action_window = []
    interval_count = 0
    prev_episode = int(state_records.loc[0]['image'].split('_')[0])
    for i in range(len(state_action_records)):
        current_frame = state_action_records.loc[i]['image-1']
        state_id = state_records.index[state_records['image'] == current_frame].item()

        onion_frame_dist = onion_dist[state_id]
        eef_frame_dist = eef_dist[state_id]
        yolo_frame_dist = yolo_dist[state_id]
        action_frame = state_action_records.iloc[i]['action']

        episode = int(current_frame.split('_')[0])
        if episode == prev_episode:

            onion_window.append(onion_frame_dist)
            eef_window.append(eef_frame_dist)
            yolo_window.append(yolo_frame_dist)
            action_window.append(action_frame)

            interval_count += 1

            if interval_count % transition_interval == 0:
                assert len(onion_window) == transition_interval
                trajectory[len(trajectory)] = {
                    'onion': np.mean(onion_window, axis=0),
                    'eef': np.mean(eef_window, axis=0),
                    'yolo': np.mean(yolo_window, axis=0),
                    'action': Counter(action_window).most_common(1)[0][0]
                }

                interval_count = 0
                onion_window = []
                eef_window = []
                yolo_window = []
                action_window = []

        else:
            if interval_count > 0:
                trajectory[len(trajectory)] = {
                    'onion': np.mean(onion_window, axis=0),
                    'eef': np.mean(eef_window, axis=0),
                    'yolo': np.mean(yolo_window, axis=0),
                    'action': Counter(action_window).most_common(1)[0][0]
                }
            np.save(f'{prev_episode}_trajectory.npy', trajectory)
            pd.DataFrame.from_dict(trajectory).to_csv(f'{prev_episode}_trajectory.csv')

            trajectory = {}
            prev_episode = episode
            interval_count = 1
            onion_window = [onion_frame_dist]
            eef_window = [eef_frame_dist]
            yolo_window = [yolo_frame_dist]
            action_window = [action_frame]

    if interval_count > 0:
        trajectory[len(trajectory)] = {
            'onion': np.mean(onion_window, axis=0),
            'eef': np.mean(eef_window, axis=0),
            'yolo': np.mean(yolo_window, axis=0),
            'action': Counter(action_window).most_common(1)[0][0]
        }
    np.save(f'{episode}_trajectory.npy', trajectory)
    pd.DataFrame.from_dict(trajectory).to_csv(f'{episode}_trajectory.csv')


def get_trajectory_new(eef_dist, yolo_dist, state_action_csv_file, state_csv_file, transition_interval=10):
    state_action_records = pd.read_csv(state_action_csv_file)
    state_records = pd.read_csv(state_csv_file, names=['image', 'onion', 'eef'])
    # onion_dist = np.load(onion_dist)
    eef_dist = np.load(eef_dist)
    yolo_dist = np.load(yolo_dist)

    trajectory = dict()

    # onion_window = []
    eef_window = []
    yolo_window = []
    action_window = []
    interval_count = 0
    prev_episode = int(state_records.loc[0]['image'].split('_')[0])
    for i in range(len(state_action_records)):
        current_frame = state_action_records.loc[i]['image-1']
        frame_id = state_records.index[state_records['image'] == current_frame].item()

        # onion_frame_dist = onion_dist[state_id]

        # eef new
        eef_frame_dist = eef_dist[frame_id]
        eef_frame_dist = np.array([eef_frame_dist[0] + eef_frame_dist[1], eef_frame_dist[2], eef_frame_dist[3]])

        # yolo new
        yolo_frame_dist = yolo_dist[frame_id]
        if state_action_records.iloc[i]['yolo_new'] == 'none':
            yolo_frame_dist = np.array([1., 0, 0, 0])
        elif state_action_records.iloc[i]['yolo_new'] == 'unknown':
            yolo_frame_dist = np.array([0., 1, 0, 0])
        else:
            yolo_frame_dist = np.concatenate([[0, 0], yolo_frame_dist])

        # action new
        action_frame = state_action_records.iloc[i]['action']
        if action_frame == 'Claim':
            action_frame = 'Pick'

        episode = int(current_frame.split('_')[0])
        if episode == prev_episode:

            # onion_window.append(onion_frame_dist)
            eef_window.append(eef_frame_dist)
            yolo_window.append(yolo_frame_dist)
            action_window.append(action_frame)

            interval_count += 1

            if interval_count % transition_interval == 0:
                assert len(eef_window) == transition_interval
                trajectory[len(trajectory)] = {
                    # 'onion': np.mean(onion_window, axis=0),
                    'eef': np.mean(eef_window, axis=0),
                    'yolo': np.mean(yolo_window, axis=0),
                    'action': Counter(action_window).most_common(1)[0][0]
                }

                interval_count = 0
                # onion_window = []
                eef_window = []
                yolo_window = []
                action_window = []

        else:
            if interval_count > 0:
                trajectory[len(trajectory)] = {
                    # 'onion': np.mean(onion_window, axis=0),
                    'eef': np.mean(eef_window, axis=0),
                    'yolo': np.mean(yolo_window, axis=0),
                    'action': Counter(action_window).most_common(1)[0][0]
                }
            np.save(f'dist/{prev_episode}_trajectory.npy', trajectory)
            pd.DataFrame.from_dict(trajectory).to_csv(f'dist/{prev_episode}_trajectory.csv')

            trajectory = {}
            prev_episode = episode
            interval_count = 1
            # onion_window = [onion_frame_dist]
            eef_window = [eef_frame_dist]
            yolo_window = [yolo_frame_dist]
            action_window = [action_frame]

    if interval_count > 0:
        trajectory[len(trajectory)] = {
            # 'onion': np.mean(onion_window),
            'eef': np.mean(eef_window, axis=0),
            'yolo': np.mean(yolo_window, axis=0),
            'action': Counter(action_window).most_common(1)[0][0]
        }
    np.save(f'dist/{episode}_trajectory.npy', trajectory)
    pd.DataFrame.from_dict(trajectory).to_csv(f'dist/{episode}_trajectory.csv')


if __name__ == '__main__':
    # df = generate_csv(save_pwd='transition.csv')
    # df = generate_csv(pwd='./data/RealSense/val', csv_file='val_label.csv', save_pwd='val_transition.csv')
    # transition = get_transition('./transition.csv')
    # get_transition_prob(transition, './transition.npy')
    # get_trajectory(onion_dist='dist/onion_dist.npy', eef_dist='dist/eef_dist.npy', yolo_dist='dist/yolo_dist.npy', state_action_csv_file='val_transition.csv', state_csv_file='data/RealSense/val/val_label.csv', transition_interval=15)

    # new mdp
    generate_new_eef('data.csv', './dist/data_new_mdp.csv')
    transition = get_transition_new('./dist/data_new_mdp.csv')
    get_transition_prob(transition, np_pwd='dist/transition_new_dist.npy', pd_pwd='dist/transition_new_dist.csv')
    get_trajectory_new(eef_dist='dist/eef_dist.npy', yolo_dist='dist/yolo_dist.npy', state_action_csv_file='data/RealSense/val/val_data_new.csv', state_csv_file='data/RealSense/val/val_label.csv', transition_interval=10)
