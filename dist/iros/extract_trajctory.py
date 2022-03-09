import numpy as np
import pandas as pd


interval = 10

trajectory_start = [0, 211, 392, 563, 744, 925, 1115, 1265, 1466, 1657, 1843, 2032, 2224, 2409, 2613]

action_dict = {
    'p': 0,
    'i': 1,
    'c': 2,
    'b': 3
}


eef = np.load('eef_dist.npy')
grasp = np.load('grasp_dist.npy')
yolo = np.load('yolo_dist.npy')

test_csv = pd.read_csv('test_w_action.csv', names=['image', 'eef', 'grasp', 'action'])
action = test_csv['action']
action_init = np.zeros((len(action), 4))
for i in range(len(action)):
    action_init[i][action_dict[action[i]]] = 1
action = action_init


assert len(eef) == len(grasp) == len(yolo) == len(action)

eef_total = []
grasp_total = []
yolo_total = []
action_total = []

eef_traj = []
grasp_traj = []
yolo_traj = []
action_traj = []

eef_temp = []
grasp_temp = []
yolo_temp = []
action_temp = []

for i in range(len(eef)):
    if i in trajectory_start and i > 0:
        eef_dist = np.sum(eef_temp, axis=0) / len(eef_temp)
        grasp_dist = np.sum(grasp_temp, axis=0) / len(grasp_temp)
        yolo_dist = np.sum(yolo_temp, axis=0) / len(yolo_temp)
        action_dist = np.sum(action_temp, axis=0) / len(action_temp)

        assert abs(sum(eef_dist)) - 1 < 1e-6
        assert abs(sum(grasp_dist)) - 1 < 1e-6
        assert abs(sum(yolo_dist)) - 1 < 1e-6
        assert abs(sum(action_dist)) - 1 < 1e-6

        eef_traj.append(eef_dist.copy())
        grasp_traj.append(grasp_dist.copy())
        yolo_traj.append(yolo_dist.copy())
        action_traj.append(action_dist.copy())

        eef_total.append(np.array(eef_traj).copy())
        grasp_total.append(np.array(grasp_traj).copy())
        yolo_total.append(np.array(yolo_traj).copy())
        action_total.append(np.array(action_traj).copy())

        eef_traj = []
        grasp_traj = []
        yolo_traj = []
        action_traj = []

        eef_temp = []
        grasp_temp = []
        yolo_temp = []
        action_temp = []

    elif len(eef_temp) % interval == 0 and len(eef_temp) > 0:
        eef_dist = np.sum(eef_temp, axis=0) / len(eef_temp)
        grasp_dist = np.sum(grasp_temp, axis=0) / len(grasp_temp)
        yolo_dist = np.sum(yolo_temp, axis=0) / len(yolo_temp)
        action_dist = np.sum(action_temp, axis=0) / len(action_temp)

        assert abs(sum(eef_dist)) - 1 < 1e-6
        assert abs(sum(grasp_dist)) - 1 < 1e-6
        assert abs(sum(yolo_dist)) - 1 < 1e-6
        assert abs(sum(action_dist)) - 1 < 1e-6

        eef_traj.append(eef_dist.copy())
        grasp_traj.append(grasp_dist.copy())
        yolo_traj.append(yolo_dist.copy())
        action_traj.append(action_dist.copy())

        eef_temp = []
        grasp_temp = []
        yolo_temp = []
        action_temp = []

    eef_temp.append(eef[i].copy())
    grasp_temp.append(grasp[i].copy())
    yolo_temp.append(yolo[i].copy())
    action_temp.append(action[i].copy())

eef_dist = np.sum(eef_temp, axis=0) / len(eef_temp)
grasp_dist = np.sum(grasp_temp, axis=0) / len(grasp_temp)
yolo_dist = np.sum(yolo_temp, axis=0) / len(yolo_temp)
action_dist = np.sum(action_temp, axis=0) / len(action_temp)

eef_traj.append(eef_dist.copy())
grasp_traj.append(grasp_dist.copy())
yolo_traj.append(yolo_dist.copy())
action_traj.append(action_dist.copy())

eef_total.append(np.array(eef_traj).copy())
grasp_total.append(np.array(grasp_traj).copy())
yolo_total.append(np.array(yolo_traj).copy())
action_total.append(np.array(action_traj).copy())

np.save('eef_iros.npy', eef_total)
np.save('grasp_iros.npy', grasp_total)
np.save('yolo_iros.npy', yolo_total)
np.save('action_iros.npy', action_total)
