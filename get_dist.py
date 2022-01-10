import numpy as np


def get_dist(prob, l):
    length = [l[i] - l[i-1] for i in range(1, len(l))]
    assert len(prob) == len(length)
    result = []
    for i in range(len(prob)):
        tmp = np.repeat([prob[i]], length[i], axis=0)
        result.extend(tmp)
    return result


if __name__ == '__main__':
    episode_19_index = [255, 410, 487, 615, 691]
    episode_19_prob = [[1.72424e-02, 9.82910e-01], [9.97070e-01, 3.12424e-03],
                       [6.31332e-03, 9.93164e-01], [9.96094e-01, 4.16565e-03]]
    episode_20_index = [160, 284, 418, 490, 571]
    episode_20_prob = [[9.86816e-01, 1.42822e-02], [9.96582e-01, 3.24821e-03],
                       [7.66754e-03, 9.92188e-01], [8.71277e-03, 9.91699e-01]]
    result = []
    result.extend(get_dist(episode_19_prob, episode_19_index))
    result.extend(get_dist(episode_20_prob, episode_20_index))

    result = np.stack(result)
    np.save('./dist/yolo.npy', result)
