# README

There are 4 distributions in total, which are actions, eef, grasp and yolo.

There are 15 trajectories in total. First 7 trajectories are unblemished onions, and the rest 8 trajectories are blemished onion.

Calculate the distribution over batch of 10 frames, which is about 0.333s.

## eef

[onConveyor, Inspection, AtBin]

## grasp

[no, yes]

## yolo

[unknown, blemished, unblemished]

## actions

[pick, inspect, place_on_conveyor, place_at_bin]

# Usage

To open the file,

```
import numpy as np

eef_iros = np.load('eef_iros.npy', allow_pickle=True)
grasp_iros = np.load('grasp_iros.npy', allow_pickle=True)
yolo_iros = np.load('yolo_iros.npy', allow_pickle=True)
action_iros = np.load('action_iros.npy', allow_pickle=True)
```


