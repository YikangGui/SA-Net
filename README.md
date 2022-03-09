# SA-Net

## Distribution
There are total 847 distributions extracted from 2 trajectories. 

1. The first 436 distributions are from trajectory 1. 
2. The rest 411 distributions are from trajectory 2.

## Accuracy
The accuracy of SA-Net is the following:

val onion acc: 91.61% | val eef acc: 95.15% | val onion loss: 1.369 | val eef loss: 2.926

The accuracy of YOLO is 100%.

## MDP
Transition:
```
**state(onion_location, eef_location, onion_yolo) -> action(eef) -> next state(onion_location, eef_location, onion_yolo)**

onConveyor, onConveyor, unblemished -> Pick -> atHome, atHome, unblemished

atHome, atHome, unblemished -> Pick -> inFront, inFront, unblemished

inFront, inFront, unblemished -> Inspect -> atHome, atHome, unblemished

atHome, atHome, unblemished -> PlaceOnConveyor -> onConveyor, onConveyor, unblemished

onConveyor, onConveyor, unblemished -> Claim -> onConveyor, atHome, unblemished 
                                             -> onConveyor, atHome, blemished   (two possible outcomes)

onConveyor, atHome, unblemished -> Claim -> onConveyor, onConveyor, unblemished

onConveyor, atHome, blemished -> Claim -> onConveyor, onConveyor, blemished

onConveyor, onConveyor, blemished, -> Pick -> atHome, atHome, blemished

atHome, atHome, blemished -> Pick -> inFront, inFront, blemished

inFront, inFront, blemished -> Inspect -> atHome, atHome, blemished

atHome, atHome, blemished -> PlaceOnBin -> atBin, atBin, blemished

atBin, atBin, blemished -> Claim -> onConveyor, atBin, blemished
                                 -> onConveyor, atBin, unblemished (two possible outcomes)
                                 
onConveyor, atBin, blemished -> Claim -> onConveyor, atHome, blemished

onConveyor, atBin, unblemished -> Claim -> onConveyor, atHome, unblemished

onConveyor, atHome, blemished -> Claim -> onConveyor, onConveyor, blemished

onConveyor, atHome, unblemished -> Claim, -> onConveyor, onConveyor, unblemished
```
State:
``` 
Onion location: 
    0. atHome
    1. onConveyor
    2. inFront 
    3. atBin

End effctor location:
    0. atHome
    1. onConveyor
    2. inFront 
    3. atBin

Onion YOLO stats:
    0. blemished
    1. unblemished
```
Action:
```
End effector action:
    0. Claim
    1. Pick
    2. Inspect
    3. PlaceOnConveyor
    4. PlaceOnBin
```
