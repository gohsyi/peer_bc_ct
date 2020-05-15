#!/bin/bash

for ((i=9; i>0; i--));
do
  python -m stable_baselines.ppo2.peer_behavior_clone \
  logs/PongNoFrameskip-v4/baseline/${i}000000_steps/expert.npz \
  --logdir logs/PongNoFrameskip-v4/${i}000000_steps_peer0/ \
  --num-epochs 50 \
  --val-interval 100 \
  --val-episodes 10
done;
