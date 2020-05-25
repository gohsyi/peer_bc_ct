#!/bin/bash

for ((i=9; i>0; i--));
do
  python -m stable_baselines.ppo2.peer_behavior_clone \
  logs/$1NoFrameskip-v4/baseline/${i}e6_steps/expert.npz \
  --env $1NoFrameskip-v4 \
  --note ${i}e6steps_peer$2/ \
  --peer $2
done;
