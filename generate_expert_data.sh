#!/bin/bash

for ((i=1; i<10; i++));
do
  python -m stable_baselines.gail.dataset.record_expert logs/PongNoFrameskip-v4/baseline/rl_model_${i}000000_steps.zip --logdir logs/PongNoFrameskip-v4/baseline/${i}000000_steps/
done;
