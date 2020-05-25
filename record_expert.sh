#!/bin/bash

for ((i=1; i<10; i++));
do
  python -m stable_baselines.gail.dataset.record_expert \
  logs/$1NoFrameskip-v4/baseline/rl_model_${i}000000_steps.zip \
  --note baseline/${i}e6_steps \
  --env $1NoFrameskip-v4
done;
