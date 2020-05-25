#!/bin/bash

python -m stable_baselines.ppo2.copier --env $1NoFrameskip-v4 \
--note individual --individual &

python -m stable_baselines.ppo2.copier --env $1NoFrameskip-v4 \
--note copier &

python -m stable_baselines.ppo2.copier --env $1NoFrameskip-v4 \
--note peer0dot5_start2000_end10000 --peer 0.5 \
--start-episode 2000 --end-episode 8000

python -m stable_baselines.ppo2.copier --env $1NoFrameskip-v4 \
--note peer0dot2_start2000_end10000 --peer 0.2 \
--start-episode 2000 --end-episode 8000
