# Peer Behavior Cloning and Peer Co-Training

The implementation is based on [stable-baselines](https://github.com/Stable-Baselines-Team/stable-baselines). Thanks to the original authors!

## Installation

```
conda create -n peer-bc-ct python==3.6
conda activate peer-bc-ct
pip install -e . 
```

## Peer BC

To replicate the experiments, we first need to train an imperfect expert policy.

```
python -m stable_baselines.ppo2 
```

We need to generate expert dataset first using `stable_baselines/ppo2/record_expert.py`

For example, we want to generate dataset for `PongNoFrameskip-v4`:

```
python -m stable_baselines.ppo2.record_expert logs/PongNoFrameskip-v4/baseline/rl_model_2000000_steps.zip --note baseline/2e6_steps --env PongNoFrameskip-v4
```

The expert policy 

Then, we can run peer

## Peer CT

### Example

```
python -m stable_baselines.ppo2.copier --env Acrobot-v1 --policy mlp
```