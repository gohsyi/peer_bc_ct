from stable_baselines import PPO2, logger
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.cmd_util import atari_arg_parser, make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset

parser = atari_arg_parser()
parser.add_argument('--policy', type=str, default='CnnPolicy',
                    choices=['CnnPolicy', 'CnnLstmPolicy', 'CnnLnLstmPolicy',
                             'MlpPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy'],
                    help='Policy architecture')
parser.add_argument('--peer', type=float, default=0.,
                    help='Coefficient of the peer term. (default: 0)')
parser.add_argument('--logdir', type=str, default='logs/',
                    help='Log path')
parser.add_argument('--expert', type=str, default=None,
                    help='Expert model path.')
parser.add_argument('--num-epochs', type=int, default=1000)
args = parser.parse_args()

logger.configure(args.logdir)

dataset = ExpertDataset(
    expert_path=args.expert, traj_limitation=1, batch_size=128)

env = VecFrameStack(make_atari_env(args.env, 1, args.seed), 4)

model = PPO2(args.policy, env, verbose=1)

# Pretrain the PPO2 model
model.pretrain(dataset, n_epochs=args.num_epochs, peer=args.peer)

# As an option, you can train the RL agent
# model.learn(int(1e5))

# Test the pre-trained model
env = model.get_env()
obs = env.reset()

reward_sum = 0.0
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        logger.logkv('ep_rewards', reward_sum)
        logger.dumpkvs()
        reward_sum = 0.0
        obs = env.reset()
env.close()
