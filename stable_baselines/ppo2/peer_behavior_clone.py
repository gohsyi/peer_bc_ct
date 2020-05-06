from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from stable_baselines.common.cmd_util import atari_arg_parser

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset

parser = atari_arg_parser()
parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm', 'mlp'],
                default='cnn', help='Policy architecture')
parser.add_argument('--peer', type=float, default=0.,
                help='Coefficient of the peer term. (default: 0)')
parser.add_argument('--log', type=str, default='logs',
                help='Log note')
parser.add_argument('--note', type=str, default='test',
                help='Log path')
parser.add_argument('--expert', type=str, default=None,
                help='Expert model path.')
parser.add_argument('--num-epochs', type=int, default=1000)
args = parser.parse_args()

dataset = ExpertDataset(
    expert_path=args.expert, traj_limitation=1, batch_size=128)

model = PPO2(args.policy, args.env, verbose=1)
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
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()
env.close()
