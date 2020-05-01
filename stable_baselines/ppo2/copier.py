from stable_baselines import PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, \
    CnnLnLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import VecFrameStack


def train(env_id, num_timesteps, seed, policy, n_envs=8, nminibatches=4,
          n_steps=128):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param n_envs: (int) Number of parallel environments
    :param nminibatches: (int) Number of training minibatches per update.
        For recurrent policies, the number of environments run in parallel
        should be a multiple of nminibatches.
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env
        is number of environment copies running in parallel)
    """

    env = VecFrameStack(make_atari_env(env_id, n_envs, seed), 4)
    policy = {
        'cnn': CnnPolicy,
        'lstm': CnnLstmPolicy,
        'lnlstm': CnnLnLstmPolicy,
        'mlp': MlpPolicy}[policy]
    models = {
        "A": PPO2(
            policy=policy, policy_kwargs={'view': 'even'}, n_steps=n_steps,
            env=VecFrameStack(make_atari_env(env_id, n_envs, seed), 4),
            nminibatches=nminibatches, lam=0.95, gamma=0.99, noptepochs=4,
            ent_coef=.01, learning_rate=2.5e-4,
            cliprange=lambda f: f * 0.1, verbose=1),
        "B": PPO2(
            policy=policy, policy_kwargs={'view': 'odd'}, n_steps=n_steps,
            env=VecFrameStack(make_atari_env(env_id, n_envs, seed), 4),
            nminibatches=nminibatches, lam=0.95, gamma=0.99, noptepochs=4,
            ent_coef=.01, learning_rate=2.5e-4,
            cliprange=lambda f: f * 0.1, verbose=1)}

    n_batch = n_envs * n_steps
    n_updates = num_timesteps // n_batch
    for t in range(n_updates):
        for view in "A", "B":
            models[view].learn(n_batch)

    for view in "A", "B":
        models[view].env.close()
        del models[view]  # free memory


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm', 'mlp'],
                        default='cnn', help='Policy architecture')
    parser.add_argument('--log', type=str, default='test',
                        help='Log ID')
    args = parser.parse_args()
    logger.configure(args.log)
    train(
        args.env,
        num_timesteps=args.num_timesteps,
        seed=args.seed,
        policy=args.policy)


if __name__ == '__main__':
    main()
