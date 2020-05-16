import os
import copy
import numpy as np
import tensorflow as tf

from stable_baselines import PPO2, logger
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.cmd_util import (
    make_atari_env, atari_arg_parser)
from stable_baselines.common.policies import (
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy)


class View():
    def __init__(self, model, peer=0., learning_rate=2.5e-4, epsilon=1e-5):
        self.model = model
        self.peer = peer
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        with self.model.graph.as_default():
            with tf.variable_scope('copier'):
                self.obs_ph, self.actions_ph, self.actions_logits_ph = \
                    self.model._get_pretrain_placeholders()
                actions_ph = tf.expand_dims(self.actions_ph, axis=1)
                one_hot_actions = tf.one_hot(actions_ph, self.model.action_space.n)
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.actions_logits_ph,
                    labels=tf.stop_gradient(one_hot_actions))
                self.loss = tf.reduce_mean(self.loss)
                # calculate peer term
                self.peer_actions_ph = tf.placeholder(
                    actions_ph.dtype, actions_ph.shape, "peer_action_ph")
                peer_onehot_actions = tf.one_hot(
                    self.peer_actions_ph, self.model.action_space.n)
                # Use clipped softmax instead of the default
                # peer_term = tf.nn.softmax_cross_entropy_with_logits_v2(
                #     logits=self.actions_logits_ph,
                #     labels=tf.stop_gradient(peer_onehot_actions))
                softmax_actions_logits_ph = tf.nn.softmax(
                    self.actions_logits_ph, axis=1) + 1e-8
                peer_term = tf.reduce_mean(-tf.reduce_sum(
                    tf.stop_gradient(peer_onehot_actions) *
                    tf.log(softmax_actions_logits_ph), axis=-1))
                self.peer_term = self.peer * peer_term
                self.loss -= self.peer_term

            self.optim_op = self.model.trainer.minimize(
                self.loss, var_list=self.model.params)

    def learn(self, obses, actions):
        peer_actions = copy.deepcopy(actions)
        np.random.shuffle(peer_actions)
        feed_dict = {
            self.obs_ph: obses,
            self.actions_ph: actions[: None],
            self.peer_actions_ph: peer_actions[:, None],
            self.model.learning_rate_ph: self.learning_rate
        }
        train_loss, peer_loss, _ = self.model.sess.run(
            [self.loss, self.peer_term, self.optim_op], feed_dict)
        logger.logkv("copier loss", train_loss)
        logger.logkv("peer term", peer_loss)
        logger.dumpkvs()


def train(env_id, num_timesteps, seed, policy, n_envs=8, nminibatches=4,
          n_steps=128, peer=0., individual=False):
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
    :param n_steps: (int) The number of steps to run for each environment
        per update (i.e. batch size is n_steps * n_env where n_env is
        number of environment copies running in parallel)
    """

    policy = {
        'cnn': CnnPolicy,
        'lstm': CnnLstmPolicy,
        'lnlstm': CnnLnLstmPolicy,
        'mlp': MlpPolicy
    }[policy]

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

    views = {view: View(models[view], peer=peer) for view in ("A", "B")}

    n_batch = n_envs * n_steps
    n_updates = num_timesteps // n_batch
    for t in range(n_updates):
        for view in "A", "B":
            models[view].learn(n_batch)
        if not individual:
            for view, other_view in zip(("A", "B"), ("B", "A")):
                obses, _, _, actions, _, _, _, _, _ = models[other_view].rollout
                views[view].learn(obses, actions)

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
    parser.add_argument('--peer', type=float, default=0.,
                        help='Coefficient of the peer term. (default: 0)')
    parser.add_argument('--note', type=str, default='test',
                        help='Log path')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='If true, no co-training is applied.')
    args = parser.parse_args()
    logger.configure(os.path.join('logs', args.env, args.note))
    logger.info(args)
    train(
        args.env,
        num_timesteps=args.num_timesteps,
        seed=args.seed,
        policy=args.policy,
        peer=args.peer,
        individual=args.individual
    )


if __name__ == '__main__':
    main()
