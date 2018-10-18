import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading
import argparse
import global_utils as gu
import globals as g
from PPONet import PPONet
from Worker import Worker

'''
Base DPPO code is from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/discrete_DPPO.py
'''



parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='log directory', default='./log/')
parser.add_argument('--savedir', help='save directory', default='./save_model/')
parser.add_argument('--gamma', default=0.95)
parser.add_argument('--iteration', default=int(1e7))
parser.add_argument('--n_worker', default=2)
parser.add_argument('--actor_lr', default=1e-4)
parser.add_argument('--critic_lr', default=1e-4)
parser.add_argument('--epsilon_surrogate', default=0.2)
parser.add_argument('--update_step', default=15)
parser.add_argument('--MAX_EP', default = 3000)
parser.add_argument('--EP_LEN', default = 500)
parser.add_argument('--BATCH_SIZE', default = 65)
parser.add_argument('--GAME', default='CartPole-v0')
args = parser.parse_args()


env = gym.make(args.GAME)
state_dim = env.observation_space.shape
action_dim = env.action_space.n


if __name__=="__main__":
    print("Global Variables Initialize")
    g.initialize()

    GLOBAL_PPO = PPONet(state_dim, action_dim, args)

    workers = [Worker(wid=i, GLOBAL_PPO=GLOBAL_PPO, args=args) for i in range(args.n_worker)]

    COORD = tf.train.Coordinator()

    threads = []

    for worker in workers:  # worker threads
        worker.set_COORD(COORD)
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)

    # add a PPO updating thread
    GLOBAL_PPO.set_COORD(COORD)
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    plt.plot(np.arange(len(gu.GLOBAL_RUNNING_R_get())), gu.GLOBAL_RUNNING_R_get())
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.show()
    env = gym.make('CartPole-v0')
    while True:
        s = env.reset()
        for t in range(1000):
            env.render()
            s, r, done, info = env.step(GLOBAL_PPO.action_estimate(s))
            if done:
                break