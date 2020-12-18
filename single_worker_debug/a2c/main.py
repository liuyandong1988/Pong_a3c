from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train
import gym


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.001,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=True,
                    help='use an optimizer without shared momentum.')


if __name__ == '__main__':
    # 控制线程并发数, 目的不是在numpy进程中使用OMP线程, open multi-processing
    os.environ['OMP_NUM_THREADS'] = '1'
    # 使用CUDA_VISIBLE_DEVICES=0,1,2,3 python xxx.py来设置该程序可见的gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()
    # print(args.seed)
    # torch.manual_seed(args.seed)
    # - atari env - #
    env = create_atari_env(args.env_name)
    # --- global network --- #
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    print(shared_model)
    # -Asynchronous multiprocess training- #
    # https://zhuanlan.zhihu.com/p/78349516
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    # --- multi process --- #
    # https://www.cnblogs.com/pythoncainiao/p/10264139.html
    #
    processes = []
    # multiprocess share value
    counter = mp.Value('i', 0)
    # multiprocess lock
    lock = mp.Lock()

    #  --- test --- #
    # 测试线程完成对worker训练效果的检测
    # rank: 线程数目,
    # args：参数
    # shared_model： pytorch 多进程,
    # counter： multi-process share value
    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    # --- training --- #

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        # mp.set_start_method("spawn")
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
