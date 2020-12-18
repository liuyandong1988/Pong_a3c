import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_atari_env
from model import ActorCritic

import os
import time
from collections import deque

frames_memory = deque(maxlen=4)

def ensure_shared_grads(model, shared_model):
    """
    Upload the local gradient to global network
    """
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    """
    worker train: 分布式workers进入自己的local network，与pong环境交互训练 global AC 网络。
    将更新后的梯度上传到 global Network， 从global Network上下载新的参数。
    """

    print('Subprocess id ：', os.getpid(), 'Parent process id ：', os.getppid())

    # each worker has own interacted env
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    # local AC network
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        # update the gradient in the global network
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    model.train()

    # --- begin to training loop --- #
    state = env.reset()
    state = torch.from_numpy(state)
    while len(frames_memory) != 4:
        frames_memory.append(state)
    done = True
    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            # for lstm input
            # h0: num_layer, batch, hidden_size: 1, 1, 32
            cx = torch.zeros(1, 1, 32)
            hx = torch.zeros(1, 1, 32)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []
        episode_step = 0

        # --- After Num-step update the model (Monte Carlo)
        # one episode game: 20 steps
        for step in range(args.num_steps):

            start_time = time.time()
            episode_length += 1
            # --- choose the action by AC network --- #
            # state: collection.deque
            value, logit = model(frames_memory, hx, cx)
            value = value.squeeze(dim=0)
            logit = logit.squeeze(dim=0)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            # policy gradient
            entropies.append(entropy)
            # action
            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            state, reward, done, _ = env.step(action.numpy())
            # env.render()
            # episode is done: game is over / max train step / for loop num step
            done = done or episode_length >= args.max_episode_length
            # win:1, lose:-1, continue:0
            reward = max(min(reward, 1), -1)
            # total interact counts
            with lock:
                counter.value += 1
            # record the info for update the local AC model
            state = torch.from_numpy(state)
            frames_memory.append(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_step = step + 1
            if done:
                state = env.reset()
                state = torch.from_numpy(state)
                frames_memory.clear()
                while len(frames_memory) != 4:
                    frames_memory.append(state)
                break

        # # step/episode step, MC rewards, MC time
        # print('MC-step/episode-step: {}/{} Reward:{} Time:{:.3f}'.
        #       format(episode_step, args.num_steps, sum(rewards), time.time() - start_time))
        # --- Update the AC model params by return reward --- #
        R = torch.zeros(1, 1)
        if not done:
            value, _, = model(frames_memory, hx, cx)
            R = value.detach()
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        # calculate the loss function: critic loss and actor loss
        # gamma: 0.99
        # gae_lambda: 1.00
        # entropy_coef: 0.01     regularization
        # max_grad_norm: 50
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]
        loss_func = policy_loss + args.value_loss_coef * value_loss
        optimizer.zero_grad()
        loss_func.backward()
        # clip gradient: The gradient disappears or explodes
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # upload the new gradient
        ensure_shared_grads(model, shared_model)
        optimizer.step()
