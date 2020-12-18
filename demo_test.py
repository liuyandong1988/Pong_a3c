import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from multi_worker.envs import create_atari_env
from model import ActorCritic
import os

def demo(model_path):

    # create env
    env = create_atari_env('PongDeterministic-v4')
    # local AC network
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    # --- test game loop --- #
    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    start_time = time.time()

    # a quick hack to prevent the agent from sticking
    # actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            # load the training model params
            model.load_state_dict(torch.load(model_path))
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        # without calculating the gradient
        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        # get the action by the actor network
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        state, reward, done, _ = env.step(action[0, 0])
        env.render()
        time.sleep(0.02)
        # episode reward
        reward_sum += reward
        # # a quick hack to prevent the agent from sticking
        # actions.append(action[0, 0])
        # if actions.count(actions[0]) == actions.maxlen:
        #     done = True
        if done:
            state = env.reset()
            # test once in each minutes
            time.sleep(3)
        state = torch.from_numpy(state)


if __name__ == '__main__':
    model_path = './model/agent_400000'
    demo(model_path)
