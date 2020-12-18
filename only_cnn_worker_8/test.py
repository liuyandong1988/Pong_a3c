import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from envs import create_atari_env
from model import ActorCritic
import os


def test(rank, args, shared_model, counter, save_mark=True):

    print(' Test subprocess id ：', os.getpid(), 'parent process id ：', os.getppid())

    # --- create test env： pong --- #
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    # local AC network
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    # pytorch 网络inference 模式
    # pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    # 不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大；在模型测试阶段使用
    model.eval()

    # --- test game loop --- #
    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    start_time = time.time()

    # a quick hack to prevent the agent from sticking
    actions = deque(maxlen=100)
    episode_length = 0
    # draw the reward curve on the tensorboard
    tb = SummaryWriter()
    test_cnts = 1
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            # update the model params from global network
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        # without calculating the gradient
        with torch.no_grad():
            value, logit = model(state.unsqueeze(0))
        # get the action by the actor network
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        # episode reward
        reward_sum += reward

        # a quick hack to prevent the agent from sticking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                counter.value,
                counter.value / (time.time() - start_time),
                reward_sum,
                episode_length))
            episode_length = 0
            actions.clear()
            state = env.reset()
            tb.add_scalar('Pong-reward', reward_sum, test_cnts)
            test_cnts += 1
            # --- save the model --- #
            if reward_sum >= 18 and save_mark:
                save_folder = './model'
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
                model_name = 'agent_{}'.format(test_cnts)
                torch.save(model.state_dict(), os.path.join(save_folder, model_name))
            reward_sum = 0
            # test once in each minutes
            time.sleep(60)

        state = torch.from_numpy(state)
