import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# def normalized_columns_initializer(weights, std=1.0):
#     out = torch.randn(weights.size())
#     out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
#     return out
#
#
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         weight_shape = list(m.weight.data.size())
#         fan_in = np.prod(weight_shape[1:4])
#         fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
#         w_bound = np.sqrt(6. / (fan_in + fan_out))
#         m.weight.data.uniform_(-w_bound, w_bound)
#         m.bias.data.fill_(0)
#     elif classname.find('Linear') != -1:
#         weight_shape = list(m.weight.data.size())
#         fan_in = weight_shape[1]
#         fan_out = weight_shape[0]
#         w_bound = np.sqrt(6. / (fan_in + fan_out))
#         m.weight.data.uniform_(-w_bound, w_bound)
#         m.bias.data.fill_(0)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space):

        super(ActorCritic, self).__init__()

        num_outputs = action_space.n
        self.conv1_1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)

        self.conv1_2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)

        self.lstm = nn.LSTM(input_size=16 * 11 * 11, hidden_size=32)

        self.critic_linear = nn.Linear(32, 1)
        self.actor_linear = nn.Linear(32, num_outputs)

        # self.apply(weights_init)
        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        # self.train()

    def forward(self, inputs, h0, c0):

        f1, f2, f3, f4 = inputs[0], inputs[1], inputs[2], inputs[3]

        # data dimension switch
        # cnn input: batch*channel*height*width = 1*1*42*42
        f1 = f1.view(1, 1, 42, 42)
        f2 = f2.view(1, 1, 42, 42)
        f3 = f3.view(1, 1, 42, 42)
        f4 = f4.view(1, 1, 42, 42)

        # 1*1*42*42 -> 1*32*21*21
        x1 = F.relu(self.conv1_1(f1))
        x2 = F.relu(self.conv2_1(f2))
        x3 = F.relu(self.conv3_1(f3))
        x4 = F.relu(self.conv4_1(f4))

        # 1*32*21*21 -> 1*16*11*11
        x1 = F.relu(self.conv1_2(x1))
        x2 = F.relu(self.conv2_2(x2))
        x3 = F.relu(self.conv3_2(x3))
        x4 = F.relu(self.conv4_2(x4))

        x1 = x1.view(-1, 16 * 11 * 11)
        x2 = x2.view(-1, 16 * 11 * 11)
        x3 = x3.view(-1, 16 * 11 * 11)
        x4 = x4.view(-1, 16 * 11 * 11)
        # seq, batch, input_size: 4*1*1936
        lstm_inputs = torch.cat((x1, x2, x3, x4)).view(4, 1, -1)
        # inputs:
        # output: seq_len, batch, num_directions * hidden_size 4, 8, 1
        # hn: num_layers * num_directions, batch, hidden_size 1, 8, 1
        output, (hn, cn) = self.lstm(lstm_inputs, (h0, c0))

        return self.critic_linear(hn), self.actor_linear(hn)
