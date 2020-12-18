import math

import torch
import torch.optim as optim


"""
对于每个网络模型参数都使用state['exp_avg']和state['exp_avg_sq']来保存 梯度 和 梯度的平方 的移动平均值。
第一次更新的时候没有state，即len(state) == 0，
所以两个数值都需要使用torch.zeros_like(p.data)来初始化为 [公式] ，之后每次都只需要从state中取出该值使用和更新即可。
state['step']用于保存本次更新是优化器第几轮迭代更新参数
"""

class SharedAdam(optim.Adam):
    """
    Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        # 网络模型参数和优化器的参数都保存在列表 self.param_groups 的元素中，
        # 该元素以字典形式存储和访问具体的网络模型参数和优化器的参数。
        # 所以，可以通过两层循环访问网络模型的每一个参数 p 。
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # ************************************************************#
                # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # change in pytorch 1.5.0
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # ************************************************************#


                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
