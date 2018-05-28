# -*- coding: utf-8 -*-
import numpy as np

__author__ = 'l'
__date__ = '2018/5/28'

"""
情景模拟
"""

from build_model.model import DQN
from build_model.status import Status
from build_model.virtual_situation import VirtualSituation
from build_model.action import Action


def get_training_data(model, vs):
    """
    获得一批次训练数据
    :param model:
    :param vs:
    :return:
    """
    observation = vs.get_observation()
    memory = model.memory
    for i in range(200):
        # 基于一些策略选择下一步动作
        action = model.choose_action(observation)
        # 模拟进行每一步操作
        observation_, reward, done = vs.step(action)
        transition = model.store_transition(observation, action, reward, observation_)
        memory[i] = transition
        # 新的观测值赋值
        observation = observation_
        if done:
            vs = VirtualSituation()
    return memory


def show_step(model, vs):
    """
    模型训练
    :param model: DQN模型
    :param vs:虚拟情景
    :return:
    """
    training_data = get_training_data(model, vs)
    model.memory = training_data
    np.savetxt('./data.csv', training_data, delimiter=',')


def main():
    """
    主要方法
    :return:
    """
    vs = VirtualSituation()  # 模拟情景
    status = Status(vs)
    d_model = DQN(Action.actions_len, status.get_feature_num(), learning_rate=10e-3, reward_decay=0.9,
                  e_greedy=1, replace_target_iter=200, memory_size=200
                  # output_graph = True
                  )
    show_step(d_model, vs)


if __name__ == '__main__':
    main()
