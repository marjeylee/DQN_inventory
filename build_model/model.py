"""
决策模型
"""

import numpy as np
import tensorflow as tf

from utility import combine_file_path


def leaky_relu(x, leak=0.2, name="LeakyRelu"):
    """
    leak relu
    :param x:
    :param leak:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


class DQN:
    """
    构建DQN模型
    """

    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None, output_graph=False,
                 ):
        """

        :param n_actions: 总共有多少种行为
        :param n_features:  一个status有多少种feature
        :param learning_rate: 学习率
        :param reward_decay: 奖励衰减率
        :param e_greedy: 丢弃率
        :param replace_target_iter: 迭代多少次进行替换
        :param memory_size: 保存多少条数据
        :param batch_size: 每批次训练条目数
        :param e_greedy_increment: 丢弃增量
        :param output_graph: 是否输出图片
        """
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0  # 学习次数
        # 初始化记忆库 [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0
        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')  # 获取所有变量
        e_params = tf.get_collection('eval_net_params')  # 获取所有变量
        # 逐个赋值
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()
        if output_graph:
            """
            是否输出日志图
            """
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, combine_file_path("build_model/Model/model"))
        self.cost_his = []
        self.cost = None

    def save_model(self):
        """
        保存模型
        :return:
        """
        saver = tf.train.Saver()
        saver.save(self.sess, combine_file_path("build_model/Model/model"))

    def _build_net(self):
        """
        建立神经网络
        :return:
        """
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 输入
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 计算损失函数
        with tf.variable_scope('eval_net'):
            # 存储变量
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 1000  # 第一层神经元个数
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            # 用于训练的神经网络。最后赋值给目标神经网络
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            with tf.variable_scope('l1_1'):
                w1_1 = tf.get_variable('w1_1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b1_1 = tf.get_variable('b1_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1_1 = tf.nn.relu(tf.matmul(l1, w1_1) + b1_1)
            # 第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1_1, w2) + b2

        with tf.variable_scope('loss'):
            """计算损失函数"""
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            """用于训练"""
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        """target net"""
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            # 被赋值的神经网络
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            with tf.variable_scope('l1_1'):
                w1_1 = tf.get_variable('w1_1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                b1_1 = tf.get_variable('b1_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1_1 = tf.nn.relu(tf.matmul(l1, w1_1) + b1_1)
            # 第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1_1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0  # 存储库数量
        s = s.reshape((19,))
        s_ = s_.reshape((19,))
        tmp = np.array([a, r]).reshape((2,))
        transition = np.hstack((s, tmp, s_))  # axis 第二层合成
        return transition
        # 把旧的记忆库替换成新的
        # index = self.memory_counter % self.memory_size
        # self.memory[index, :] = transition
        # self.memory_counter += 1

    def choose_action(self, observation):
        # observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # 前向传播，获取每个action的q value
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            # print(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 初始化进行替换
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        # 随机从记忆库选择batch数据进行训练
        self.memory_counter = len(self.memory[:, ])
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # 训练模型
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
