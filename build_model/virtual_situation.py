"""
建立虚拟模型
"""
from build_model.action import Action
from obj.Customer import Customer
from obj.Factory import Factory
from obj.vehicle import Vehicle
import numpy as np


class VirtualSituation:
    """
    建立虚拟场景
    """

    def __init__(self):
        """
        构建虚拟环境
        """
        self.current_time = 0  # 当前时间
        self.p1 = Factory(name='p1', location=[4, 1], identify=0)
        self.p2 = Factory(name='p2', location=[1, 7], identify=1)
        self.c1 = Customer(name='c1', location=[4, 3], capacity=500, inventory=356, consume=1, identify=2)
        self.c2 = Customer(name='c2', location=[7, 4], capacity=400, inventory=232, consume=2, identify=3)
        self.c3 = Customer(name='c3', location=[2, 6], capacity=500, inventory=444, consume=2, identify=4)
        self.c4 = Customer(name='c4', location=[8, 7], capacity=500, inventory=500, consume=1, identify=5)
        self.c5 = Customer(name='c5', location=[7, 8], capacity=550, inventory=500, consume=1, identify=6)
        # 车辆初始位置设定为在p1。
        self.vehicle = Vehicle(name='vehicle', location=[4, 1], capacity=500, inventory=500, unload_each_time=10,
                               location_name='p1')

    def spend_one_hour(self):
        """
        度过一小时时间，时间加1，增加所有消耗。
        :return:
        """
        self.current_time = self.current_time + 1
        if self.c1.inventory - self.c1.consume >= 0:
            self.c1.inventory = self.c1.inventory - self.c1.consume
        else:
            self.c1.inventory = 0
        if self.c2.inventory - self.c2.consume >= 0:
            self.c2.inventory = self.c2.inventory - self.c2.consume
        else:
            self.c2.inventory = 0
        if self.c3.inventory - self.c3.consume >= 0:
            self.c3.inventory = self.c3.inventory - self.c3.consume
        else:
            self.c3.inventory = 0
        if self.c4.inventory - self.c4.consume >= 0:
            self.c4.inventory = self.c4.inventory - self.c4.consume
        else:
            self.c4.inventory = 0
        if self.c5.inventory - self.c5.consume >= 0:
            self.c5.inventory = self.c5.inventory - self.c5.consume
        else:
            self.c5.inventory = 0

    def get_address_num(self, loc):
        """
        获取汽车所在位置对应的id
        :param loc:
        :return:
        """
        for i in [self.p1, self.p2, self.c1, self.c2, self.c3, self.c4, self.c5]:
            if loc[0] == i.location[0] and loc[1] == i.location[1]:
                return i.identify
        raise Exception('没找到id')

    def reset(self):
        """
        重置
        :return:
        """
        self.vehicle = Vehicle(name='vehicle', location=[4, 1], capacity=50, inventory=50, unload_each_time=10,
                               location_name='p1')

    @staticmethod
    def get_distance(a, b):
        """
        获取两对象之间的绝对距离
        :param a:
        :param b:
        :return:
        """
        return np.sum(np.abs(np.array(a.location) - np.array(b.location)))

    def step(self, action):
        """
        记录每个action的结果
        :param action:
        :return:
        """
        action = Action.actions[action]
        time_before_action = self.current_time
        action = action.replace('to_', '')
        self.go_to_new_location(action)  # 到新的目的地去，返回距离
        unload_volume = self.update_status()  # 更新车辆操作，返回卸货量
        observation_, reward = self.get_observation(), 0
        time_after_action = self.current_time
        cost_time = time_after_action - time_before_action
        if cost_time > 0:
            reward = float(unload_volume) / float(cost_time)
        if self.validate_is_stockout():
            reward = -1000
            return observation_, reward, True
        return observation_, reward, False

    def get_observation(self):
        """
        获取观测值
        :return:
        """
        v = self.vehicle
        obs = np.array(
            [0, 0, 0, 0, 0, 0, 0, v.capacity, v.inventory, self.c1.capacity, self.c1.inventory, self.c2.capacity,
             self.c2.inventory, self.c3.capacity,
             self.c3.inventory, self.c4.capacity, self.c4.inventory, self.c5.capacity, self.c5.inventory],
            dtype=np.float).reshape((1, 19))
        address_num = self.get_address_num(v.location)
        obs[0][address_num] = 1
        return obs

    def go_to_new_location(self, destination):
        """
        到达新的地点，更新
        :param destination:
        :return:distance
        """
        self.vehicle.location_name = destination
        destination = getattr(self, destination)
        distance = self.get_distance(self.vehicle, destination)  # 获取距离
        self.vehicle.location = destination.location  # 修改位置
        if distance > 0:
            for _ in range(int(distance)):
                self.spend_one_hour()
        return distance

    def update_status(self):
        """
        更新当前状态。
        :return:
        """
        current_location = getattr(self, self.vehicle.location_name)
        current_location_name = self.vehicle.location_name
        if current_location_name.find('p') >= 0:
            """在工厂，加满油，耗时1"""
            self.spend_one_hour()
            self.vehicle.inventory = self.vehicle.capacity
            return 0
        """到客户那"""
        # 加10单位油，需要时间1
        if self.vehicle.inventory >= self.vehicle.unload_each_time and (
                current_location.capacity >= current_location.inventory + self.vehicle.unload_each_time):
            """只有满足了以上情况，才加油"""
            self.vehicle.inventory = self.vehicle.inventory - self.vehicle.unload_each_time
            current_location.inventory = current_location.inventory + self.vehicle.unload_each_time
            self.spend_one_hour()
            return self.vehicle.unload_each_time
        return 0

    def validate_is_stockout(self):
        """
        判断进行完操作是否脱销 ，即各个customer的inventory小于2
        :return:
        """
        if self.c1.inventory <= self.c1.consume or self.c2.inventory <= self.c2.consume or self.c3.inventory <= \
                self.c3.consume \
                or self.c4.inventory <= self.c4.consume or self.c5.inventory <= self.c5.consume:
            return True
        return False


def main():
    vs = VirtualSituation()
    distance = vs.get_distance(vs.p1, vs.c1)
    vs.get_observation()
    print(distance)


if __name__ == '__main__':
    main()
