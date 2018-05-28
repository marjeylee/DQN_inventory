"""
车辆
"""


class Vehicle:
    def __init__(self, name=None, location=None, capacity=None, inventory=None
                 , unload_each_time=None, location_name=None):
        self.location = location  # 位置
        self.name = name  # 名称
        self.capacity = capacity  # 容量
        self.inventory = inventory  # 当前车上库存
        self.unload_each_time = unload_each_time  # 每个卸货量
        self.location_name = location_name
