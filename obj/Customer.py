"""
用户
"""


class Customer:
    def __init__(self, name=None, location=None, capacity=None, inventory=None
                 , consume=None, identify=None):
        self.location = location  # 位置
        self.name = name  # 名称
        self.capacity = capacity  # 容量
        self.inventory = inventory  # 当前库存
        self.consume = consume  # 每小时消耗量
        self.identify = identify
