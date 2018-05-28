"""
地图描述  在一个（10，10）中包含一辆汽车，两个工厂，5个用户
"""


class MapLocation:
    def __init__(self):
        """
        定义对象位置
        """
        self.factory_location = [(4, 1), (1, 7)]
        self.customer_location = [(4, 3), (7, 4), (2, 6), (8, 7), (7, 8)]
