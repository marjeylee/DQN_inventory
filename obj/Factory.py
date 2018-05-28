"""
工厂,假设前提：工厂库存无线，生产率无线
"""


class Factory:
    def __init__(self, name=None, location=None, identify=None):
        self.location = location  # 位置
        self.name = name  # 名称
        self.identify = identify
