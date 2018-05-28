"""
状态对象类
"""


class Status:
    def __init__(self, situation):
        """
        初始化
        :param situation:
        """
        self.situation = situation

    @staticmethod
    def get_feature_num():
        """
        获取一个状态有多少种feature。目前 5个customer+ 1辆车
        :return:
        """
        return 19
