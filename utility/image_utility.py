# -*- coding: utf-8 -*-
__author__ = 'l'
__date__ = '2018/5/24'
import cv2 as cv


def show_img(new_img):
    """
    显示图片
    :param new_img:
    :return:
    """
    cv.imshow("Image", new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
