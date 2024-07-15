import cv2
import numpy as np
import time


def show_image(image):
    cv2.imshow('show_image', image)
    cv2.waitKey(0)


def rgb_to_gray_cv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def rgb_to_binary_cv(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image


def make_kernel(width, height):
    kernel = np.ones((width, height), int)
    return kernel


def func_time_wrapper(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        result = end_time - start_time
        print('func time is %.3fs' % result)
        return res

    return inner
