import cv2
import numpy as np


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


def make_border_alg(image):
    return image
