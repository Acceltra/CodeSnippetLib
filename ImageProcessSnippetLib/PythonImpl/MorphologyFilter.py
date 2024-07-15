import cv2
import numpy as np
import PythonCommonTools as pyool


@pyool.func_time_wrapper
def image_erode(image, kernel):
    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]
    kernel_wing_w = int((kernel_w - 1) / 2)
    kernel_wing_h = int((kernel_h - 1) / 2)
    start_pos_x = kernel_wing_w
    start_pos_y = kernel_wing_h
    res_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, kernel_wing_h, kernel_wing_h, kernel_wing_w, kernel_wing_w, cv2.BORDER_REPLICATE)

    for i in range(start_pos_x, image.shape[0] - start_pos_x - 1):
        for j in range(start_pos_y, image.shape[1] - start_pos_y - 1):
            sum_val = image[i - kernel_wing_w: i + kernel_wing_w + 1, j - kernel_wing_h: j + kernel_wing_h + 1]
            if np.sum(sum_val * kernel) < 255 * kernel_w * kernel_h:
                res_image[i, j] = 0
            else:
                res_image[i, j] = image[i, j]

    return res_image


@pyool.func_time_wrapper
def image_dilate(image, kernel):
    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]
    kernel_wing_w = int((kernel_w - 1) / 2)
    kernel_wing_h = int((kernel_h - 1) / 2)
    start_pos_x = kernel_wing_w
    start_pos_y = kernel_wing_h
    res_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, kernel_wing_h, kernel_wing_h, kernel_wing_w, kernel_wing_w, cv2.BORDER_REPLICATE)

    for i in range(start_pos_x, image.shape[0] - start_pos_x - 1):
        for j in range(start_pos_y, image.shape[1] - start_pos_y - 1):
            sum_val = image[i - kernel_wing_w: i + kernel_wing_w + 1, j - kernel_wing_h: j + kernel_wing_h + 1]
            if np.sum(sum_val * kernel) > 0:
                res_image[i, j] = 255
            else:
                res_image[i, j] = image[i, j]

    return res_image


def image_open(image, kernel):
    image = image_erode(image, kernel)
    image = image_dilate(image, kernel)
    return image


def image_close(image, kernel):
    image = image_dilate(image, kernel)
    image = image_erode(image, kernel)
    return image

