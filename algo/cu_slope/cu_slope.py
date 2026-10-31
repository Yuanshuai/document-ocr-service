import cv2
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndi
from config import slope_width, rotate_batch, rotate_batch_time, rotate_step


def read_low_pic(img_path):
    img = cv2.imread(img_path)
    target_width = min(img.shape[1], slope_width)
    target_height = int(img.shape[0] * target_width / img.shape[1])
    img = cv2.resize(img, (target_width, target_height))
    return img


def matrix_positive(angle):
    return [[np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]]


def matrix_negative(angle):
    return [[np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]]


def offset_positive(angle, center):
    return cp.asarray(center - center.dot(matrix_positive(angle)))


def offset_negative(angle, center):
    return cp.asarray(center - center.dot(matrix_negative(angle)))


def slope_correct(img_path):
    import math
    image = read_low_pic(img_path)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center = np.array([image.shape[0] / 2, image.shape[1] / 2])
    image = cp.asarray(image)
    rotate_time = rotate_batch * rotate_batch_time
    result = cp.zeros(2 * rotate_time + 1)
    for i in range(0, rotate_batch_time):
        left_list = cp.zeros((rotate_batch, image.shape[0], image.shape[1]))
        right_list = cp.zeros((rotate_batch, image.shape[0], image.shape[1]))
        for j in range(0, rotate_batch):
            target_angel = rotate_step * (i * rotate_batch + j) * math.pi / 180
            left_list[j] = ndi.affine_transform(image, cp.asarray(matrix_negative(target_angel)).T,
                                                offset=offset_negative(target_angel, center), cval=255.0, order=0)
            right_list[j] = ndi.affine_transform(image, cp.asarray(matrix_positive(target_angel)).T,
                                                 offset=offset_positive(target_angel, center), cval=255.0, order=0)
        left_res = cp.sum(left_list, axis=2) / 1000
        left_res = cp.var(left_res, axis=1)
        right_res = cp.sum(right_list, axis=2) / 1000
        right_res = cp.var(right_res, axis=1)
        for j in range(0, rotate_batch):
            result[rotate_time - i * rotate_batch - j] = left_res[j]
            result[rotate_time + i * rotate_batch + j] = right_res[j]
    result = result.get()
    index = np.argmax(result)
    angle = (index - rotate_time) * rotate_step
    img_result = cv2.imread(img_path)
    if angle == 0:
        return img_result
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    del image
    del result
    del left_res
    del right_res
    del left_list
    del right_list
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())
    # print(pinned_mempool.n_free_blocks())
    return cv2.warpAffine(img_result, matrix, [img_result.shape[1], img_result.shape[0]], borderValue=(255, 255, 255))


if __name__ == '__main__':
    import time

    start = time.time()
    img = slope_correct('/tmp/_nlgp1u4v7.jpg')
    print(time.time() - start)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
