import cv2
import numpy as np


class Utils:

    @staticmethod
    def __fill_list(img_list, row, col):
        new_img_list = img_list.copy()
        to_fill = abs(len(img_list) - row * col)
        for _ in range(to_fill):
            new_img_list.append(np.zeros_like(img_list[0]))
        return new_img_list

    @staticmethod
    def __match_dims(img_list):
        return [img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]

    @staticmethod
    def stack_images(row, col, img_list, scale):
        row_merged = []
        img_list_mod = Utils.__match_dims(img_list)
        img_list_filled = Utils.__fill_list(img_list_mod, row, col)
        for i in range(row):
            start, end = i * col, i * col + col
            scaled_img = [cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                          for img in img_list_filled[start: end]]
            row_merged.append(np.hstack(scaled_img))
        if row > 1:
            return np.vstack(row_merged)
        else:
            return row_merged[0]