# -*- coding:utf-8 -*-
# name: tools
# author: bqh
# datetime:2020/8/15 14:43
# =========================
import numpy as np
from numpy import cos, sin
import cv2


def solve(box):
    """
     绕 cx,cy点 w,h 旋转 angle 的坐标
     x = cx-w/2
     y = cy-h/2
     x1-cx = -w/2*cos(angle) +h/2*sin(angle)
     y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

     h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
     w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
     (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

     """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2

    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    if abs(sinA) > 1:
        angle = None
    else:
        angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def rotate(x, y, angle, cx, cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    """
    # angle = angle*pi/180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


def xy_rotate_box(cx, cy, w, h, angle):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    """

    cx = float(cx)
    cy = float(cy)
    w = float(w)
    h = float(h)
    angle = float(angle)
    x1, y1 = rotate(cx - w / 2, cy - h / 2, angle, cx, cy)
    x2, y2 = rotate(cx + w / 2, cy - h / 2, angle, cx, cy)
    x3, y3 = rotate(cx + w / 2, cy + h / 2, angle, cx, cy)
    x4, y4 = rotate(cx - w / 2, cy + h / 2, angle, cx, cy)
    return x1, y1, x2, y2, x3, y3, x4, y4


def union_rbox(result, alpha=0.1):
    """
    按行合并box
    """

    def diff(box1, box2):
        """
        计算box1,box2之间的距离
        """
        cy1 = box1['cy']
        cy2 = box2['cy']
        h1 = box1['h']
        h2 = box2['h']

        return abs(cy1 - cy2) / max(0.01, min(h1 / 2, h2 / 2))

    def sort_group_box(boxes):
        """
        对box进行排序, 并合并box
        """
        N = len(boxes)
        boxes = sorted(boxes, key=lambda x: x['cx'])
        text = ' '.join([bx['text'] for bx in boxes])
        box4 = np.zeros((N, 8))
        for i in range(N):
            cx = boxes[i]['cx']
            cy = boxes[i]['cy']
            degree = boxes[i]['degree']
            w = boxes[i]['w']
            h = boxes[i]['h']
            x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(cx, cy, w, h, degree / 180 * np.pi)
            box4[i] = [x1, y1, x2, y2, x3, y3, x4, y4]

        x1 = box4[:, 0].min()
        y1 = box4[:, 1].min()
        x2 = box4[:, 2].max()
        y2 = box4[:, 3].min()
        x3 = box4[:, 4].max()
        y3 = box4[:, 5].max()
        x4 = box4[:, 6].min()
        y4 = box4[:, 7].max()
        angle, w, h, cx, cy = solve([x1, y1, x2, y2, x3, y3, x4, y4])
        return {'text': text, 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'degree': angle / np.pi * 180}

    newBox = []
    for line in result:
        if len(newBox) == 0:
            newBox.append([line])
        else:
            check = False
            for box in newBox[-1]:
                if diff(line, box) > alpha:
                    check = True

            if not check:
                newBox[-1].append(line)
            else:
                newBox.append([line])
    newBox = [sort_group_box(bx) for bx in newBox]
    return newBox
