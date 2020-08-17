# -*- coding:utf-8 -*-
# name: face
# author: bqh
# datetime:2020/8/14 18:05
# =========================

from face_detect import FaceDetection
from models.inception_resnet_v1 import FaceRec
import torch
import numpy as np
import cv2


class Face(object):
    def __init__(self, ssd_model_path, resnet_model_path):
        self.detect_handle = FaceDetection(ssd_model_path)
        self.face_handle = FaceRec(resnet_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_score = 0.5
        self.input_size = 150  # inception resnet网络的输入大小，将人脸图片调整成最大边150的图片
        self.avg_pixel = (127.5, 127.5, 127.5)
        self.std_pixel = (128, 128, 128)

    def img_normalization(self, image):
        im_shape = image.shape
        im_size_max = np.max(im_shape[0:2])
        resize = float(self.input_size) / float(im_size_max)
        img = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img = img.astype('float')
        # 图片需要归一化后再送inception_resnet
        img -= self.avg_pixel
        img /= self.std_pixel
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
        img = img.to(self.device)
        return img

    def _get_face_tensor(self, img, bbox):
        left_x = int(bbox[0])
        left_y = int(bbox[1])
        right_x = int(bbox[2])
        right_y = int(bbox[3])
        face_img = img[left_y:right_y, left_x:right_x]
        face_img = self.img_normalization(face_img)
        return face_img

    def get_face_bbox(self, img):
        """
        :param img: 输入的原始图片
        :return: 返回图片中包含的人脸bbox，没有检测到则返回空列表
        """
        face_bboxs = self.detect_handle.predict(img)
        return face_bboxs

    def get_face_embedding(self, img, face_bboxs):
        """
        :param img: 原始图片
        :param face_bboxs: 图片上对应的人脸坐标
        :return: 返回每个bbox中的人脸对应的embedding数据
        """

        embeddings = []
        for bbox in face_bboxs:
            if bbox[4] < self.confidence_score:
                continue
            face_tensor = self._get_face_tensor(img, bbox)
            face_embedding = self.face_handle.predict(face_tensor)
            embeddings.append(face_embedding.data.cpu().numpy()[0])

        if len(embeddings) > 0:
            embeddings = np.array(embeddings)
            embeddings = np.reshape(embeddings, (-1, 512))
        return embeddings