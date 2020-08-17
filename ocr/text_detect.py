# -*- coding:utf-8 -*-
# name: ocr_main
# author: bqh
# datetime:2020/8/15 15:55
# =========================

import torch
import torch.nn.functional as F
from models.ctpn import CTPN
from utils.ctpn_utils import *


class TextDetection(object):
    def __init__(self, model_path):
        torch.set_grad_enabled(False)
        self.cuda_enabled = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_enabled else "cpu")
        self.model = CTPN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.input_h = 720  # 将输入图片统一缩放到高为720
        self.nms_threshold = 0.3
        self.confidence_threshold = 0.2
        self.avg_pixel = (103.939, 116.779, 123.68)  # BGR格式

    def img_normalization(self, image):
        im_shape = image.shape
        resize = float(self.input_h) / float(im_shape[0])
        img = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img = img.astype('float')
        scale = np.array([img.shape[1], img.shape[0]] * 4)
        img -= self.avg_pixel
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.to(self.device)
        return img, scale, resize

    def predict(self, img, expand=True):
        """
        :param img: cv2格式，BGR
        :param expand:
        :return: 文本行的bbox和每个bbox的分数;bbox:[n, 8] 8个坐标分别为(左上，右上，左下，右下)
        """
        image, scale, resize = self.img_normalization(img)
        _, _, im_height, im_width = image.shape
        cls, regr = self.model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(im_height / 16), int(im_width / 16)), 16)
        bbox = bbox_transfor_inv(anchor, regr)
        bbox = clip_box(bbox, [im_height, im_width])

        fg = np.where(cls_prob[0, :, 1] > self.confidence_threshold)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        keep_index = filter_bbox(select_anchor, 16)

        # nms
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, self.nms_threshold)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        text_connect = TextProposalConnectorOriented()
        text_bboxs = text_connect.get_text_lines(select_anchor, select_score, [im_height, im_width])

        if expand:
            for idx in range(len(text_bboxs)):
                text_bboxs[idx][0] = max(text_bboxs[idx][0] - 10, 0)
                text_bboxs[idx][2] = min(text_bboxs[idx][2] + 10, im_width - 1)
                text_bboxs[idx][4] = max(text_bboxs[idx][4] - 10, 0)
                text_bboxs[idx][6] = min(text_bboxs[idx][6] + 10, im_width - 1)

        bboxs = text_bboxs[:, 0:8] / resize
        score = text_bboxs[:, -1]
        return bboxs, score

    def draw_result(self, image, text_bboxs):
        for box in text_bboxs:
            box = [int(j) for j in box]
            cv2.line(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.line(image, (box[0], box[1]), (box[4], box[5]), (0, 0, 255), 2)
            cv2.line(image, (box[6], box[7]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.line(image, (box[4], box[5]), (box[6], box[7]), (0, 0, 255), 2)
        cv2.imshow('text_dect', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = r'E:\code\OCR\ocr.pytorch-master\test_images\idcard-demo.jpeg'
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image = cv2.imread(img_path)
    weights = os.path.join(root, 'ocr/weights/ctpn.pth')
    detector = TextDetection(weights)
    bboxs, _ = detector.predict(image)
    detector.draw_result(image, bboxs)
