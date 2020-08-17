#-*- coding:utf-8 -*-
# name: ocr_main
# author: bqh
# datetime:2020/8/15 15:55
#=========================
from models.crnn import CRNN
from models.keys import alphabetChinese
from text_detect import TextDetection
from PIL import Image
from idcard import IDCard
import cv2


class CardOcrModel(object):
    def __init__(self, ctpn_model_path, lstm_model_path):
        nclass = len(alphabetChinese) + 1
        self.detect_handle = TextDetection(ctpn_model_path)
        self.ocr_handle = CRNN(32, 1, nclass, 256, leakyRelu=False, alphabet=alphabetChinese)
        self.ocr_handle.load_weights(lstm_model_path)

    def _get_text_image(self, image, box):
        left_y = int(box[1])  # 左上
        left_x = int(box[0])
        right_y = int(box[7])  # 右下
        right_x = int(box[6])
        return image[left_y:right_y, left_x:right_x]

    def predict(self, image):
        """
        :param image: cv2打开，BGR格式
        :return: 返回image上的所有字符
        """
        result = []
        text_bboxs, score = self.detect_handle.predict(image)
        for index, bbox in enumerate(text_bboxs):
            text_img = self._get_text_image(image, bbox)
            img = Image.fromarray(cv2.cvtColor(text_img, cv2.COLOR_BGR2RGB)).convert('L')
            text_str = self.ocr_handle.predict(img)
            result.append({'text': text_str,
                           'cx': int(bbox[0]),
                           'cy': int(bbox[1]),
                           'w': int(bbox[6] - bbox[0]),
                           'h': int(bbox[7] - bbox[1]),
                           'degree': 0
                           })
        id_card = IDCard(result)
        print(id_card.idcard_info)


if __name__ == '__main__':
    import os
    import time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(root, 'ocr/test/idcard-demo.jpeg')
    image = cv2.imread(img_path)
    ctpn_model_path = os.path.join(root, 'ocr/weights/ctpn.pth')
    lstm_model_path = os.path.join(root, 'ocr/weights/lstm_ocr.pth')
    ocr = CardOcrModel(ctpn_model_path, lstm_model_path)
    start = time.time()
    ocr.predict(image)
    endtime = time.time()
    print('ocr run time: {0}s'.format(endtime - start))


