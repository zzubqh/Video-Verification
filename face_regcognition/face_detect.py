# -*- coding:utf-8 -*-
# name: inference_ssd
# author: bqh
# datetime:2020/8/14 18:57
# =========================

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils.config import ssd_clf
from utils.prior_box import PriorBox
from utils.nms import nms
import cv2
from models.ssd import SSD
from utils.box_utils import decode, decode_landm


class FaceDetection(object):
    def __init__(self, model_path, net_input_size=300):
        torch.set_grad_enabled(False)
        self.cuda_enabled = torch.cuda.is_available()
        self.cfg = ssd_clf
        self.net = SSD(is_train=False)
        self.net = self.load_model(self.net, model_path)
        self.net = self.net.eval()
        cudnn.benchmark = True
        self.device = torch.device("cuda" if self.cuda_enabled else "cpu")
        self.net = self.net.to(self.device)
        self.input_size = net_input_size
        self.nms_threshold = 0.6
        self.confidence_threshold = 0.02
        self.vis_thres = 0.5
        self.avg_pixel = (103.939, 116.779, 123.68) # BGR格式

    def load_model(self, model, pretrained_path):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if not self.cuda_enabled:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def predict(self, image):
        """
        :param image: 输入图片，cv2格式
        :return: 返回图片中的人脸bbox
        """
        img, scale, resize = self.img_normalization(image)
        loc, conf, landms = self.net(img)
        dets = self.postpreprocess(img, loc, conf, landms, scale, resize)
        bbox = []
        for b in dets:
            if b[4] > self.vis_thres:
                bbox.append(b)
        return bbox

    def img_normalization(self, image):
        im_shape = image.shape
        im_size_max = np.max(im_shape[0:2])
        resize = float(self.input_size) / float(im_size_max)
        img = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img = img.astype('float')
        im_height, im_width, _ = image.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= self.avg_pixel
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        return img, scale, resize

    def postpreprocess(self, image, loc, conf, landms, scale, resize):
        _, _, im_height, im_width = image.shape
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # nms
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]
        dets = np.concatenate((dets, landms), axis=1)

        return dets

    def draw_result(self, image, dets):
        for b in dets:
            if b[4] < self.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return image


if __name__ == "__main__":
    import tqdm
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    face_dect = FaceDetection(os.path.join(root, r"face_regcognition/weights/face_detect.pth"))
    video_path = os.path.join(root, r"face_regcognition/test/check.mp4")
    out_path = os.path.join(root, r"face_regcognition/test/hek_out.avi")
    src_video = cv2.VideoCapture(video_path)
    fps = src_video.get(cv2.CAP_PROP_FPS)
    size = int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out_video = cv2.VideoWriter(out_path, fourcc, fps, size)
    frames = src_video.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm.tqdm(range(int(frames))):
        ret, frame = src_video.read()
        dets = face_dect.predict(frame)
        dest_frame = face_dect.draw_result(frame, dets)
        out_video.write(dest_frame)
    src_video.release()
    out_video.release()
    cv2.destroyAllWindows()