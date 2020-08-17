#-*- coding:utf-8 -*-
# name: face_demo
# author: bqh
# datetime:2020/8/17 11:56
#=========================
import os
from face import *
import time


def create_dataset(face_handle, img_dir):
    import glob
    img_paths = glob.glob(img_dir + '/*.jpeg') + glob.glob(img_dir + '/*.JPG')
    dataset = {}
    for img_path in img_paths:
        img = cv2.imread(img_path)
        bboxs = face_handle.get_face_bbox(img)
        embedding = face_handle.get_face_embedding(img, bboxs)
        key_name = os.path.basename(img_path)
        dataset[key_name.split('.')[0]] = embedding[0]
    return dataset


def caculate_dis(x, y):
    num = np.sum(x * y)  # 若为行向量则 A * B.T
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return np.abs(sim)


def frame_process(face_handle, dataset, frame):
    bboxs = face_handle.get_face_bbox(frame)
    image = frame.copy()
    embeddings = face_handle.get_face_embedding(frame, bboxs)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, face_embedding in enumerate(embeddings):
        max_similar = -10000.0
        name = ''
        for key in dataset.keys():
            dis = caculate_dis(face_embedding, dataset[key])
            if dis > max_similar:
                max_similar = dis
                name = key
        if max_similar < 0.8:
            name = 'other'
        # 绘制结果
        b = bboxs[index]
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = int(b[0])
        cy = int(b[1]) - 12
        cv2.putText(image, name, (cx, cy), font, 0.5, (255, 0, 0))

    return image


def demo():
    img_dir = r'dataset'  # 你的数据集文件夹，一张图片一个人，图片名称为label
    face_handle = Face('weights/face_detect.pth',
                       'weights/20180402-114759-vggface2.pt')
    dataset = create_dataset(face_handle, img_dir)

    capture = cv2.VideoCapture(0)
    while True:
        ref, frame = capture.read()
        start_time = time.time()
        image = cv2.flip(frame, 1, dst=None)
        image = frame_process(face_handle, dataset, image)
        end_time = time.time()
        seconds = end_time - start_time
        fps = np.ceil(1/seconds)
        text = 'fps: {0}'.format(fps)
        cx = frame.shape[0] - 50
        cy = 20
        cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow("face demo", image);
        if cv2.waitKey(30) >= 0:
            capture.release()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()