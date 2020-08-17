# 项目说明
人脸识别模块，用于人脸检测和识别
# 人脸检测
- 采用基于ssd的人脸检测网络，标记5个关键点。测试视频：test/video.mp4
- 人脸检测demo：face_detect.py
- 运行demo，下载预训练模型[face_detect.pth](https://pan.baidu.com/s/1a_AxbkZwbCEash43ovouJQ)，提取码：9tef,存入weights文件夹下
- 然后运行：
`python face_detect.py`
- 返回的bbox已经还原到了输入图片尺寸下，在后续分类时可以直接使用
# 人脸识别
- 采用基于incepetion_resnet_v1网络在vggface数据集下训练
- face.py已经封装好了对图片中的人脸提取和获取对应的embeding
- 在上面提供的链接中下载预训练模型20180402-114759-vggface2.pt，存入weights文件夹下
- 运行face_demo.py，将自动打开笔记本的摄像头进行人脸识别，将代码中的img_dir修改成你的数据集路径，一张图片一个人，图片名为label
- 计算相似度的时候只是简单的循环做余弦相似度，在数据很大的时候还需另行处理，此处只做演示使用
# 演示demo
![image](https://github.com/zzubqh/Video-Verification/blob/master/face_regcognition/result/face_demo.png)
# 参考项目
- https://github.com/timesler/facenet-pytorch
