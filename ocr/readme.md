# 项目说明
ocr模块，识别图片中的身份证并结构化输出
# 文本行检测CTPN
- 用VGG16的前5个Conv stage（到conv5）得到feature map(W\*H*C)
- 在Conv5的feature map的每个位置上取3\*3*C的窗口的特征，这些特征将用于预测该位置k个anchor（anchor的定义和Faster RCNN类似）对应的类别信息，位置信息。
- 将每一行的所有窗口对应的3\*3\*C的特征（W\*3\*3\*C）输入到RNN（BLSTM）中，得到W*256的输出
- 将RNN的W*256输入到512维的fc层
- fc层特征输入到三个分类或者回归层中。第二个2k scores 表示的是k个anchor的类别信息（是字符或不是字符）。第一个2k vertical coordinate和第三个k side-refinement是用来回归k个anchor的位置信息。2k vertical coordinate表示的是bounding box的高度和中心的y轴坐标（可以决定上下边界），k个side-refinement表示的bounding box的水平平移量。这边注意，只用了3个参数表示回归的bounding box，因为这里默认了每个anchor的width是16，且不再变化（VGG16的conv5的stride是16）。回归出来的box如Fig.1中那些红色的细长矩形，它们的宽度是一定的。
- 用简单的文本线构造算法，把分类得到的文字的proposal合并成文本线
- CTPN的数据集制作可以参考：https://blog.csdn.net/qq_36810544/article/details/103372949
# 文字识别CRNN
- 采用BiLstm + ctc实现不定长文字的识别
- ctc算法的详情请参见：https://blog.csdn.net/qq_36810544/article/details/90210550
- 为保证项目版本的统一，本项目全部采用pytorch实现，网上有很多keras版本的实现，但是训练的时候经常会碰到""InvalidArgumentError sequence_length(0) <=30"的错误，具体可以参考：https://blog.csdn.net/qq_36810544/article/details/104271708
# 运行demo
- 下载模型：https://pan.baidu.com/s/1vswXxRqtl01Uux5BPZC4gA 提取码：1lc6，将下载的文件存入weights文件夹下
- idcard.py用于对识别出的身份证信息进行结构化识别
- 运行ocr.py

![image](https://github.com/zzubqh/Video-Verification/blob/master/ocr/test/idcard-demo.jpeg) 
识别结果
![image](https://github.com/zzubqh/Video-Verification/blob/master/ocr/test/ocr%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C.png)
# 参考连接
- https://github.com/opconty/pytorch_ctpn
- https://arxiv.org/abs/1609.03605
- http://arxiv.org/abs/1507.05717
- https://github.com/meijieru/crnn.pytorch

