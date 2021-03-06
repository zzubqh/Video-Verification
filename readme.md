# 前言
目的用于离线视频鉴证，python3.6 + pytorch1.5实现。
# 目的
作为CV和NLP的综合演示demo。
# 功能
 - 检测视频中的人脸是否全程无遮挡
 - 验证视频中的人是否是视频中使用的身份证
 - 验证视频中出示的身份证是否与预留的身份证信息吻合
 - 验证视频中的用户是否正确回答了审核人员的提问
# 目录结构
 - asr: 用于提供语音到文字的转换(待更新)
 - check_answer: 用于校验用户的回答(待更新)
 - chinese_corrected: 用于中文词语的自动纠错(待更新)
 - face_regcognition: 用于提供人脸识别功能
 - ocr:  用于提供身份证的内容识别
# 说明
 - 先上传的都是inference代码，已完成ocr和人脸识别相关功能，中文的自动纠错和用户的回答判定以及脸部的遮挡检测还在整理中。后续持续更新各功能模块和用于训练的代码。
 - 本项目中用到了很多开源项目，若用于商业项目还请联系源作者和我，谢谢！
 - 若需要在移动端部署，还需要做模型量化/压缩、转换工作，模型转换可以参考我的博客：https://blog.csdn.net/qq_36810544/article/details/106911025。 
 - 如果有错误还请指出，可以在我博客下留言或者给我发邮件：zzubqh@gmail.com
 - 最后，欢迎star或fork
