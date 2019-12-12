利用opencv dnn模块加载SSD网络模型实现图像的分类

文件包含：
1.测试图片 pic/
2.网络模型 SSD_300x300/
3.代码文件 main.cpp

学习笔记

1.可以使用
Mat blobImg = dnn::blobFromImage(src,1.0f, Size(300, 300), Scalar(104, 117, 123), false, false);
代替图像预处理，其功能包括了 
1.减均值 2.缩放 3.通道交换（可选）
需要注意：不是所有的深度学习架构执行减均值和缩放！

2.detectionMat 为7 x 10的矩阵，存储的数据依次为 0，目标数组下标，置信概率，目标在图像上的比率（左上右下坐标点）

Powered by sannnnn
