利用opencv dnn模块加载FCN网络模型实现图像中目标分类后进行分割颜色填充

文件包含：
1.测试图片 pic/
2.代码文件 main.cpp

学习笔记
1.传统的CNN结构中，前5层是卷积层，第6层和第7层分别是一个长度为4096的一维向量，第8层是长度为1000的一维向量，分别对应1000个类别的概率。
FCN将这3层表示为卷积层，卷积核的大小(通道数，宽，高)分别为（4096,1,1）、（4096,1,1）、（1000,1,1）。所有的层都是卷积层，故称为全卷积网络。

2.上采样通过反卷积实现。对第5层的输出（32倍放大）反卷积到原图大小，一些细节无法恢复。
将第4层的输出和第3层的输出也依次反卷积，分别需要16倍和8倍上采样。

3.优点：接受任意大小的输入图像，不要求所有的训练图像和测试图像具有同样的尺寸；更加高效，避免了使用像素块而带来的重复存储和计算卷积的问题。
缺点：结果不够精细,对图像中的细节不敏感;缺乏空间一致性。

参考链接：[全卷积网络（FCN）与图像分割](https://blog.csdn.net/taigw/article/details/51401448)
