[CNN.md](CNN.md)
> 1. 卷积核的原理
> 2. 感受野
> 3. 图片通过卷积核后，输出特征尺寸的计算
> 4. padding及其实现
> 5. BN：调整每层输入，防止梯度消失
> 6. pooling：减少特征数量，提取图片特征
> 7. droupt
> 8. 数据增强

以下训练实例中都包含断点续训

[cifat10_class.py](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/cifar10_class.py#L19)：数据集为cifa10，使用class搭建CNN

[cifar10_sequential.py](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/cifar10_sequential.py)：数据集为cifar10，使用Sequential搭建CNN

[fashion_augment.py](https://github.com/yiyading/NLP-and-ML/blob/master/AI_PRACTICE/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/fashion_augment.py)：数据集为fashion_mnist，对图像进行数据增强并可视化loss和acc曲线


