## 总结

这个简单神经网络采用了多层感知机的形式，一个输入层，三个中间层，一个输出层。

中间层全部使用的 2 维卷积模型，激活函数使用的 ReLU。在第 2、第 3 个中间层使用前，加入了正弦时间步的嵌入。

主类为 NaiveConvNet，辅助副类为 TimeAwareConv。

## NaiveConvNet

NaiveConvNet 中构造函数 init 所定义的 TimeAwareConv 类，相当于初始化了三个新的 TimeAwareConv 对象。在这里传入的参数同样也走的是 TimeAwareConv 中的构造函数。

而 forward 函数中调用的这三个新对象的方法，实际上走的是 TimeAwareConv 中的 forward 函数。

## TimeAwareConv

构造函数：

1. time_proj：一个线性网络，用于调整 t_emb 的维度，与图像维度，也就是通道数对齐。
2. conv：二维卷积
3. act：激活函数

forward 函数：

首先判断 t_emb 是否为 None：

1. 如果是，直接对图像进行卷积，然后接一个激活函数
2. 如果不是，先对 t_emb 进行维度（通道数）对齐，再仿照 x 的形状进行对齐。然后对图像 x 进行卷积，叠加一个 t_emb，再进行激活函数。
