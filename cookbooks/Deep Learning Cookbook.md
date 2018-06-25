## Deep Learning cookbook

[toc]
### 写在前面
本文主要是自己学习总结使用，有许多结论和图片来自网络，有侵权请联系792706244@qq.com，马上删除。
### Models
#### DNN
- 前向传播


<center>
<img src="http://p9l49hjew.bkt.clouddn.com/9ad0c03dec66ea163087f9ddd380961c.jpg" width="50%", height="50%">
</center>

`"+1"`称之为偏置点，公式中用`$b_i^{(l)}$`表示，`$z_1^{(2)}$`表示第二层第一个神经元的logit，`$W_{12}^{(2)}$`表示第二层第一个神经元和第一层第二个神经元的系数，`$a_1^{(2)}$`表示第二层第一个神经元的激活值，其中第一层激活即为数据本身，`f`表示激活函数，对于一个神经元的前向传播可以表示如下：

```math
z_1^{(2)} = W_{11}^{(2)}*a_1^{(1)} + W_{12}^{(2)}*a_2^{(1)} + W_{13}^{(2)}*a_3^{(1)} + b_1^{(2)} 

a_1^{(2)} = f(z_1^{(2)})
```
对于一层神经元可以表示如下：
```math
\mathbf{Z}^{(l+1)} = \mathbf{W}^{(l+1)} \mathbf{a}^{(l)} + \mathbf{b}^{l+1}

\mathbf{a}^{(l+1)} = f(\mathbf{Z}^{(l+1)})
```

- 反向传播
![](http://p9l49hjew.bkt.clouddn.com/18e44741836fcd111b1d0a366f448014.jpg)
在上图中`l-1`层下标用`i`表示，`l`层下标由`j`表示，`l+1`层下标由`k`表示，以计算`$W_{ji}^{l}$`和`$b_j^{l}$`为例推导反向传播过程。
```math
W^{(l)}_{ji}= W^{(l)}_{ji} - \alpha \frac{\partial{J(W, b)}}{\partial{W_{ji}^{(l)}}}

b^{(l)}_j = b^{(l)}_j - \alpha \frac{\partial{J(W, b)}}{\partial{b_{j}^{(l)}}}
```
在前向传播中
```math
Z_j^{(l)} = \displaystyle{\sum_{i=1}^{n_{(l-1)}}}W_{(ji)}^{(l)}a_i^{(l-1)} + b_j^{(l)}
```
有下式成立
```math
\frac{\partial{J(W, b)}}{\partial{W^{(l)}_{ji}}} = \frac{\partial{J(W, b)}}{\partial{Z_j^{(l)}}} * \frac{\partial{Z_j^{(l)}}}{\partial{W_{ji}^{(l)}}} = \frac{\partial{J(W, b)}}{\partial{Z_j^{(l)}}} * a_i^{(l-1)}

\frac{\partial{J(W, b)}}{\partial{b^{(l)}_{j}}} = \frac{\partial{J(W, b)}}{\partial{Z_j^{(l)}}} * \frac{\partial{Z_j^{(l)}}}{\partial{b_{j}^{(l)}}} = \frac{\partial{J(W, b)}}{\partial{Z_j^{(l)}}} 

```
由上式可知，关键在于求`$\frac{\partial{J(W, b)}}{\partial{Z_j^{(l)}}}$`，将`$\frac{\partial{J(W, b)}}{\partial{Z_j^{(l)}}}$`记为`$\delta_i^{(l)}$`。
在反向传播过程中，前层导数和后层相关：
```math
Z_k^{(l+1)} = \displaystyle{\sum_{i=1}^{n_{(l)}}}W_{kj}^{(l+1)}a_j^{(l)} + b_k^{(l+1)}
```
根据上式：
```math
\delta_{i}^{(l)} = \frac{\partial{J(W, b)}}{\partial{Z_j^{(l)}}} = \displaystyle{\sum_{k=1}^{n_{(l+1)}}}\frac{\partial{J(W, b)}}{\partial{Z_k^{(l+1)}}} * \frac{\partial{Z_k^{(l+1)}}}{\partial{Z_j^{(l)}}} =  \displaystyle{\sum_{k=1}^{n_{(l+1)}}}\frac{\partial{J(W, b)}}{Z_k^{(l+1)}} * \frac{\partial{Z_k^{(l+1)}}}{\partial{a_j^{(l)}}} * \frac{\partial{a_j^{(l)}}}{\partial{Z_j^{(l)}}} = \displaystyle{\sum_{k=1}^{n_{(l+1)}}} \delta_k^{(l+1)} * W_{kj}^{(l+1)} * f^{'}(Z_j^{(l)})
```
因此：
```math
\frac{\partial{J(W, b)}}{\partial{W^{(l)}_{ji}}} = \delta_{i}^{j} * a_{i}^{l-1} = (\displaystyle{\sum_{k=1}^{n_{(l+1)}}} \delta_k^{(l+1)} * W_{kj}^{(l+1)} * f^{'}(Z_j^{(l)})) * a_{i}^{(l-1)}

\frac{\partial{J(W, b)}}{\partial{b^{(l)}_{j}}} = \displaystyle{\sum_{k=1}^{n_{(l+1)}}} \delta_k^{(l+1)} * W_{kj}^{(l+1)} * f^{'}(Z_j^{(l)}) 
```
对于最后一层：
```math
\delta_p^{L} = \frac{\partial{J(W, b)}}{\partial{Z_p^{L}}} = \frac{\partial{J(W, b)}}{\partial{a_p^{L}}} * \frac{\partial{a_p^{L}}}{\partial{Z_p^{L}}} = \frac{\partial{J(W, b)}}{\partial{a_p^{L}}} * f^{'}(Z_p^{L})
```

#### CNN
#### RNN
#### GAN
#### Attention


 
### Activation Functions
max-out

### Regularization Tricks

#### Dropout
- dropout 作用
    - 大规模的深度神经网络参数很多，训练速度很慢，很容易过拟合，dropout能够提高训练速度和减少过拟合
    - 大规模的神经网络需要大量的训练数据使得参数得到充分的训练，然而现实中往往会出现训练数据不足的情况，dropout能够部分减少这种影响。
    - 低代价的模型平均，理论上讲，n个神经元的神经网络，模型结构最多有`$2^n$`中，但是参数总数任然是`$O(n^2)$`或者更少。


- how dropout works<br> 
dropout 在前向传播的时候，每个神经元以一定的概率暂时被丢弃，**这里丢弃是暂时性的，并不是说抛弃了，很有可能这一次被丢弃的神经元会出现在下一次训练的网络中**。由于丢弃神经元的随机性，在每次的训练过程中，神经网络的结构都可能不同。训练结束后，保留所有节点用于测试。上述过程描述成下图所示：
<center>
<img src="http://p9l49hjew.bkt.clouddn.com/db1be2ffeb346edcd05e71f9e1f1ea4c.jpg"   width="50%" height="50%"/>
</center>

在没有使用dropout，前向的操作如下面公式所示：
```math
z_i^{(l+1)} = \mathbf{w}_i^{(l+1)} * \mathbf{a}^l + b_i^{(l+1)}

a_i^{(l+1)} = f(z_i^{(l+1)})   
```
在使用dropout后，前向的操作如下面公式所示：
```math
r_j^{(l)} ∼ Bernoulli(p)

\mathbf{\tilde{a}}^l = \mathbf{a}^l * \mathbf{r}^l

z_i^{(l+1)} = \mathbf{w}_i^{(l+1)} * \mathbf{\tilde{a}}^l + b_i^{(l+1)}

a_i^{(l+1)} = f(z_i^{(l+1)}) 
```

其中`$\mathbf{r}^l$`在`Goodfellow 《deep learning》`书中称之为掩码，掩码一般服从`$Bernoulli$`分布，分布由一个采样概率决定，其实这个采样概率就是我们在使用dropout时候设定的`dropout prob`，`Goodfellow`根据dropout使用位置不同，给出了一个采样的经验值，**超参数的采样概率为 1，隐藏层的采样概率通常为 0.5，输入的采样概率通常为 0.8**。

**权重比例推断规则**:模型测试时，不会丢弃神经元，但是这个时候需要对所有系数乘以采样概率p，达到模型平均的效果。一个帮助理解的example如下：

<center>
<img src="http://p9l49hjew.bkt.clouddn.com/41016c67fb292f74ee3b983752a80a83.jpg" width="50%" height="50%"/>
</center>


- questions
    - 使用dropout在反向的时候所有神经元都参与梯度下降吗？
    >  Forward and backpropagation for that
 training case are done only on this thinned network

    - 对输入层采样是不是相当于做另外一种cross validation?
    - ==当Dropout作用于线性回归时，相当于每个输入特征 具有不同权重衰减系数的 L2权重衰减==
    - dropout是一种模型bagging吗？<br>
    > 不是，bagging  是独立训练好很多模型，但是dropout其实在训练 过程中参数共享。
    
    - 在常用的深度学习框架中，测试时候需要自己对模型对权重乘以`dropout prob`吗？
    > TODO
    
- tricks
    - Nitish Srivastava, Geffory Hinton 的这篇[论文](chrome-extension://gfbliohnnapiefjpjlpjnehglfpaknnc/pages/pdf_viewer.html?r=http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)和max-norm一起使用有更好的效果
- exps<br>
    **TODO**

#### Weight Decay
#### Batch Normalization
#### Early Stoping
#### Mini Batch
#### max-norm Regularization

### Optimizer 
#### SGD
#### Adam

### References
- [1] [ufldl教程](http://ufldl.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
