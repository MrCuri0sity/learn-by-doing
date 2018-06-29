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