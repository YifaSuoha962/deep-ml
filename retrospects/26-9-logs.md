### 2026/9/16

**Softmax Activation Function Implementation 中的stability技巧不熟。**
- logits - logits.max(dim=-1, keepdim=True) 让exp后的上界变小，因此防止上溢出。

**Numerically Stable Cross-Entropy 中的gather技巧不熟。**
- 在计算交叉熵时要对softmax取log，用上面的stability技巧后，softmax的输出可能会有0.0的情况，这样log(0.0)就会变成-inf，导致梯度爆炸。优化方案是直接调torch.logsumexp
![alt text](image.png)
- gather时关注两个要素，一是dim，二是index和原始数据的对齐关系：dim表示沿着哪个维度取值，index要在取数的dim上和原始tensor对齐。

------
### 2026/9/17

**Solving linear square systems, 对linalg.lstsq()的输出结果不熟。**
[numpy-api文档](https://numpy.com.cn/doc/stable/reference/generated/numpy.linalg.lstsq.html); [torch-api文档](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html).


------
### 2026/9/18


**多维特征加和的维度广播问题**
- 为什么X(batch, feature) + W(feature,)不需要对W手动扩充维度(unsqueeze.(0))？
- 而(B,) 不能直接和 (B,C,H,W) 相加，需要进行维度对齐？

这涉及到**PyTorch Broadcasting 规则**

<div style="background-color:#FFE0BD; padding:12px 16px; border-radius:8px; border:1px solid #F0C89A;">

PyTorch 广播时，**会从最后一个维度开始向前比较**两个 tensor 的 shape。每一维只要满足下面任意一个条件，就可以广播：

1. 两个维度相等；
2. 其中一个维度是 1；
3. 某个 tensor 在这一维根本不存在，可以视为补了一个 1。

</div>
Case 1符合该原则，而Case 2不符合该原则，因此需要手动扩充维度。


**torch.distributions 使用规则**
torch没给出直接计算概率的api，都是用log_prob()，然后再exp()得到概率。因为概率值通常很小，直接计算概率容易下溢出，而log_prob()的输出是对数值，通常不会下溢出。

samle() 是从分布中采样，得到样本的期望。
[torch.distributions文档](https://docs.pytorch.org/docs/2.14/distributions.html)