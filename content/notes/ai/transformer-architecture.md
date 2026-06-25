---
title: "Transformer 架构详解"
category: "ai"
date: "2026-06-15"
tags: ["transformer", "attention", "deep-learning", "nlp"]
difficulty: "intermediate"
concept: "Self-Attention 机制与 Transformer 架构"
prerequisites: ["线性代数基础", "神经网络基础", "Python"]
---

## 核心概念

Transformer 是 2017 年由 Vaswani 等人在论文 "Attention Is All You Need" 中提出的架构，彻底改变了 NLP 领域。

### Self-Attention 机制

Self-Attention 的核心思想是：让序列中的每个元素都能直接访问所有其他元素，并根据相关性加权聚合信息。

数学表达：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- **Q (Query)**：查询向量，表示"我在找什么"
- **K (Key)**：键向量，表示"我有什么"
- **V (Value)**：值向量，表示"我的内容是什么"
- **√d_k**：缩放因子，防止点积过大导致梯度消失

### Multi-Head Attention

多头注意力通过并行运行多个注意力机制来捕获不同类型的依赖关系：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.shape[0]

        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)
```

### Positional Encoding

由于 Transformer 没有循环或卷积结构，需要显式地注入位置信息：

```python
def sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
```

## 架构总览

```
输入 → Embedding + Positional Encoding
     → Multi-Head Self-Attention
     → Add & Norm
     → Feed Forward Network
     → Add & Norm
     → 输出
```

## 常见面试题

1. **为什么需要缩放因子 √d_k？**
   当 d_k 很大时，点积的值会变得很大，导致 softmax 输出接近 one-hot，梯度接近零。缩放可以缓解这个问题。

2. **Self-Attention 和 Cross-Attention 的区别？**
   - Self-Attention：Q、K、V 来自同一序列
   - Cross-Attention：Q 来自一个序列，K、V 来自另一个序列

3. **Transformer 的时间复杂度是多少？**
   O(n²d)，其中 n 是序列长度，d 是维度。这也是长序列处理的瓶颈。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
