## Attention机制

<!-- https://blog.csdn.net/weikai_w/article/details/103872506
https://blog.csdn.net/qq_41058526/article/details/80783925
https://www.cnblogs.com/rainwelcome/p/12775158.html
https://www.zhihu.com/question/68482809
https://blog.csdn.net/hahajinbu/article/details/81940355
https://www.statist.cn/2020/06/26/Attention/
https://drive.google.com/file/d/1WWBx0p2wi3AUjLwX7B11Ty9cB9Xed7WM/view?usp=sharing
https://blog.csdn.net/xiaosongshine/article/details/90573585

https://tobiaslee.top/2017/08/15/Attention-Mechanism-%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/?nsukey=V%2FjG4ffNSKB%2BuFrDuO6smXiszBk53uNHsKs%2F5D2N2iYvLlcsLvt7Jn6PG3eqo5QJsWtrVlbbJIlAe%2BWKr9NiBDRwr2ywPs8nSg2xKm8Dw%2FUXBJiiGdYWazxkotA8pi7%2BOPACiChd4xJM1%2BzEwdcfXVkoYz2rHnMvZJHfwCUhj6XSmprRP5XnitFGqizfJ3HukGxR5uC3Kb3kGewXPwQbmA%3D%3D -->

<div align=center>
    <img src="zh-cn/img/attention/p0.jpg" /> 
</div>

我们介绍了seq2seq(sequence-to-sequence)模型是对序列的item建模，这样的item可以有多种，例如文字，语音，或者图像等，输出的内容也是序列的item。分别根据Encoder和Decoder模态的不同，Seq2Seq可以用来处理多模态的端到端任务（end-end），比如：Text-Text， Text-Image， Image-Text, Speech-Text, Text-Speech等。

但其明显存在如下缺陷：

+ 训练速度慢，计算无法并行化（根本原因是encoder和decoder阶段中的RNN/LSTM/GRU的结构，由于decoder实际上是一个语言模型，因此其时间复杂度为O(n)）；
+ 对于长序列来说，效果不理想；无论多长的句子，都会被编码成一个固定长度的向量，作为解码器的输入。那么对于长句子而言，编码过程中势必就会有信息的损失；

Seq2Seq的缺陷可以采用Attention机制解决。本节我们将给读者介绍 soft attention, hard attention, global attention, local attention,动态atetntion，静态attention, 关于self attention, multi-head attention, mask multi-head attention,我们将在Transformer(Attention is all your need)的讲解中介绍。

### 1. soft attention, global attention, 动态attention

这三个其实就是soft attention，也是最常见的attention，是在求注意力分配概率分布的时候，对于输入句子`X`中任意一个单词都给出个概率，是个概率分布，把attention变量（context vecor）用$c_t$
表示，attention得分在经过了softmax过后的权值用$\alpha$表示

<div align=center>
    <img src="zh-cn/img/attention/p9.png" /> 
</div>
*论文：Neural machine translation by jointly learning to align and translate*

<div align=center>
    <img src="zh-cn/img/attention/p10.png" /> 
</div>

**Bahdanau Attention**是Bahdanau在2015年提出（Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/pdf/1409.0473.pdf>），在Seq2Seq模型的基础上，加上Attention机制做机器翻译任务，实现了更好的效果。

<div align=center>
    <img src="zh-cn/img/attention/p1.jpg" /> 
</div>

<div align=center>
    <img src="zh-cn/img/attention/p2.png" /> 
</div>

+ Encoder是一个双向RNN，这样做的好处就是能够在一些语序影响翻译的语言中表现得更好，比如：后面的词语对冠词、代词翻译提供参考。
+ 对输入的每一个隐变量进行attention加权，解码的时候将整个context信息进行weighted-sum传入，这种方法称为soft attention（软对齐），也叫global attention，因为每个输入词的$h_j=concat(\overleftarrow{h_j},\overrightarrow{h_j})$ 都参与了权重的计算，这种方法方便梯度的反向传播，便于我们训练模型。
+ Encoder阶段并没有什么特殊的地方
+ Decoder阶段与传统encoder-decoder模型相同，只不过context vector `c`变为了$c_i$.其中的$c_i$是对每一个输入的${x_1,...,x_T}$encoder后的隐状态进行weighted sum（如上图所示）

<div align=center>
    <img src="zh-cn/img/attention/p3.png" /> 
</div>

注意： $h_j$为encoder的隐藏状态，$s_j$为decoder的隐含状态； $a_{ij}$的值越高，表示第$i$个输入出在第$j$个输入上分配的注意力越多，在生成第i个输出的时候受到第j个输入的影像也就越大；$e_{ij}$是encoder $i$处隐藏状态和decoder $j-1$处的隐藏状态的匹配（match),此处的alignment model $a$是和其他神经网络一起训练的，其反应了$h_j$的重要性。

其余部分均与传统的seq2seq相同，语言模型：

<div align=center>
    <img src="zh-cn/img/attention/p4.png" /> 
</div>

在RNN中，$t$时刻的隐藏状态$s_t$:
<div align=center>
    <img src="zh-cn/img/attention/p5.png" /> 
</div>

**Luong Attention** 论文：Effective Approaches to Attention-based Neural Machine Translation <https://arxiv.org/abs/1508.04025>
该模型与Bahdanau Attention类似，不同点在于：介绍了几种不同的score function方法。Luong对于两种Attention（Global Attention和Local Attention）进行描述，其核心在于：在获得上下文`c`的计算中，是否所有Encoder的hidden states都参与计算，全部参与的就称为Global Attention，部分参与的就称为Local Attention。如下图所示，蓝色部分是Encoder，红色部分表示Decoder，灰色部分表示将attention的$C_t$和decoder的$h_t$链接再经过softmax分类的过程。其中，所有的hidden states都参与了$C_t$的计算。

<div align=center>
    <img src="zh-cn/img/attention/p6.jpg" /> 
</div>

关于attention权重还是通过softmax获得，表示源端第`s`个词和目标端第`t`个输出所对应的权重，$\bar{h}_ s$表示源端的词对应的状态，$h_t$表示目标端的时序对应的状态，通过这两个状态的分数获得其权重：

<div align=center>
    <img src="zh-cn/img/attention/p7.jpg" /> 
</div>


其分数由下式score函数得到：

<div align=center>
    <img src="zh-cn/img/attention/p8.jpg" /> 
</div>

Bahdanau论文中用的是第三种，而在这篇论文中发现，第一种对于 Global Attention 效果更好，而第二种应用在 Local Attention 效果更好。

### 2. hard attention

soft attention是给每个单词都赋予一个单词match概率，那么如果不这样做，直接从输入句子里面找到某个特定的单词，然后把目标句子单词和这个单词对齐，而其它输入句子中的单词硬性地认为对齐概率为0，这就是Hard Attention Model的思想。

<div align=center>
    <img src="zh-cn/img/attention/p11.png" /> 
</div>

hard attention的这个pt，跟下面讲的local attention机制的找法差不多。hard attention 一般用在图像里面，当图像区域被选中时，权重为1，剩下时候为0。

hard attention的思想是在寻址时不再考虑所有value的加权求和，而是只考虑用最重要的value来表示上下文向量。这个value可以是通过取最大attention权重的value，也可以是通过对attention score表示的多项式分布采样得到的value。hard attention并不常见，因为有max pooling或者采样操作，所以整个模型不再可微，需要通过方差约归或者强化学习等技术来帮助训练。


### 3. local attention

Luong et al. 2015提出了局部注意力(local attention)和全局注意力机制(global attention)。由于全局注意力需要考虑所有的value的加权和，计算量大。所以考虑只在一个窗口内进行加权求和，即局部注意力(local attention)。

<div align=center>
    <img src="zh-cn/img/attention/p12.png" /> 
</div>

如上图所示，这里很明显的不同是：只有部分 hidden states 参与了 $C_t$ 的计算，另外多了一个 $p_t$，这是用于指示对齐位置（也就是哪一部分 $h_i$ 参与上下文运算的计算）的一个实数.

<div align=center>
    <img src="zh-cn/img/attention/p13.jpg" /> 
</div>

有2种对齐位置的方式，分别为：

+ monotonic alignment： 直接设置$p_t=t$的一对一方式（显然在NMT场景中不符合逻辑），然后窗口内的attention矩阵还是通过下式计算；

<div align=center>
    <img src="zh-cn/img/attention/p14.jpg" /> 
</div>

+ Predictive alignment

$$p_t=S.sigmoid(v^{T}_ p\tanh(W_ph_t))$$

这里的$S$是源句的长度，这样通过$sigmoid$ 函数我们就能保证$p_t$ 一定在我们的句子范围之中，$v$和$W$ 都是需要模型去学习的参数。

选取在窗口范围$[p_t-D，p_t+D]$的 hidden states，计算权重向量$a_t$

$$a_t(s)=align(h_t,\bar{h}_ s)\exp(-\frac{(s-p_t)^2}{2\sigma^2})$$

这里的$D$(窗口)是通过经验选取的参数（玄学），而$a_t$就是一个固定长度为$2D+1$的向量，实际上就是在原来的对齐函数乘上了一个高斯分布来体现距离对权重的影响。

几种attention的比较结果：

这里，dot对global更好，general对local更好，-m表示Monotonic alignment, -p表示Oredictive alignment

<div align=center>
    <img src="zh-cn/img/attention/p15.png" /> 
</div>


### 4.静态attention

静态attention,对输出句子共用一个St的attention就够了，一般用在Bi-LSTM的首尾hidden state输出拼接起来作为$s_t$（在图所示中为$u$）

<div align=center>
    <img src="zh-cn/img/attention/p16.png" /> 
</div>

*论文：Teaching Machines to Read and Comprehend 以及
Supervised Sequence Labelling with Recurrent Neural Networks*


上面这个图是从论文摘过来的静态attention的示意图，有读者可能会注意到的是：这个前面的每个hidden state 不应该都和这里的$u$算一次attention score吗，怎么这里只有一个$r$和$u$进行了交互？

其实这里的$r$表示的是加权平均的self attention(关于self attention我们将在Transformer中详细介绍)，这个权就是attention $c_t$向量，这个图里面把attention $c_t$的计算过程省略了。直接跳到了$c_t$和$s_t$计算真正的$s^{’}t$的部分。这里面用的实际的attention score的计算并不是用点积，是additive attention:

<!-- <div align=center>
    <img src="zh-cn/img/attention/p17.png" /> 
</div> -->

+ 点积attention score(Basic dot-product attention): 这是我们常见的attention score的计算方式

$$e_i=s^{T}h_i \in \mathcal{R}$$

+ 乘法attemtion score(Multiplicative attention):

$$e_i=s^{T}Wh_i \in \mathcal{R}$$

+ 加法attention score(Additive attention):

$$e_i=v^{T}\tanh(W_1h_i+W_2s) \in \mathcal{R}$$




静态attention,其实是指对于一个文档或者句子，计算每个词的注意力概率分布，然后加权得到一个向量来代表这个文档或者句子的向量表示。跟soft attention的区别是，soft attention在Decoder的过程中每一次都需要重新对所有词计算一遍注意力概率分布，然后加权得到context vector，但是静态attention只计算一次得到句子的向量表示即可。（这其实是针对于不同的任务而做出的改变）


### 5.强制前向attention

soft attention在逐步生成目标句子单词的时候，是由前向后逐步生成的，但是每个单词在求输入句子单词对齐模型时，并没有什么特殊要求。强制前向attention则增加了约束条件：**要求在生成目标句子单词时，如果某个输入句子单词已经和输出单词对齐了，那么后面基本不太考虑再用它了，因为输入和输出都是逐步往前走的，所以看上去类似于强制对齐规则在往前走**。

### 6.注意力矩阵

每个输出都有一个长为$T_x$的注意力向量，那么将这些向量合起来看，就是一个矩阵。对其进行可视化，得到如下结果

<div align=center>
    <img src="zh-cn/img/attention/p18.jpg" /> 
</div>

其中$x$轴表示待翻译的句子中的单词(英语)，$y$轴表示翻译以后的句子中的单词(法语)。可以看到尽管从英语到法语的过程中，有些单词的顺序发生了变化，但是attention模型仍然很好的找到了合适的位置。换句话说，就是两种语言下的单词**“对齐”**了。因此，也有人把注意力模型叫做**对齐(alignment)模型**。而且相比于用语言学实现的硬对齐，这种基于概率的软对齐更加优雅，因为能够更全面的考虑到上下文的语境。