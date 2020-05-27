## word2vec
------

<!-- NNLM -->
<!-- LBL -->
<!-- RNNLM -->
<!-- C&W -->
<!-- CBOW和skip-gram -->

<!-- https://www.jianshu.com/p/edbeeda5d746 -->

<!-- https://www.jianshu.com/p/0a5c2223a573 -->
<!-- <div align=center>
<img src="zh-cn/img/ResNet/0.png" /> 
</div> -->

<div align=center>
    <img src="zh-cn/img/word2vec/p8.png" /> 
</div>

### 1. NNLM（Bengio，2001,2003）

该模型在学习语言模型的同时，也得到了词向量。NNLM对n元语言模型进行建模，估算$Pr(w_i|w_1,w2,..,w_{i-1})$的值。但与传统方法不同的是，NNLM不通过词频的方法对n元条件概率进行估计，而是直接通过一个神经网络结构，对其进行建模求解。

<div align=center>
    <img src="zh-cn/img/word2vec/p1.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/word2vec/p5.png" /> 
</div>

*神经网络语言模型（NNLM）结构图*

**输入层**： one-hot(可以)、上文concat,采用输入层词$w_{i-(n-1)},...,w_{i-1}$的词向量的顺序拼接

$$x=[e(w_{i-(n-1)});,...;e(w_{i-2},e(w_{i-1}))]$$

**隐藏层**： tanh激活函数

$$h=tanh(b^{(1)}+Hx)$$

**输出层**：$|V|$个节点

$$y=b^{(2)}+Wx+Uh$$

将y转换为对应的概率值：

<div align=center>
    <img src="zh-cn/img/word2vec/p2.png" /> 
</div>

由于Bengio等人的工作只考虑对语言模型的建模，词向量是其副产品，因此他们并没有指出哪一套向量作为词向量效果更好。

**参数更新**：

<div align=center>
    <img src="zh-cn/img/word2vec/p3.png" /> 
</div>

其中θ为模型中的所有参数，包括词向量和网络结构中的权重U、H、b。

**改进**：独热换成分布式的实数词向量。

<div align=center>
    <img src="zh-cn/img/word2vec/p4.PNG" /> 
</div>

!> 注意： 

1. 在NNLM模型中，词向量出现在两个地方，一个是输入层的词向量$e(w)$，另一个是隐藏层的词向量$U$, $U$的维度是$|V|X|h|$，这可以看做是$|V|$个$|h|$维的行向量，其中的每一个向量均可以看做某个词字模型中的另一个词向量，记为$e^{'}(w)$。在不考虑W的情况下，每个词在模型中有两套词向量。通常在实际工作中只用第一个作为词向量。

2. 将$y$展开，得到：
$$y(w_i)=b^{(2)}+e^{'}(w_i)^{T}tanh(b^{(1)}+H[e(w_{i-(n-1),...,e(w_{i-1})})])=E(w_i;w_{i-(n-1):i-1})$$
称为能量函数

### 2. log双线性语言模型（LBL，Mnih和Hinton，2007）

2007年，Mnih和Hinton 在神经网络语言模型（NNLM）的基础上提出了log双线性语言模型（Log-Bilinear Language Model，LBL）。LBL的模型结构是一个log 双线性结构，能量函数为：
$$E(w_i;w_{i-(n-1):i-1})=b^{(2)}+e(w_i)^Tb^{(1)}+e(w_i)^TH[e(w_{i-(n-1)}),...,e(w_{i-1})]$$
与之前的NNLM相比，没有非线性的激活函数tanh。

LBL模型的能量函数与NNLM的能量函数主要有两个区别：

1. LBL 模型中，没有非线性的激活函数tanh，而由于NNLM 是非线性的神经网络结构，激活函数必不可少；
2. LBL 模型中，只有一份词向量e，也就是说，无论一个词是作为上下文，还是作为目标词，使用的是同一份词向量。其中第二点（只有一份词向量），只在原版的LBL模型中存在，后续的改进工作均不包含这一特点。
3. 改进模型有：层级log双线性语言模型（hierarchical LBL，HLBL）和基于向量的逆语言模型（inverse vector LBL，ivLBL)等。

### 3. RNNLM(循环神经网络语言模型)

循环神经网络语言模型（Recurrent Neural Network based Language Model，RNNLM）则直接对$P(w_i|w_1,...,w_{i-1})$进行建模（注意不是$P(w_i|w_{i-(n-1)},...,w_{i-1})$, 该模型就是把NNLM隐藏层变成了RNN,每个隐藏层包含此前所有上文信息。RNNLM里面最厉害的就属ELMO（我们在后续章节会详细讲到）了，该模型利用多层双向LSTM的加权和来表示词向量，其中权重可以根据任务动态调整。

<div align=center>
    <img src="zh-cn/img/word2vec/p6.png" /> 
</div>

*RNNLM模型结构*

RNNLM的核心在于其隐藏层的算法：
$$h(i)=\phi(e(w_i))+Wh(i-1))$$

$\phi$为非线性激活函数，使用迭代的方式直接对所有上文进行建模，$h(i)$表示文本中第i个词$w_i$所对应的隐藏层，该隐藏层由当前词的词向量和上一个词对应的隐藏层$h(i-1)$结合得到，每一个隐藏层包含了当前次的信息及上一个隐藏层的信息，输出层与NNLM计算方法一致。

### 4. C&W模型 (Collobert和Weston，2008)

与前面的三个基于语言模型的词向量生成方法不同，C&W模型是第一个直接以生成词向量为目标的模型。

<div align=center>
    <img src="zh-cn/img/word2vec/p7.png" /> 
</div>

*C&W模型结构图*

语言模型的目标是求解$P(w_i|w_1,...,w_{i-1})$，其中隐藏层到输出层的矩阵运算时最耗时的部分。因此，前面的各个词向量模型中，几乎都有对这一部分做优化的步骤，如层级softmax,分组softma和噪声对比估算。C&W模型的目标是更快速的生成词向量，因此**并没有采取语言模型得方式**，去求解上述条件概率，转而采用了另一种更高效的方式，直接对n元短于打分。对于语料中出现的n元短语，模型会对其打高分；而对于语料中没有出现的随机短语，模型会对其打低分。通过这种方式，C&W模型可以更直接的学习到符合分布假说的词向量。

具体而言，对于整个语料，C&W模型需要最小化：
$$\sum_{(w,c)\in D}\sum_{w^{'}\in V}\max(0,1-score(w,c)+score(w^{'},c))$$
其中$(w,c)$为语料中选出的一个n元短语$w_{i-(n-1)/2},...,w_{i+(n-1)/2}$,$w$为序列中的中间词，也是目标词，即$w_i$, $c$表示$w$的上下文，$w^{'}$为字典中的某一个词。正样本$(w,c)$来自语料，而负样本$(w^{'},c)$则是将正样本序列中的中间词替换为其他词。
即：
$$(w,c)=w_{i-(n-1)/2},...,w_{i+(n-1)/2}$$
$$(w^{'},c)=w_{i-(n-1)/2},...,w_{i-1},w^{'},w_{i+1},...,w_{i+(n-1)/2}$$

C&W模型与NNLM相比，主要的不同点在于C&W模型将目标词放到了输入层，同时输出层也从语言模型的$|V|$个节点变为一个节点,这个节点的数值表示对这组n元短语的打分。这个区别使得C&W模型成为神经网络词向量模型中最为特殊的一个，其它模型的目标词均在输出层，只有C&W模型的目标词在输入层。

下面我们将详细的介绍word2vec的原理和如何实现。

### 5. word2vec

<!-- 词嵌入 https://mp.weixin.qq.com/s/72bNiX8MesA82pthoGgqCQ -->
<!-- cbow https://mp.weixin.qq.com/s/eaiZDWALxym1Vt5qWx5n7A -->
<!-- skip-gram https://mp.weixin.qq.com/s/2Q8ZSUEHtqPM_1MUUkSVnQ -->
<!-- https://www.zybuluo.com/Dounm/note/591752 -->
<!-- https://mp.weixin.qq.com/s/E9t1QFiJpsuZXlohFkcCmg -->
<!-- https://mp.weixin.qq.com/s/lANKP0dUHs27VEZtziQaQQ -->
<!-- https://mp.weixin.qq.com/s/tEDSbLyXkFQc-9FeoXix9w -->

<!-- https://github.com/zlsdu/Word-Embedding/blob/master/word2vec/word2vec_report.md -->
<!-- tensorflow https://mp.weixin.qq.com/s/hfAEgSS1PTZbO1ls-1Rs2g -->
<!-- gensim https://mp.weixin.qq.com/s/qIBtAXDoTsP7yhbcNuuKHg -->
<!-- https://github.com/zlsdu/Word-Embedding/tree/master/word2vec -->

<!-- CBOW
skip-gram
层次softmax huffman树和haffman编码
负采样

keras实现词向量 -->

**1. CBOW(Continous bag of words)**

word2vec是一种将word转为向量的方法，其包含两种算法，分别是skip-gram和CBOW，它们的最大区别是skip-gram是通过中心词去预测中心词周围的词，而CBOW是通过周围的词去预测中心词。

word2vec的方法是在2013年的paper 《Efficient Estimation of Word Representations in Vector Space》中提出的，作者来自google，文章下载链接：<https://arxiv.org/pdf/1301.3781.pdf>

文章提出了这两种方法如下图所示：

<div align=center>
    <img src="zh-cn/img/word2vec/p9.png" /> 
</div>

你现在看这张图可能一头雾水，不知所措，没关系，我们慢慢来学习!

在处理自然语言时，通常将词语或者字做向量化，例如独热编码，例如我们有一句话为：“关注数据科学杂谈公众号”，我们分词后对其进行独热编码，结果可以是：

“关注”： [1,0,0,0]  *向量的维度为词库的大小$|v|$*

“数据科学”： [0,1,0,0]

“杂谈”：  [0,0,1,0]

“公众号”：  [0,0,0,1]

但是独热编码在大量数据的情况下会出现维度灾难，通过观察我们可以知道上面的独热编码例子中，如果不同的词语不是4个而是n个，则独热编码的向量维度为$1Xn$，也就是说，任何一个词的独热编码中，有一位为1，其他n-1位为0，这会导致数据非常稀疏（0特别多，1很少），存储开销也很大（n很大的情况下）。

那有什么办法可以解决这个问题呢？

它的思路是通过训练(词的分布式表示），将每个词都映射到一个较短的词向量上来。这个较短的词向量维度是多大呢？这个一般需要我们在训练时自己来指定。现在很常见的例如300维。
例如下面图展示了四个不同的单词，可以用一个可变化的维度长度表示（图中只画出了前4维），其实可以是多少维由自己指定。假设为4维。

<div align=center>
    <img src="zh-cn/img/word2vec/p10.png" /> 
</div>

大家如果细心，会发现在展示的这些维度中的数字已经不是1和0了，而是一些其他的浮点数。 这种将高维度的词表示转换为低维度的词表示的方法，我们称之为词嵌入（word embedding）。

<div align=center>
    <img src="zh-cn/img/word2vec/p11.png" /> 
</div>

*将一个3维词向量表示转为2维词向量表示*

有意思的发现是，当我们使用词嵌入后，词之间可以存在一些关系，例如：king的词向量减去man的词向量，再加上woman的词向量会等于queen的词向量！

$e(King)-e(Man)+e(Woman)=e(Queen)$$

<div align=center>
    <img src="zh-cn/img/word2vec/p12.png" /> 
</div>

出现这种神奇现象的原因是，我们使用的分布式表示的词向量包含有词语上下文信息。怎么理解上下文信息呢？

其实很简单，我们在上学时，做阅读理解经常会提到联系上下文，所谓的上下文信息无非是当前内容在文本中前后的其他内容信息。如下图所示，learning这个词的上下文信息可以是它左右两边的content标记的内容。

<div align=center>
    <img src="zh-cn/img/word2vec/p13.png" /> 
</div>

试想一下，如果这里的learning换成studying，是不是这句话仍然很合适呢？毕竟这两个单词都是学习的意思。再转换一下思维，由于在当前上下文信息中，learning和studying都可以出现，是不是learning和studying是近义词了呢？没错，在当前的CBOW下确实是这样，甚至man和woman，cat和dog都可能是近义词。其实就是拥有相似或者相同的上下文的多个词可能是近义词或者同义词。

这里慢慢将CBOW的算法思想透露出来了，因为CBOW就是通过当前中心词的上下文单词信息预测当前中心词。

此时再来看CBOW这张示意图，是不是有点感觉了？

<div align=center>
    <img src="zh-cn/img/word2vec/p14.png" /> 
</div>

接下来进入具体的算法模型部分！

首先我们需要训练CBOW模型，该模型的结构如下图：

<div align=center>
    <img src="zh-cn/img/word2vec/p15.png" /> 
</div>

这张图略微复杂，我们需要从最左边开始看，最左边的一列是当前词的上下文词语，例如当前词的前两个词和后两个词，一共4个上下文词。这些上下文词即为图中的$x_{1k},x_{2k},...x_{Ck}$。

这些词是独热编码表示，维度为$1\times V$（虽然图上画得像列向量$V\times 1$，这图画的容易理解错误，其中$|V|$为词库的大小，也就是有多少个不同的词，则独热编码的维度为多少，也就是$|V|$个不同的词）。

每个上下文的词向量都需要乘以一个共享的矩阵$W$，由于整个模型是一个神经网络结构，我们将这个存在于输入层和隐藏层之间的矩阵称为$W_1$，该矩阵的维度为$V\times N$，其中V如前所述，N为我们自己定义的一个维度。独热编码向量$1\times V$乘上维度为$V\times N$的矩阵$W_1$，结果是$1\times N$的向量。

这里因为一个中心词会有多个上下文词，而每个上下文词都会计算得到一个$1\times N$向量，我们将这些上下文词的$1\times N$向量相加取平均(激活函数内部的计算），得到中间层（隐藏层）的向量，这个向量也为$1\times N$，之后，这个向量需要乘以一个$N\times V$的矩阵$W_2$，最终得到的输出层维度为$1\times V$。

然后将$1\times V$的向量softmax归一化(暂时先这么理解，其实这里有trick)处理得到新的$1\times V$向量，在V个取值中概率值最大的数字对应的位置所表示的词就是预测结果。

以上过程是CBOW模型的前向计算过程。

其训练过程如下：

1. 当前词的上下文词语的独热编码输入到输入层。
2. 这些词分别乘以同一个矩阵$W_1$后分别得到各自的$1\times N$向量。
3. 将这些$1\times N$向量取平均为一个$1\times N$向量。
4. 将这个$1\times N$向量乘矩阵$W_2$，变成一个$1\times V$向量。
5. 将$1\times V$向量softmax归一化后输出取每个词的概率向量$1\times V$。
6. 将概率值最大的数对应的词作为预测词。
7. 将预测的结果$1\times V$向量和真实标签$1\times V$向量（真实标签中的V个值中有一个是1，其他是0）计算误差，一般是交叉熵。
8. 在每次前向传播之后反向传播误差，不断调整$W_1$和$W_2$矩阵的值。
9. 预测的时候，做一次前向传播即可得到预测的中心词结果。

你可能会想，word2vec不是要将词转为分布式表示的词嵌入么？怎么变成预测中心词了？怎么变成了语言模型了？

其实我们在做CBOW时，最终要的是$W_1$这个$V\times N$矩阵

因为我们是要将词转换为分布式表示的词嵌入，我们先将词进行独热编码，每个词的向量表示是$1\times V$的，经过乘以$W_1$后，根据矩阵乘法的理解，假设$1\times V$向量中第n位是1，其他是0，则矩阵乘法结果是得到了$W_1$矩阵中的第n行结果，也就是将词表示为了一个$1\times N$的向量，一般N远小于V，这也就将长度为V的独热编码稀疏词向量表示转为了稠密的长度为N的词向量表示。

<div align=center>
    <img src="zh-cn/img/word2vec/p16.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/word2vec/p17.png" /> 
</div>

**2.skip-gram**

有了CBOW的介绍，对于skip-gram model的理解就非常容易了。

<div align=center>
    <img src="zh-cn/img/word2vec/p18.png" /> 
</div>

1. one-hot编码。每个单词形成$V\times 1$的向量；对于整个词汇表就是$V\times V$的矩阵。
2. lookup table查表降维。根据索引映射，将每个单词映射到$d$维空间，通过这样的方式就可以将所有的单词映射到矩阵$W$上（矩阵的形状为$V\times d$），并且每个单词与矩阵中的某列一一对应。
3. skip-gram模型训练。初始化一个d维空间的矩阵作为权重矩阵$W^{′}$，该矩阵的形状为$V\times d$，值得注意的是，目前我们已经有了两个d维空间的矩阵。要清楚这两个矩阵分别都是干嘛的，一个是作为中心词时的向量$v_i$，一个是作为背景词时的向量$u_i$（每个词都有机会成为中心词，同时也会成为其他中心词的背景词，因为窗口再变动）。
4. 取出中心词的词向量$v_{c}$(它的形状为d维的向量$1\times d$)，与权重矩阵$W^{′}$中的其他词做内积，此时会得到每个词的计算结果，即：$v_{c}\times u_{o}^{T}$。
​
5. softmax层的映射。在上一步中得到了V个数字，那么此时我们需要做的就是计算每个词的概率，即：
$$\frac{\exp(u^T_o\times v_c)}{\sum_{i\in V}\exp(u^T_i\times v_c)}$$
大家要注意此时的i它表示的是词典中任意的一个词。因此分母部分的计算量极其大。

6. .概率最大化。不要忘记了此时的学习相当于监督学习，有明确的背景词输入的，期望窗口内的词的输出概率最大。因此我们的目标变为极大化概率$P(w_o \mid w_c)$，在不影响函数单调性的前提下我们变为极大化函数：$\log P(w_o \mid w_c)$（对数似然）,但是，我们都明白，计算一个函数的最大值不如计算一个函数的最小值来的方便，因此这里给这个函数进行单调性变换：$-\log P(w_o \mid w_c)$
7. 极小化目标函数。通过上面的变换，此时已经变为一个数学优化问题，梯度下降法更新参数。
8. 更新中心词的向量$v_c$，上面已经推导出结果了，$v_c := v_c - \alpha * \nabla P(w_o \mid w_c)$，后面减去的即为梯度：
$$\nabla P(w_o \mid w_c)= u_{o} -\sum_{j\in V}(\frac{\exp(u_j^{T} v_c)}{ \sum_{i\in V} \exp(u_i^{T} v_c)}) u_j$$
*简单分析一下：大括号里面的是一个概率值，然后乘上一个矩阵(向量)$u_i$，仍然是一个矩阵，相当于对矩阵进行了加权，然后$u_o$这个背景词矩阵再减去这个加权矩阵，就得到了$v_c$的梯度值。如此给上一个学习率就可以迭代更新了。*

9. 注意到，这里面多了个下标j，这个下标代表的背景词，j的取值范围是$-m \leq j \leq m$, j不能为0，这么做才能与我们最初的设定相对应吧！一个中心词，推断多个背景词。
10. 根据同样的原理，在窗口进行移动了之后，可以进行同样的更新
11. 在词汇表中所有的词都被训练之后。我们得到了每个词的两个词向量分别为$v_i,u_i$,$v_i$是作为中心词时的向量，我们一般用这个作为最终的选择。$u_i$对应的就是作为背景词时的向量了。

<div align=center>
    <img src="zh-cn/img/word2vec/p19.png" /> 
</div>

**3.word2vec中训练的trick**

一般神经网络语言模型在预测的时候，输出的是预测目标词的概率，也就是说我每一次预测都要基于全部的数据集进行计算，这无疑会带来很大的时间开销。不同于其他神经网络，word2vec提出两种加快训练速度的方式，一种是**Hierarchical softmax**，另一种是**Negative Sampling**。

**（1）hierarchical softmax**

a.哈夫曼树和哈夫曼编码

哈夫曼树： 一种带权路径长度最短的二叉树，也称为最优二叉树。

<div align=center>
    <img src="zh-cn/img/word2vec/p20.jpg" /> 
</div>

带权路径长度的计算：
$$WPL=8\times 3+6\times 3+5\times 3+3\times 4+1\times 4+15\times 1$$

<div align=center>
    <img src="zh-cn/img/word2vec/p21.jpg" /> 
</div>

b. hierarchical softmax

<div align=center>
    <img src="zh-cn/img/word2vec/p22.png" /> 
</div>

和传统的神经网络输出不同的是，word2vec的hierarchical softmax结构是把输出层改成了一颗哈夫曼树，其中图中白色的叶子节点表示词汇表中所有的$|V|$个词,黑色节点表示非叶子节点,每一个叶子节点也就是每一个单词,都对应唯一的一条从根节点出发的路径。我们的目的是使的$w=w_o$这条路径的概率最大，即: $P(w=w_o|w_i)$最大,假设最后输出的条件概率是$w_2$最大，那么我只需要去更新从根结点到$w_2$这一个叶子结点的路径上面节点的向量即可，而不需要更新所有的词的出现概率，这样大大的缩小了模型训练更新的时间。

我们应该如何得到某个叶子结点的概率呢？

<div align=center>
    <img src="zh-cn/img/word2vec/p23.png" /> 
</div>

假设我们要计算$w_2$叶子节点的概率，我们需要从根节点到叶子结点计算概率的乘积。我们知道，本模型替代的只是原始模型的softmax层，因此，某个非叶子节点的值即隐藏层到输出层的结果仍然是$u_j$，我们对这个结果进行sigmoid之后，得到节点往左子树走的概率$p$，$1-p$则为往右子树走的概率。关于这棵树的训练方式比较复杂，但也是通过梯度下降等方法，这里不详述，感兴趣的可以阅读论文「word2vec Parameter Learning Explained」 

对于我们介绍的CBOW其训练过程变为：

<div align=center>
    <img src="zh-cn/img/word2vec/p24.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/word2vec/p25.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/word2vec/p26.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/word2vec/p27.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/word2vec/p28.png" /> 
</div>

**（2）Negative Sampling 负采样**

传统神经网络在训练过程中的每一步，都需要计算词库中其他词在当前的上下文环境下出现的概率值，这导致计算量十分巨大。

<div align=center>
    <img src="zh-cn/img/word2vec/p29.png" /> 
</div>

然而，对于word2vec中的特征学习，可以不需要一个完整的概率模型。CBOW和Skip-Gram模型在输出端使用的是一个 「二分类器(即Logistic Regression)」 ，来区分 「目标词和词库中其他的 k个词（也就是把目标词作为一类，其他词作为另一类）」。下面是一个CBOW模型的图示，对于Skip-Gram模型输入输出是倒置的。

<div align=center>
    <img src="zh-cn/img/word2vec/p30.png" /> 
</div>

此时，最大化的目标函数如下：

$$J_{NEC}=log Q_{\theta}(D=1|w_t,h)+kE_{\tilde{w}\sim P_{noise}}[log Q_{\theta(D=0|\tilde{w}),h}]$$

其中，$Q_{\theta}(D=1|w_t,h)$为 「二元逻辑回归」 的概率，具体为在数据集D中、输入的embedding vector θ、上下文为h的情况下词语w出现的概率；公式后半部分为k个从 [噪声数据集] 中随机选择 k个对立的词语出现概率(log形式)的期望值。可以看出，目标函数的意义是显然的，即尽可能的 [分配(assign)] 高概率给真实的目标词，而低概率给其他 k 个 [噪声词]，这种技术称为 「负采样(Negative Sampling)」

这种想法来源于 「噪声对比评估方法（NEC）」，大致思想是：假设$X=(x_1,x_2,⋯,x_{T_d})$是从真实的数据（或语料库）中抽取样本，但是样本服从什么样的分布我们不知道，那么先假设其中的每个$x_i$服从一个未知的概率密度函数$p$。这样我们需要一个相对可参考的分布反过来去估计概率密度函数$p$，这个可参考的分布或称之为噪音分布应该是我们知道的，比如高斯分布，均匀分布等。假设这个噪音分布的概率密度函数$pn$，从中抽取样本数据为$Y=(y_1,y_2,⋯,y_{T_n})$，而这个数据称之为噪声样本，我们的目的就是通过学习一个分类器把这两类样本区别开来，并能从模型中学到数据的属性，噪音对比估计的思想就是“通过比较而学习”。

具体来说，word2vec里面的负采样：将输出层的V个样本分为正例(Positive Sample)也就是目标词对应的项，以及剩余V−1个负例(Negative Samples)。举个例子有个样本phone
number，这样wI=phone,wO=number, 正例就是number这个词，负例就是不太可能与phone共同出现的词。负采样的思想是每次训练只随机取一小部分的负例使他们的概率最小，以及对应的正例概率最大。随机采样需要假定一个概率分布，word2vec中直接使用词频作为词的分布，不同的是频数上乘上0.75，相比于直接使用频次作为权重，取0.75幂的好处可以减弱不同频次差异过大带来的影响，使得小频次的单词被采样的概率变大。

$$weight(w)=\frac{count(w)^{0.75}}{\sum_{u}count(w)^{0.75}}$$

负采样定义的损失函数如下：

$$E=-log \sigma (v_{wO}^{'}h)-\sum_{wj\in W_{neg}}log \sigma(-v^{'}_{wj}h)$$

损失函数，一部分是正样本（期望输出的词），另一部分是负采样随机抽取出来的负样本集合，$V^{'}_{wO}$是输出向量


如果大家理解的还不是很深的话，接下来将通过谷歌发布的tensorflow官方word2vec代码解析加深理解。代码链接：<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py>

**4.tensorflow和gensim实现word2vec**

!> Tensorflow

**Example-0: skip-gram模型，采用负采样的trick进行训练**

```python
# 参考 https://github.com/zlsdu/Word-Embedding/blob/master/word2vec/word2vec_tensorflow.py
# 为方便读者代码的阅读，我们加入了详细的注释

# 导入必要的库
import jieba
import math
import random
import collections
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class LoadData():
    '''
    This class is created by zhanglei at 2019/06/10.
    The environment: python3.5 or later and tensorflow1.10 or later.
    The functions include load data，build vocabulary，pretrain，generate batches.
    '''
    def __init__(self):
        '''
        数据加载相关参数设置
        '''
        self.corpus_path = '../../data/wiki.zh.text.jian.part'   #wiki中文语料库文件路径，未分词的原始语料
        self.stop_words_path = '../../data/stop_words.txt'       #停用词文件路径
        self.vocabulary_size = 200000  #词典大小
        self.batch_size = 128          #batch大小
        self.num_skips = 2             #中心词使用的次数
        self.skip_window = 1           #skipgram算法窗口大小
        
    def read_data(self):
        """
        读取文本，把文本的内容的所有词放在一个列表
        self.stop_words_path:停用词文件路径
        self.corpus_path:语料库文件路径
        Returen:
            vocabularys_list = [词表]
        """
        # 读取经过预处理后的语料库数据
        lines = open(self.stop_words_path, 'r').readlines()
        stop_words = {word.strip():1 for word in lines}
        vocabularys_list = []
        wiki_zh_data = []
        with open(self.corpus_path, "r") as f:
            line = f.readline()
            while line:
                raw_words = list(jieba.cut(line.strip())) # 对读入的每行语料进行分词
                raw_words = [raw_words[i] for i in range(len(raw_words))
                             if raw_words[i] not in stop_words and raw_words[i] != ' '] # 去停用词
                vocabularys_list.extend(raw_words)  # 保存分词到列表 
                wiki_zh_data.append(raw_words)  # 将分词去停用词后的每一行作为一个列表，存入一个列表
                line = f.readline() # 继续读入下一行
        return vocabularys_list, wiki_zh_data
    
    def know_data(self, vocabularys, topk):
        '''
        查看语料库信息，包括词频数、最高词频词语
        Args:
            vocabularys: 所有经过分词去停用词后词语信息
        '''
        vocab_dict = {}
        for i in range(len(vocabularys)):
            if vocabularys[i] not in vocab_dict:
                vocab_dict[vocabularys[i]] = 0
            vocab_dict[vocabularys[i]] += 1
        vocab_list = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True)
        print('词典中词语总数:{}'.format(len(vocab_dict)))
        print('top{}词频词语信息:{}'.format(topk, vocab_list[:topk]))
        
    def build_dataset(self, vocabularys):
        '''
        对词表中词出现的个数进行统计,并且将出现的罕见的词设置成了 UNK
        Args:
            vocabularys: 词典
        Return:
            data_index_list: [[word,index], [], ...]
            word_index_dict：{word:index, ...}
            index_word_dict: {index:word, ...}
        '''
        word_count_list = [['UNK', -1]]
        word_index_dict = {}
        index_word_dict = {}
        data_index_list = []
        #extend是扩展添加list中的内容
        #collections.Counter是将数字按key统计称dict的形式,并且按照了数量进行了排序, most_common方法是将格式转化成list，并且只取参数的数量
        word_count_list.extend(collections.Counter(vocabularys).most_common(self.vocabulary_size - 1))
        for word, _ in word_count_list:
            word_index_dict[word] = len(word_index_dict)
        unk_count = 0
        for word in vocabularys:
            if word in word_index_dict:
                index = word_index_dict[word]
            else:
                index = 0  
                unk_count += 1
            data_index_list.append(index)
        word_count_list[0][1] = unk_count
        index_word_dict = dict(zip(word_index_dict.values(), word_index_dict.keys()))
        return data_index_list, word_index_dict, index_word_dict, word_count_list
    
    def generate_batch(self):
        '''
        其中collections.deque(maxlen)是python中的双向列表，通过设置maxlen则列表会自动控制大小
        当append新的元素到尾部时，便会自动将头部的元素删除，始终保持 2*skip_window+1的窗口大小
        batch_size: batch大小
        num_skips: 中心词使用的次数，中心词预测窗口中label重复使用的次数
        skip_window: 设置窗口大小，窗口大小为2*skip_window+1
        Return:
            batch_text: 一个batch文本
            batch_label: 一个batch标签
        '''
        global data_index
        assert self.batch_size % self.num_skips == 0    #断言：判断assert condition中condition是真是假
        assert self.num_skips <= 2 * self.skip_window
        batch_text = np.ndarray(shape=(self.batch_size), dtype=np.int32) # 因为我们要的是input word的index
        batch_label = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        # buffer中存储的一个窗口2*skip_window的总共数据
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data_index_list[data_index])
            data_index = (data_index + 1) % len(data_index_list)
        # num_skips代表一个中心词使用的次数，因为便需要控制num_skips和skip_window的大小关系
        # 通过skip_window拿到处于中间位置的词，然后用他去预测他周围的词,周围词选取是随机的
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch_text[i * self.num_skips + j] = buffer[self.skip_window]
                batch_label[i * self.num_skips + j, 0] = buffer[target]
            # 在一个中心词num_skips次数结束之后，便将窗口往后移动，重新加一个新的词进来
            # maxlen会自动保持窗口的总大小，都会自动往回移动一个单词
            buffer.append(data_index_list[data_index])
            data_index = (data_index + 1) % len(data_index_list)
        return batch_text, batch_label


class SkipgramModel():
    '''
    This class is created by zhanglei at 2019/06/10.
    The environment: python3.5 or later and tensorflow1.10 or later.
    The functions include set parameters，build skipgram model.
    '''
    def __init__(self, valid_examples):
        '''
        skipgram模型相关参数设置，并加载模型
        '''
        self.batch_size = 128           #batch大小
        self.vocabulary_size = 200000   #词典大小
        self.embedding_size = 256       #word embedding大小
        self.num_sampled = 32         #负采样样本的数量
        self.learning_rate = 0.5        #学习率
        self.valid_examples = valid_examples
        self.skipgram()
        
    def skipgram(self):
        '''
        skipgram模型结构
        '''
        tf.reset_default_graph()
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        #word embedding进行随机初始化
        with tf.name_scope('initial'):
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
            # tf.nn.embedding_lookup（params, ids）:params可以是张量也可以是数组等，id就是对应的索引，其他的参数不介绍。
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                            stddev=1.0 / math.sqrt(self.embedding_size)))
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]),dtype=tf.float32)

        with tf.name_scope('loss'):
            #采用nce_loss损失函数，并进行负采样
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases, 
                                             inputs=self.embed, 
                                             labels=self.train_labels,
                                             num_sampled=self.num_sampled, 
                                             num_classes=self.vocabulary_size))
        #使用梯度下降优化算法
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        #验证集，筛选与验证集词向量相似度高的词向量
        with tf.name_scope('valid'):
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, valid_dataset)
            self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)


def train_model(loadData, skipgram):
    '''
    启动tf.Session()加载模型进行训练
    '''
    num_steps = 100000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = loadData.generate_batch()
            feed_dict = {skipgram.train_inputs: batch_inputs, skipgram.train_labels: batch_labels}
            _, loss_val = sess.run([skipgram.optimizer, skipgram.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 100 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step {} : {}".format(step, average_loss))
                average_loss = 0

            if step % 1000 == 0:
                sim = skipgram.similarity.eval()
                for i in range(len(skipgram.valid_examples)):
                    valid_word = index_word_dict[skipgram.valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[:top_k]
                    log_str = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = index_word_dict[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = skipgram.normalized_embeddings.eval() 


if __name__ == '__main__':
    #加载数据类
    loadData = LoadData()
    #读取语料库原文件，得到词表信息
    vocabularys, wiki_zh_data = loadData.read_data()
    loadData.know_data(vocabularys, topk=10)
    data_index_list, word_index_dict, index_word_dict, word_count_list = loadData.build_dataset(vocabularys)
    valid_word = ['中国', '学院', '中心', '北京', '大学', '爱', "不错", "中文", "幸福"]  #验证集
    valid_examples =[word_index_dict[li] for li in valid_word]    #验证机index
    global data_index
    data_index = 0
    #加载skipgram模型
    skipgram = SkipgramModel(valid_examples)
    #进行模型训练，最终得到word2vec模型副产物word embedding
    train_model(loadData, skipgram)

```


!> gensim

**Example-0: 中文语料的处理**

基于gensim库的word2vec需要一系列的句子作为输入，其中每个语句都是一个词汇列表（经过分词处理）,如果是句子，需要进行分词;
如果是文件，需要将文件处理为每一行对应一个句子（已经分词，以空格隔开)。

```python
# 法一：预料处理文列表
# 缺点： 把Python内置列表当作输入很方便，但当输入量很大的时候，大会占用大量内存。
from gensim.models import Word2Vec
sentences = [["Python", "深度学习", "机器学习"], ["NLP", "深度学习", "机器学习"]]
model = Word2Vec(sentences, min_count=1)
```
```python
# 法二： 语料是文件（迭代器）
# 一般我们的语料是在文件中存放的，首先，需要保证语料文件内部每一行对应一个句子（已经分词，以空格隔开）
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
 
sentences = MySentences('/some/directory') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)

```
```python
# 法三：

#作用同下一个类，对一个目录下的所有文件生效，对子目录无效
#该路径下的文件 只有后缀为bz2，gz和text的文件可以被读取，其他的文件都会被认为是text文件
#一个句子即一行，单词需要预先使用空格分隔
#源处填写的必须是一个目录，务必保证该目录下的文件都能被该类读取。如果设置了读取限制，那么只读取限定的行数。
# gensim.models.word2vec.PathLineSentences(source, max_sentence_length=10000, limit=None)
sentences = PathLineSentences(path)
```

```python
# 法四：对于单个文件语料
# 每一行对应一个句子（已经分词，以空格隔开），我们可以直接用LineSentence把txt文件转为所需要的格式
from gensim import Word2Vec
from gensim.Word2Vec import LineSentence
from gensim.test.utils import common_texts, get_tmpfile
 
# inp为输入语料
inp = 'wiki.zh.text.jian.seg.txt' # 处理好的语料
sentences = LineSentences(inp)
path = get_tmpfile("word2vec.model") #创建临时文件
model = Word2Vec(sentences, size=100, window=5, min_count=1)
model.save("word2vec.model")

```
```python
# 法五： 从语料库获取文件

# 从一个叫‘text8’的语料库中获取数据，该语料来源于以下网址，参数max_sentence_length限定了获取的语料长度
# http://mattmahoney.net/dc/text8.zip
gensim.models.word2vec.Text8Corpus(fname="test8.zip", max_sentence_length=10000)
```

**Example-1: 一个简单的训练的例子**
```python
# 导入模块并设置日志记录
import gensim, logging
from pprint import pprint
from smart_open import smart_open
import os
import jieba

# jieba.load_userdict(r"G:\chinese-opinion-target-extraction-master\dataset\dictionary\dictionary.txt")  
#载入自定义词典，提高分词的准确性,因为数据量较大，需要花点时间加载
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

```python
# 未分词的语料
sentences = [
    
'2018年10月份，麻省理工学院的Zakaria el Hjouji, D. Scott Hunter等学者发表了《The Impact of Bots on Opinions in Social Networks》，',
'该研究通过分析 Twitter 上的机器人在舆论事件中的表现，证实了社交网络机器人可以对社交网络舆论产生很大的影响，不到消费者总数1%的活跃机器人，就可能左右整个舆论风向。',
'麻省理工学院研究组的这项工作，最大的发现是，影响社交网络舆论所需要的机器人，其实是很少的。少数活跃的机器人，可以对网络舆论产生重大影响。',
'机器人检测算法，会判断某用户是机器人的概率，但实际操作中研究者发现，该算法把几个经常转发但不常被@ 的真实用户也当做了机器人。所以研究者对有实名认证的 Twitter 用户做了筛查，把他们都归为真实用户。',
'不论是真实的推特用户还是推特机器人，它们的三个基本操作是，关注、转发和评论（类似微博）。通过跟踪这些互动，研究者可以有效量化 Twitter 账号的行为。',
'直觉上，那些不太关注别人的用户，也不太可能关注你。而且社交圈子重叠很重要，如果 A 和 B 是好友，那么关注 A 的用户就有较大概率关注 B。',
'虽然人们在收到新信息时会更新他们的观点，但这个过程会随着时间的推移而减弱，他们的观点会逐渐变得顽固。Zaman 认为，如果你已经掌握了很多信息，你就越来越难听从别人的观点，新说法不会改变你的看法。',
'该研究团队基于过往研究，给出了网络舆论模型的核心假设：',
'社交网络中的个人是基于其朋友推文中的观点，来更新自身的观点；',
'网络中某些用户的观点是顽固的，其观点不会轻易改变，而且顽固用户会推动其他用户（摇摆不定的中间派）改变观点。',
'虽然社交媒体机器人不会带来物理威胁，但它们却可能有力影响到网络舆论。在微博里，各类水军已经经常出现在营销造势、危机公关中。虽然你能一眼识别出谁是水军，但仍然可能不知不觉地被他们影响。',
'这些机器人看似僵尸，发起声来，比人类响亮得多，可能只要几十个几百个就足够扭转舆论！',
'所以，从社会化媒体数据挖掘的角度来看，信息的真实性并不重要，只要文章、帖子或者评论能影响到浏览者或受众，具有一定的（潜在）影响力，这类社媒数据数据就值得去挖掘。',
'更进一步说，跟销售数据反映消费者决策价值、搜索数据反映消费者意图价值相比，虽然社会化媒体文本数据的价值密度最低，好比是蕴藏金子和硅、却提炼极为困难的沙子，但由于它在互联网领域的分布极为广泛，',
'且蕴含着对客观世界的细节描述和主观世界的宣泄（情绪、动机、心理等），其最大价值在于潜移默化地操控人的思想和行为的影响力，',
'通过社会化媒体挖掘，我们可以得到对目标受众具有（潜在）影响力的商业情报。淘沙得金，排沙简金，最终得到的分析结果用以预判受众的思考和行为，为我们的生产实践服务。'
          ]
```

```python
data_cut = [jieba.lcut(i) for i  in sentences] 
#对语句进行分词
data_cut = [' '.join(jieba.lcut(i)) for i  in sentences]
#用空格隔开词汇，形成字符串，便于后续的处理
stoplist = [i.strip() for i in open('datasets/stopwords_zh.txt',encoding='utf-8').readlines()]  
#载入停用词列表
sentences = [[word for word in document.strip().split() if word not in stoplist] for document in data_cut]   
#过滤语句中的停用词sentences[:3]  #展示预处理后语句列表中的3个样例
```

```python
# 在这些语句上训练word2vec模型
model = gensim.models.Word2Vec(sentences,size=50,min_count=1,iter=20)
```

```python
#做一个简单的相似词检索操作，可能是训练语料太少的缘故，得到的结果没有太make sense
model.wv.most_similar('社交网络')  
```

**Example-2:**

```python
import collections
from gensim.models import word2vec
from gensim.models import KeyedVectors


def stat_words(file_path, freq_path):
    '''
    统计词频保存到文件，了解数据集基本特征 
    Args:
        file_path: 语料库文件路径
        freq_path: 词频文件保存路径
    Retrun:
        word_list = [[word:count],...]
    '''
    fr = open(file_path, 'r') #从语料库文件中读取数据并统计词频
    lines = fr.readlines()
    text = [line.strip().split(' ') for line in lines]
    fr.close()
    word_counts = collections.Counter()  # 统计词频常用的方法
    for content in text:
        word_counts.update(content)
    word_freq_list = sorted(word_counts.most_common(), key=lambda x:x[1], reverse=True)
    fw = open(freq_path, 'w') #将词频数据保存到文件
    for i in range(len(word_freq_list)):
        content = ' '.join(str(word_freq_list[i][j]) for j in range(len(word_freq_list[i])))
        fw.write(content + '\n')
    fw.close()
    return word_freq_list


def get_word_embedding(input_corpus, model_path):
    '''
    利用gensim库生成语料库word embedding
    Args:
        input_corpus: 语料库文件路径
        model_patht: 预训练word embedding文件保存路径
    '''
    sentences = word2vec.Text8Corpus(input_corpus)  # 加载语料
    #常用参数介绍: size词向量维度、window滑动窗口大小上下文最大距离、min_count最小词频数、iter随机梯度下降迭代最小次数   
    model = word2vec.Word2Vec(sentences, size=100, window=8, min_count=3, iter=8)
    model.save(model_path)
    model.wv.save_word2vec_format(model_path, binary=False)


def load_pretrain_model(model_path):
    '''
    加载word2vec预训练word embedding文件
    Args:
        model_path: word embedding文件保存路径
    '''
    model = KeyedVectors.load_word2vec_format(model_path)
    print('similarity(不错，优秀) = {}'.format(model.similarity("不错", "优秀")))
    print('similarity(不错，糟糕) = {}'.format(model.similarity("不错", "糟糕")))
    most_sim = model.most_similar("不错", topn=10)
    print('The top10 of 不错: {}'.format(most_sim))
    words = model.vocab


if __name__ == '__main__':
    corpus_path = '../data/toutiao_word_corpus.txt' #中文预料文件路径
    freq_path = '../data/words_freq_info.txt' #词频文件保存路径
    word_list = stat_words(corpus_path, freq_path) #统计保存预料中词频信息并保存
    model_path = 'toutiao_word_embedding.bin' #训练词向量文件保存路径
    get_word_embedding(corpus_path, model_path) #训练得到预料的词向量
    load_pretrain_model(model_path) #加载预训练得到的词向量
```