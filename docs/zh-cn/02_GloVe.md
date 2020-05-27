## GloVe

<!-- http://www.fanyeong.com/2018/02/19/glove-in-detail/ -->
<!-- http://www.fanyeong.com/2017/10/10/word2vec/ -->
<!-- https://blog.csdn.net/mr_tyting/article/details/80180780 -->

<!-- https://blog.csdn.net/linchuhai/article/details/97135612 -->

<!-- https://blog.csdn.net/buchidanhuang/article/details/98471741?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-14.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-14.nonecase -->

<!-- https://blog.csdn.net/qq_35290785/article/details/98231826 -->
<!-- https://blog.csdn.net/hao5335156/article/details/80452793 -->

<!-- https://github.com/maciejkula/glove-python -->
<!-- https://github.com/Daviddddl/glove-tools -->
<!-- https://github.com/JonathanRaiman/glove -->

**1. 共现矩阵**

!> 一切要从word representation说起，什么是Word Representation？

对人来说一个单词就是一个单词，但是对计算机来说却不是这样，那么计算机是如何处理单词或者文本的呢？最简单最直观的做法就是把单词（word）按照某种规则表达成一个向量（vector），这就是Word Representation。

!> 什么是one-hot encoding？

比如：假设我们有这样的两个文本：

D1: I like green
D2: I like red
那么针对这两个文本所组成的语料库而言，我们会得到如下所示的字典：[green, I, like, red]，那么单词”I”的one-hot encoding就是[0，1，0，0]，单词”like”则是[0，0，1，0]。

!> 什么是Word Embedding？

要理解这个概念，先理解什么是Emdedding？Embedding在数学上表示一个maping, f: X -> Y， 也就是一个function，其中该函数是injective（就是我们所说的单射函数，每个Y只有唯一的X对应，反之亦然）和structure-preserving (结构保存，比如在X所属的空间上X1 < X2,那么映射后在Y所属空间上同理 Y1 < Y2)。 那么对于word embedding，就是将单词word映射到另外一个空间，其中这个映射具有injective和structure-preserving的特点。 通俗的翻译可以认为是单词嵌入，就是把X所属空间的单词映射为到Y空间的多维向量，那么该多维向量相当于嵌入到Y所属空间中，一个萝卜一个坑。word embedding，就是找到一个映射或者函数，生成在一个新的空间上的表达，该表达就是word representation。

!> 有哪些类型的Word Embeddings？

目前主要分为两类：

+ Frequency based Embedding
+ Prediction based Embedding

Frequency based Embedding就是基于词频统计的映射方式，主要有以下三种：

+ Count Vector

这种就是最简单，最基本的词频统计算法：比如我们有N个文本（document），我们统计出所有文本中不同单词的数量，结果组成一个矩阵。那么每一列就是一个向量，表示这个单词在不同的文档中出现的次数。

+ TF-IDF Vector

TF-IDF方法基于前者的算法进行了一些改进，它的计算公式如下: $TF-IDF_{i,j}=TF_{i,j} \times IDF_{i}$, 其中，$TF_{i,j}$（term-frequence）指的是第
i个单词在第j个文档中出现的频次；而$IFD_{i}$(inverse document frequency)的计算公式如下：$IDF_{i}=\log(N/n)$,

其中，$N$表示文档的总个数，$n$表示包含该单词的文档的数量。这个公式是什么意思呢？其实就是一个权重，设想一下如果一个单词在各个文档里都出现过，那么$\frac{N}{n}=1$
所以$IDF_{i}=0$。这就意味着这个单词并不重要。这个东西其实很简单，就是在term-frequency的基础上加了一个权重，从而显著降低一些不重要/无意义的单词的frequency，比如a,an,the等。

+ Co-Occurrence Vector
这个比较有意思，中文直译过来就是协同出现向量(共现向量）。在解释这个概念之前，我们先定义两个变量：
    - Co-occurrence: 协同出现指的是两个单词w1和w2在一个Context Window范围内共同出现的次数
    - Context Window: 指的是某个单词w的上下文范围的大小，也就是前后多少个单词以内的才算是上下文？比如一个Context Window Size = 2的示意图如下：
<div align=center>
    <img src="zh-cn/img/glove/p1.png" /> 
</div>

比如我们有如下的语料库：

> He is not lazy. He is intelligent. He is smart.

我们假设Context Window=2，那么我们就可以得到如下的co-occurrence matrix：

<div align=center>
    <img src="zh-cn/img/glove/p2.png" /> 
</div>
这个方法比之前两个都要进步一点，为什么呢？ 因为它不再认为单词是独立的，而考虑了这个单词所在附近的上下文，这是一个很大的突破。 如果两个单词经常出现在同一个上下文中，那么很可能他们有相同的含义。比如vodka和brandy可能经常出现在wine的上下文里，那么在这两个单词相对于wine的co-occurrence就应该是相近的，于是我们就可以认为这两个单词的含义是相近的。共现矩阵不足： 面临稀疏性问题、向量维数随着词典大小线性增长,解决：SVD、PCA降维，但是计算量大。


**2. GloVe**

2014年Stanford NLP团队发表的GloVe
<!-- ,2017年由Facebook团队发表的fastText. -->

GloVe使用了词与词的共现（co-occurrence)信息。我们定义$X$为共现词频矩阵，其中元素$x_{ij}$为词$j$出现在词$i$的环境(context)的次数。这里的“环境”有很多种可能的定义。举个例子，在一段文本序列中，如果词$j$出现在词$i$左边或者右边不超过10个词的距离，我们认为词$j$出现在词$i$的环境一次。令$x_i=\sum_{k}x_{ik}$为任意词出现在词$i$的环境中的次数，那么，
$$P_{ij}=P(j|i)=\frac{x_{ij}}{x_i}$$
为词$j$出现在词$i$的环境的概率。这一概率也称为词$i$和词$j$的共现概率

在介绍GloVe的原理之前，先来看论文中的一个案例，假设i=ice,j=steam，并对k取不同的词汇，如“solid”，“gas”，“water”，“fashion”，根据上面的定义，我们分别计算他们的概率$P(k∣ice)$、$P(k∣steam)$，并计算两者的比率$P(k∣ice)/P(k∣steam)$，可以发现，对于“solid”，其出现在“ice”上下文的概率应该比较大，出现在“steam”上下文的概率应该比较小，因此，他们的比值应该是一个比较大的数，在下表中是8.9，而对于“gas”，出现在“ice”上下文的概率应该比较小，而出现在“steam”上下文的概率应该比较大，因此，两者的比值应该是一个比较小的数，在下表中是$8.5 \times 10^{-2}$
 ，而对于“water、fashion”这两个词汇，他们与“ice”和steam“的相关性应该比较小，因此，他们的比值应该都是接近1。因此，这样来看可以发现，比值$P(k∣ice)/P(k∣steam)$在一定程度上可以反映词汇之间的相关性，当相关性比较低时，其值应该在1附近，当相关性比较高时，比值应该偏离1比较远。

<div align=center>
    <img src="zh-cn/img/glove/p3.png" /> 
</div>


基于这样的思想，作者提出了这样一种猜想，能不能通过训练词向量，使得词向量经过某种函数计算之后可以得到上面的比值，具体如下：
$$F(w_i,w_j,\tilde{w}_ {k})=\frac{P_{ik}}{P_{jk}}$$

其中$w_i,w_j,\tilde{w_k}$为词汇i,j,k对应的词向量，其维度为d,而$\frac{P_{ik}}{P_{jk}}$则可以直接通过语料的共现矩阵计算得到，这里的$F$胃一个未知的函数。由于词向量都是在一个线性向量空间，因此，可以对$w_i,w_j$进行差分，将其转化为：
$$F(w_i-w_j,\tilde{w}_ {k})=\frac{P_{ik}}{P_{jk}}$$

由于上式中左侧括号中是两个维度为d的词向量，而右侧是一个标量，因此，很容易会想到向量的内积，因此，上式可以进一步改变为：
$$F((w_i-w_j)^T\tilde{w}_ {k})=\frac{P_{ik}}{P_{jk}}$$

由于上式中左侧是一种减法，而右侧是一种除法，很容易联想到指数计算，因此，可以把$F$限定为指数函数，此时有：
$$\exp((w_i-w_j)^T\tilde{w}_ {k})=\frac{P_{ik}}{P_{jk}}$$

因此，此时只要确保等式两边分子分母相等即可，即
$$\exp(w_i^T\tilde{w}_ {k})=P_{ik},\exp(w_j^T\tilde{w}_ {k})=P_{jk}$$
化简之后得到：
$$w_i^Tw_k=log(X_{ik})-log(X_i)$$
由于上式左侧$w_{i}^{T} w_{k}$中，调换i和k的值不会改变其结果，即具有对称性，因此，为了确保等式右侧也具备对称性，引入了两个偏置项，即
$$w_i^Tw_k=log(X_{ik})-b_i-b_k$$

此时，$\log X_{i}$已经包含在$b_{i}$当中。因此，此时模型的目标就转化为通过学习词向量的表示，使得上式两边尽量接近，因此，可以通过计算两者之间的平方差来作为目标函数，即：
$$J=\sum_{i,k=1}^{V}(w_i^T\tilde{W}_ {k}+b_i+b_k-\log(X_{ik}))^2$$

但是这样的目标函数有一个缺点，就是对所有的共现词汇都是采用同样的权重，因此，作者对目标函数进行了进一步的修正，通过语料中的词汇共现统计信息来改变他们在目标函数中的权重，具体如下：
$$J=\sum_{i,k=1}^{V}f(X_{ik})(w_i^T\tilde{W}_ {k}+b_i+b_k-\log(X_{ik}))^2$$

这里V表示词汇的数量，并且权重函数f必须具备以下的特性：

+ $f(0)=0$，当词汇共现的次数为0时，此时对应的权重应该为0。
+ $f(x)$必须是一个非减函数，这样才能保证当词汇共现的次数越大时，其权重不会出现下降的情况。
+ 对于那些太频繁的词，$f(x)$应该能给予他们一个相对小的数值，这样才不会出现过度加权。

综合以上三点特性，作者提出了下面的权重函数：
<!-- $$\begin{equation}
f(x)=\left\{
\begin{array}{rl}
(x/x_{max})^{\alpha} && if\quad x< x_{max}\\
1 && otherwise
\end{array}
\right.
\end{equation}$$
 -->

<div align=center>
    <img src="zh-cn/img/glove/p4.png" /> 
</div>

作者在实验中设定$x_{\max }=100$，并且发现$\alpha=3/4$时效果比较好。函数的图像如下图所示：

<div align=center>
    <img src="zh-cn/img/glove/p5.png" /> 
</div>


!> GloVe是如何训练的？

虽然很多人声称GloVe是一种无监督（unsupervised learing）的学习方式（因为它确实不需要人工标注label），但其实它还是有label的，这个label就是上述公式中的
$\log(X_{ij})$，而公式中的向量$w$和$\tilde{w}$就是要不断更新学习的参数，所以本质上它的训练方式跟监督学习的训练方法没什么不一样，都是基于梯度下降的。具体地，这篇论文里的实验是这么做的：采用了AdaGrad的梯度下降算法，对矩阵X中的所有非零元素进行随机采样，学习曲率（learning rate）设为0.05，在vector size小于300的情况下迭代了50次，其他大小的vectors上迭代了100次，直至收敛。最终学习得到的是两个vector是$w$和$\tilde{w}$,因为X是对称的（symmetric），所以从原理上讲
$w$和$\tilde{w}$是也是对称的，他们唯一的区别是初始化的值不一样，而导致最终的值不一样。所以这两者其实是等价的，都可以当成最终的结果来使用。但是为了提高鲁棒性，我们最终会选择两者之和$w+\tilde{w}$作为最终的vector（两者的初始化不同相当于加了不同的随机噪声，所以能提高鲁棒性）。在训练了400亿个token组成的语料后，得到的实验结果如下图所示：

<div align=center>
    <img src="zh-cn/img/glove/p6.png" /> 
</div>

这个图一共采用了三个指标：语义准确度，语法准确度以及总体准确度。那么我们不难发现Vector Dimension在300时能达到最佳，而context Windows size大致在6到10之间。

**3. Glove与LSA、word2vec的比较**

LSA（Latent Semantic Analysis）是一种比较早的count-based的词向量表征工具(我们将在后续章节详细介绍该方法），它也是基于co-occurance matrix的，只不过采用了基于奇异值分解（SVD）的矩阵分解技术对大矩阵进行降维，而我们知道SVD的复杂度是很高的，所以它的计算代价比较大。还有一点是它对所有单词的统计权重都是一致的。而这些缺点在GloVe中被一一克服了。而word2vec最大的缺点则是没有充分利用所有的语料，所以GloVe其实是把两者的优点结合了起来。从这篇论文给出的实验结果来看，GloVe的性能是远超LSA和word2vec的，但网上也有人说GloVe和word2vec实际表现其实差不多。

**4. 总结**

以上就是有关GloVe原理的介绍，作者其实也是基于最开始的猜想一步一步简化模型的计算目标，最后看GloVe的目标函数时发现其实不难计算，但是要从最开始就想到这样一个目标函数其实还是很难的。最后做一下总结：

+ Glove综合了全局词汇共现的统计信息和局部窗口上下文方法的优点，可以说是两个主流方法的一种综合，但是相比于全局矩阵分解方法，由于GloVe不需要计算那些共现次数为0的词汇，因此，可以极大的减少计算量和数据的存储空间。
+ 但是GloVe把语料中的词频共现次数作为词向量学习逼近的目标，当语料比较少时，有些词汇共现的次数可能比较少，笔者觉得可能会出现一种误导词向量训练方向的现象。



**5. 代码实现glove**

官方实现代码： [stanfordnlp](https://github.com/stanfordnlp/GloVe)<https://github.com/stanfordnlp/GloVe>

在网上有一个几个比较好的python库，感兴趣的读者可以自行在GitHub搜索，本教程使用了[JonathanRaiman](https://github.com/JonathanRaiman/glove)的python库进行实现。

```python
# pip install glove

import glove

cooccur = {
    0: {
        0: 1.0,
        2: 3.5
    },
    1: {
        2: 0.5
    },
    2: {
        0: 3.5,
        1: 0.5,
        2: 1.2
    }
}

model = glove.Glove(cooccur, d=50, alpha=0.75, x_max=100.0)

for epoch in range(25):
    err = model.train(batch_size=200, workers=9, batch_size=50)
    print("epoch %d, error %.3f" % (epoch, err), flush=True)
# The trained embeddings are now present under model.W.
```

```
Glove.init()中的参数

cooccurence dict<int, dict<int, float>> : the co-occurence matrix
alpha float : (default 0.75) hyperparameter for controlling the exponent for normalized co-occurence counts.
x_max float : (default 100.0) hyperparameter for controlling smoothing for common items in co-occurence matrix.
d int : (default 50) how many embedding dimensions for learnt vectors
seed int : (default 1234) the random seed

Glove.train() 中的参数

step_size float : the learning rate for the model
workers int : number of worker threads used for training
batch_size int : how many examples should each thread receive (controls the size of the job queue)
```