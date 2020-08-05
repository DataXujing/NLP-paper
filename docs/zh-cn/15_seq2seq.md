## Seq2Seq (sequence to sequence)

Seq2Seq技术，全称Sequence to Sequence，该技术突破了传统的固定大小输入问题框架，开通了将经典深度神经网络模型（DNNs）运用于在翻译，文本自动摘要和机器人自动问答以及一些回归预测任务上,并被证实在英语－法语翻译、英语－德语翻译以及人机短问快答的应用中有着不俗的表现。Seq2Seq被提出于2014年，最早由两篇文章独立地阐述了它主要思想，分别是Google Brain团队的《Sequence to Sequence Learning with Neural Networks》和Yoshua Bengio团队的《Learning Phrase Representation using RNN Encoder-Decoder for Statistical Machine Translation》。这两篇文章针对机器翻译的问题不谋而合地提出了相似的解决思路，Seq2Seq由此产生。



<!-- https://blog.csdn.net/u010555997/article/details/76474533

https://blog.csdn.net/zhu_zhu_zhu_zhu_zhu/article/details/94637991

https://www.jianshu.com/p/80436483b13b

https://blog.csdn.net/irving_zhang/article/details/78889364?utm_source=gold_browser_extension

https://blog.csdn.net/qq_30219017/article/details/89090690

https://blog.csdn.net/xyz1584172808/article/details/89220906

https://zhuanlan.zhihu.com/p/51383402
 -->


### 1.RNN的基本结构及使用


<div align=center>
    <img src="zh-cn/img/seq2seq/p1.png" /> 
</div>

针对不同任务，通常要对 RNN 模型结构进行少量的调整，根据输入和输出的数量，分为三种比较常见的结构：`N vs N`、`1 vs N`、`N vs 1`。

<div align=center>
    <img src="zh-cn/img/seq2seq/p2.png" /> 
</div>

*N vs N*

上图是RNN 模型的一种 `N vs N` 结构，包含 `N` 个输入$x_1, x_2, ..., x_N$，和 `N` 个输出 $y_1, y_2, ..., y_N$。`N vs N`的结构中，输入和输出序列的长度是相等的，通常适合用于以下任务：

+ 词性标注
+ 训练语言模型，使用之前的词预测下一个词等


<div align=center>
    <img src="zh-cn/img/seq2seq/p3.png" /> 
</div>

*1 vs N (1)*

<div align=center>
    <img src="zh-cn/img/seq2seq/p4.png" /> 
</div>

*1 vs N (2)*

在 `1 vs N` 结构中，我们只有一个输入 `x`，和 `N` 个输出 $y_1, y_2, ..., y_N$。可以有两种方式使用 `1 vs N`，第一种只将输入 `x` 传入第一个 RNN 神经元，第二种是将输入 `x` 传入所有的 RNN 神经元。`1 vs N` 结构适合用于以下任务：

+ 图像生成文字，输入 `x` 就是一张图片，输出就是一段图片的描述文字。
+ 根据音乐类别，生成对应的音乐。
+ 根据小说类别，生成相应的小说。

<div align=center>
    <img src="zh-cn/img/seq2seq/p5.png" /> 
</div>

*N vs 1*

在 `N vs 1` 结构中，我们有 `N` 个输入 $x_1, x_2, ..., x_N$，和一个输出 `y`。`N vs 1` 结构适合用于以下任务：

+ 序列分类任务，一段语音、一段文字的类别，句子的情感分析。
+ 文本分类

### 2.Seq2Seq结构

Seq2Seq 是一种重要的 RNN 模型，也称为 Encoder-Decoder 模型，可以理解为一种 N×M 的模型。模型包含两个部分：Encoder 用于编码序列的信息，将任意长度的序列信息编码到一个向量 c 里。而 Decoder 是解码器，解码器得到上下文信息向量 c 之后可以将信息解码，并输出为序列。Seq2Seq 模型结构有很多种，下面是几种比较常见的：

第一种

<div align=center>
    <img src="zh-cn/img/seq2seq/p6.png" /> 
</div>


第二种

<div align=center>
    <img src="zh-cn/img/seq2seq/p7.png" /> 
</div>

第三种

<div align=center>
    <img src="zh-cn/img/seq2seq/p8.png" /> 
</div>

**编码器 Encoder**

这三种 Seq2Seq 模型的主要区别在于 Decoder，他们的 Encoder 都是一样的。下图是 Encoder 部分，Encoder 的 RNN 接受输入 `x`，最终输出一个编码所有信息的上下文向量 `c`(context)，中间的神经元没有输出。Decoder 主要传入的是上下文向量 `c`，然后解码出需要的信息。

<div align=center>
    <img src="zh-cn/img/seq2seq/p9.png" /> 
</div>

从上图可以看到，Encoder 与一般的 RNN 区别不大，只是中间神经元没有输出。其中的上下文向量 `c` 可以采用多种方式进行计算。

<div align=center>
    <img src="zh-cn/img/seq2seq/p10.png" /> 
</div>

从公式可以看到，`c` 可以直接使用最后一个神经元的隐藏状态 $h_N$ 表示；也可以在最后一个神经元的隐藏状态上进行某种变换 $h_N$ 而得到，$q$函数表示某种变换；也可以使用所有神经元的隐藏状态 $h_1, h_2, ..., h_N$ 计算得到。得到上下文向量 `c` 之后，需要传递到 Decoder。

**解码器 Decoder**

Decoder 有多种不同的结构，这里主要介绍三种。

第一种

<div align=center>
    <img src="zh-cn/img/seq2seq/p11.png" /> 
</div>

第一种 Decoder 结构比较简单，将上下文向量 `c` 当成是 RNN 的初始隐藏状态，输入到 RNN 中，后续只接受上一个神经元的隐藏层状态 $h^{'}$而不接收其他的输入 `x`。第一种 Decoder 结构的隐藏层及输出的计算公式：

<div align=center>
    <img src="zh-cn/img/seq2seq/p12.png" /> 
</div>

第二种

<div align=center>
    <img src="zh-cn/img/seq2seq/p13.png" /> 
</div>

第二种 Decoder 结构有了自己的初始隐藏层状态 $h^{'}_ 0$，不再把上下文向量 `c` 当成是 RNN 的初始隐藏状态，而是当成 RNN 每一个神经元的输入。可以看到在 Decoder 的每一个神经元都拥有相同的输入 `c`，这种 Decoder 的隐藏层及输出计算公式：

<div align=center>
    <img src="zh-cn/img/seq2seq/p14.png" /> 
</div>


第三种

<div align=center>
    <img src="zh-cn/img/seq2seq/p15.png" /> 
</div>

第三种 Decoder 结构和第二种类似，但是在输入的部分多了上一个神经元的输出 $y^{'}$。即每一个神经元的输入包括：上一个神经元的隐藏层向量 $h^{'}$，上一个神经元的输出 $y^{'}$，当前的输入 `c` (Encoder 编码的上下文向量)。对于第一个神经元的输入 $y^{'}_ 0$，通常是句子其实标志位的 embedding 向量。第三种 Decoder 的隐藏层及输出计算公式：

<div align=center>
    <img src="zh-cn/img/seq2seq/p16.png" /> 
</div>



### 3.seq2seq模型详解

<!-- https://blog.csdn.net/irving_zhang/article/details/78889364?utm_source=gold_browser_extension

https://zhuanlan.zhihu.com/p/51383402 -->

seq2seq最早被用于机器翻译，后来成功扩展到多种自然语言生成任务，如文本摘要和图像标题的生成。这一节将介绍几种常见的seq2seq的模型原理，seq2seq的变形。

我们使用$x={x_1，x_2,...,x_n}$代表输入的语句，$y={y_1, y_2,..., y_n}$代表输出的语句，$y_t$代表当前输出词。在理解seq2seq的过程中，我们要牢记我们的目标(语言模型)是：


<div align=center>
    <img src="zh-cn/img/seq2seq/p17.png" /> 
</div>

即输出的$y_t$不仅依赖之前的输出${y_1, y_2,..., y_{t−1}}$，还依赖输入语句$x$，模型再怎么变化都是在上述公式的约束之下。


seq2seq最初模型，最早由Bengio等人发表在Computer Science上的论文：Learning Phrase Representations using RNN Encoder–Decoder 
for Statistical Machine Translation。对于RNN来说，$x={x_1，x_2,...,x_t}$代表输入，在每个时间步`t`，RNN的隐藏状态$h_t$由下述更新:

$$h_t=f(h_{t−1},x_t)$$

其中，`f`代表一个非线性函数。这时$h_t$就是一个`rnn_size`的隐含状态。然后需要通过一个矩阵`W`将其转成一个`symbol_size`的输出，并通过`softmax`函数将其转化为概率，然后筛选出概率最大的`symbol`为输出`symbol`。


<div align=center>
    <img src="zh-cn/img/seq2seq/p18.png" /> 
</div>


以上是RNN的基本原理，接下来介绍论文中的seq2seq模型： 

<div align=center>
    <img src="zh-cn/img/seq2seq/p19.png" /> 
</div>


模型包括encoder和decoder两个部分。首先在encoder部分，将输入传到encoder部分，得到最后一个时间步长`t`的隐藏状态`C`(context)，这就是RNNcell的基本功能。其次是decoder部分，从上述模型的箭头中可以看出，decoder的隐藏状态$h_t$就由$h_{t−1}$，$y_{t−1}$和$C$三部分构成。即： 


<div align=center>
    <img src="zh-cn/img/seq2seq/p20.png" /> 
</div>

由此我们得到了decoder的隐藏状态，那么最后的输出$y_t$从图中也可以看得出来由三部分得到，$h_{t}$，$y_{t-1}$和$C$，即： 

<div align=center>
    <img src="zh-cn/img/seq2seq/p21.png" /> 
</div>


到现在为止，我们就实现了我们的目标.

seq2seq的改进模型,改进模型介绍2014年发表的论文Sequence to Sequence Learning with Neural Networks。模型图： 

<div align=center>
    <img src="zh-cn/img/seq2seq/p22.png" /> 
</div>

可以看到，该模型和第一个模型主要的区别在于从输入到输出有一条完整的流：ABC为encoder的输入，WXYZ为decoder的输入。将encoder最后得到的隐藏层的状态$h_t$输入到decoder的第一个cell里，就不用像第一个模型一样，而每一个decoder的cell都需要$h_t$，因此从整体上看，从输入到输出像是一条“线性的数据流”。本文的论文也提出来，ABC翻译为XYZ，将encoder的input变为“CBA”效果更好。即A和X的距离更近了，更有利于seq2seq模型的交流。

具体来说，encoder的过程如下图。这和我们之前的encoder都一样。 


<div align=center>
    <img src="zh-cn/img/seq2seq/p23.png" /> 
</div>

不同的是decoder的阶段：


<div align=center>
    <img src="zh-cn/img/seq2seq/p24.png" /> 
</div>


得到了encoder represention，即encoder的最后一个时间步长的隐层$h_t$以后，输入到decoder的第一个cell里，然后通过一个激活函数和softmax层，得到候选的symbols，筛选出概率最大的symbol，然后作为下一个时间步长的输入，传到cell中。这样，我们就得到了我们的目标.


### 4.训练集测试的一些trick的使用

<!-- https://www.jianshu.com/p/80436483b13b

https://blog.csdn.net/qq_30219017/article/details/89090690

https://blog.csdn.net/xyz1584172808/article/details/89220906 -->


**1.Attention**

关于attention本节不是我们讨论的重点，读者将在attention机制这一节，详细的学习attetion机制。

在 Seq2Seq 模型，Encoder 总是将源句子的所有信息编码到一个固定长度的上下文向量 `c` 中，然后在 Decoder 解码的过程中向量 `c`都是不变的。这存在着不少缺陷：

+ 对于比较长的句子，很难用一个定长的向量 `c` 完全表示其意义。
+ RNN 存在长序列梯度消失的问题，只使用最后一个神经元得到的向量 `c` 效果不理想。
+ 与人类的注意力方式不同，即人类在阅读文章的时候，会把注意力放在当前的句子上。

Attention 即`注意力机制`，是一种将模型的注意力放在当前翻译单词上的一种机制。例如翻译 "I have a cat"，翻译到 "我" 时，要将注意力放在源句子的 "I" 上，翻译到 "猫" 时要将注意力放在源句子的 "cat" 上。

使用了 Attention 后，Decoder 的输入就不是固定的上下文向量 `c` 了，而是会根据当前翻译的信息，计算当前的 `c`。

<div align=center>
    <img src="zh-cn/img/seq2seq/p25.png" /> 
</div>


Attention 需要保留 Encoder 每一个神经元的隐藏层向量$h$，然后 Decoder 的第 `t` 个神经元要根据上一个神经元的隐藏层向量 $h^{'}_ {t-1}$ 计算出当前状态与 Encoder 每一个神经元的相关性 $e_t$。$e_t$ 是一个 `N` 维的向量 (Encoder 神经元个数为 `N`)，若 $e_t$ 的第 $i$ 维越大，则说明当前节点与 Encoder 第 $i$ 个神经元的相关性越大。$e_t$ 的计算方法有很多种，即相关性系数的计算函数 `a` 有很多种：

<div align=center>
    <img src="zh-cn/img/seq2seq/p26.png" /> 
</div>

上面得到相关性向量 $e_t$ 后，需要进行归一化，使用 softmax 归一化。然后用归一化后的系数融合 Encoder 的多个隐藏层向量得到 Decoder 当前神经元的上下文向量 $c_t$：

<div align=center>
    <img src="zh-cn/img/seq2seq/p27.png" /> 
</div>

这一部分仅对Attention机制有一个大概的了解，我们将在Attention机制的章节中详细的介绍Attention机制的发展和推导过程.


**2.Teacher Forcing训练机制**

其实RNN存在着两种训练模式(mode):

+ free-running mode
+ teacher-forcing mode

free-running mode就是大家常见的那种训练网络的方式: 上一个`state`的输出作为下一个`state`的输入。而Teacher Forcing是一种快速有效地训练循环神经网络模型的方法，该模型使用来自先验时间步长的输出作为输入。

所谓Teacher Forcing，就是在学习时跟着老师(ground truth)走!它是一种**网络训练方法**，对于开发用于机器翻译，文本摘要，图像字幕的深度学习语言模型以及许多其他应用程序至关重要。它每次不使用上一个`stat`e的输出作为下一个`state`的输入，而是直接使用训练数据的标准答案(ground truth)的对应上一项作为下一个`state`的输入。
看一下大佬们对它的评价:

Teacher Forcing工作原理: 在训练过程的 `t` 时刻，使用训练数据集的期望输出或实际输出: $y(t)$， 作为下一时间步骤的输入: $x(t+1)$，而不是使用模型生成的输出$h(t)$或预测的输出$\hat{y}(t)$。

Teacher Forcing的缺点:

Teacher Forcing同样存在缺点: 一直靠老师带的孩子是走不远的。
因为依赖标签数据，在训练过程中，模型会有较好的效果，但是在测试的时候因为不能得到ground truth的支持，所以如果目前生成的序列在训练过程中有很大不同，模型就会变得脆弱。
也就是说，这种模型的cross-domain能力会更差，也就是如果测试数据集与训练数据集来自不同的领域，模型的performance就会变差。


**3.Beam search**

Beam search 方法不用于训练的过程，而是用在测试的。

Beam search 算法在文本生成中用得比较多，用于选择较优的结果（可能并不是最优的）。接下来将以seq2seq机器翻译为例来说明这个Beam search的算法思想。
在机器翻译中，beam search算法在测试的时候用的，因为在训练过程中，每一个decoder的输出是有与之对应的正确答案做参照，也就不需要beam search去加大输出的准确率。
有如下从中文到英语的翻译：

```
中文：
我 爱 学习，学习 使 我 快乐
```

```
英语：
I love learning, learning makes me happy
```

在这个测试中，中文的词汇表是`{我，爱，学习，使，快乐}`，长度为`5`。英语的词汇表是`{I, love, learning, make, me, happy}`（全部转化为小写），长度为`6`。那么首先使用seq2seq中的编码器对中文序列（记这个中文序列为`X`）进行编码，得到语义向量`C`。

<div align=center>
    <img src="zh-cn/img/seq2seq/p28.png" /> 
</div>

得到语义向量`C`后，进入解码阶段，依次翻译成目标语言。在正式解码之前，有一个参数需要设置，那就是beam search中的beam size，这个参数就相当于`top-k`中的`k`，选择前`k`个最有可能的结果。在本例中，我们选择`beam size=3`。

来看解码器的第一个输出$y_1$，在给定语义向量`C`的情况下，首先选择英语词汇表中最有可能`k`个单词，也就是依次选择条件概率$P(y_1∣C)$前3大对应的单词，比如这里概率最大的前三个单词依次是`I，learning，happy`。

接着生成第二个输出$y_2$，在这个时候我们得到了哪些些东西呢，首先我们得到了编码阶段的语义向量`C`，还有第一个输出$y_1$。此时有个问题，$y_1$有3个，怎么作为这一时刻的输入呢（解码阶段需要将前一时刻的输出作为当前时刻的输入），答案就是都试下，具体做法是：

+ 确定`I`为第一时刻的输出，将其作为第二时刻的输入，得到在已知`(C,I)`的条件下，各个单词作为该时刻输出的条件概率$P(y_2∣C,I)$，有6个组合，每个组合的概率为$P(I∣C)P(y_2∣C,I)$。
+ 确定`learning`为第一时刻的输出，将其作为第二时刻的输入，得到该条件下，词汇表中各个单词作为该时刻输出的条件概率$P(y_2∣C,learning)$，这里同样有6种组合；
+ 确定`happy`为第一时刻的输出，将其作为第二时刻的输入，得到该条件下各个单词作为输出的条件概率$P(y_2∣C,happy)$，得到6种组合，概率的计算方式和前面一样。

这样就得到了18个组合，每一种组合对应一个概率值$P(y_1∣C)P(y_2∣C,y1)$，接着在这18个组合中选择概率值`top-3`的那三种组合，假设得到`Ilove，Ihappy，learningmake`。
接下来要做的重复这个过程，逐步生成单词，直到遇到结束标识符停止。最后得到概率最大的那个生成序列。其概率为：

$$P(Y∣C)=P(y_1∣C)P(y_2∣C,y_1),...,P(y_6∣C,y_1,y_2,y_3,y_4,y_5)$$

以上就是Beam search算法的思想，当`beam size=1`时，就变成了贪心算法。

Beam search算法也有许多改进的地方，根据最后的概率公式可知，该算法倾向于选择最短的句子，因为在这个连乘操作中，每个因子都是小于1的数，因子越多，最后的概率就越小。解决这个问题的方式，最后的概率值除以这个生成序列的单词数（记生成序列的单词数为`N`），这样比较的就是每个单词的平均概率大小。
此外，连乘因子较多时，可能会超过浮点数的最小值，可以考虑取对数来缓解这个问题。

**4.有计划地学习(Curriculum Learning)**

如果模型预测的是实值(real-valued)而不是离散值(discrete value)，那么beam search就力不从心了。
因为beam search方法仅适用于具有离散输出值的预测问题，不能用于预测实值（real-valued）输出的问题。

Curriculum Learning是Teacher Forcing的一个变种：

有计划地学习的意思就是: 使用一个概率$p$去选择使用ground truth的输出$y(t)$还是前一个时间步骤模型生成的输出$h(t)$作为当前时间步骤的输入$x(t+1)$。
这个概率$p$会随着时间的推移而改变，这就是所谓的计划抽样(scheduled sampling)
训练过程会从force learning开始，慢慢地降低在训练阶段输入ground truth的频率。


### 5.Seq2Seq实战

我们在<https://github.com/DataXujing/xiaoX>中用tensorflow和pytorch实现了seq2seq的聊天机器人,读者感兴趣可以在笔者的GitHub首页看到该repo.