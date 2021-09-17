## transformaer原理
@[toc]
说明：
&#8195;&#8195;本文主要来自datawhale的开源教程[《基于transformers的自然语言处理(NLP)入门》](https://datawhalechina.github.io/learn-nlp-with-transformers)，此项目也发布在[github](https://github.com/datawhalechina/learn-nlp-with-transformers)。部分内容（章节2.1-2.5，3.1-3.2，4.1-4.2）来自北大博士后卢菁老师的《速通机器学习》一书，这只是我的一个读书笔记，进行一般性总结，所以有些地方进行了简写（比如代码部分，不要喷我。有误请反馈）。想要了解更详细内容可以参考datawhale教程（有更多的图片描述、部分动图和详细的代码）和《速通》一书。
&#8195;&#8195; 另外篇幅有限（可也能是水平有限），关于多头注意力的encoder-decoder attention模块进行运算的更详细内容可以参考[《Transformer概览总结》](https://blog.csdn.net/weixin_38224810/article/details/115587885)。从attention到transformer的API实现和自编程代码实现，可以查阅[《Task02 学习Attention和Transformer》](https://relph1119.github.io/my-team-learning/#/transformers_nlp28/task02)（这篇文章排版很好，干净简洁，看着非常舒服，非常推荐）
### 1. Transformer的兴起
&#8195;&#8195; 2017年，《Attention Is All You Need》论文首次提出了Transformer模型结构并在机器翻译任务上取得了The State of the Art(SOTA, 最好)的效果。2018年，《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》使用Transformer模型结构进行大规模语言模型（language model）预训练（Pre-train），再在多个NLP下游（downstream）任务中进行微调（Finetune）,一举刷新了各大NLP任务的榜单最高分，轰动一时。2019年-2021年，研究人员将<font color='red'>Transformer这种模型结构和预训练+微调这种训练方式相结合，提出了一系列Transformer模型结构、训练方式的改进</font>（比如transformer-xl，XLnet，Roberta等等）。如下图所示，各类Transformer的改进不断涌现。
![《A Survey of Transformers》](https://img-blog.csdnimg.cn/bbf9735b753c42a8a8535a9484526a13.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;图片来自复旦大学邱锡鹏教授：NLP预训练模型综述《A Survey of Transformers》。中文翻译可以参考：https://blog.csdn.net/Raina_qing/article/details/106374584
https://blog.csdn.net/weixin_42691585/article/details/105950385
&#8195;&#8195;另外，由于Transformer优异的模型结构，<font color='red'>使得其参数量可以非常庞大从而容纳更多的信息，因此Transformer模型的能力随着预训练不断提升，随着近几年计算能力的提升，越来越大的预训练模型以及效果越来越好的Transformers不断涌现。</font>
&#8195;&#8195;本教程也将基于[HuggingFace/Transformers, 48.9k Star](https://github.com/huggingface/transformers)进行具体编程和解决方案实现。
&#8195;&#8195;NLP中的预训练+微调的训练方式推荐阅读知乎的两篇文章 [《2021年如何科学的“微调”预训练模型？》](https://zhuanlan.zhihu.com/p/363802308) 和[《从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史》。](https://zhuanlan.zhihu.com/p/49271699)
###  2. 图解Attention
####  2.1 seq2seq
&#8195;&#8195;seq2seq模型是由编码器（Encoder）和解码器（Decoder）组成的。其中，编码器会处理输入序列中的每个元素，把这些信息转换为一个向量（称为上下文context）。当我们处理完整个输入序列后，编码器把上下文（context）发送给解码器，解码器开始逐项生成输出序列中的元素。上下文向量的长度，基于编码器 RNN 的隐藏层神经元的数量
&#8195;&#8195;如何把每个单词都转化为一个向量呢？我们使用一类称为 "word embedding" 的方法。这类方法把单词转换到一个向量空间，这种表示能够捕捉大量单词之间的语义信息。（word2vec）通常embedding 向量大小是 200 或者 300。
&#8195;&#8195;在机器翻译任务中，上下文（context）是一个向量（基本上是一个数字数组)。编码器和解码器在Transformer出现之前一般采用的是循环神经网络。上下文context向量是这类模型的瓶颈。以两个具有不同参数的LSTM分别作为encoder和decoder处理机器翻译为例，结构如下：
![seq2seq](https://img-blog.csdnimg.cn/46a00c56828945599f346b1b65a8b490.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;LSTM1为编码器，在最后时刻的上下文信息 C 包含中文“我爱你”的完整信息,传给给解码器LSTM2，作为翻译阶段,LSTM2的起始状态start。之后每时刻的预测结果作为下一时刻的输入，翻译顺序进行直到终止符<End>停止翻译。
####  2.2 循环神经网络的不足：
<font color='red'>循环神经网络的处理此类任务存在一些不足：</font>
&#8195;&#8195; 1.机器翻译中，使用LSTM的encoder只输出最后时刻的上下文信息C，而这两个模型都存在长距离衰减问题，使得C的描述能力有限。当编码句子较长时，句子靠前部分对C的影响会降低；
&#8195;&#8195; 2.解码阶段，随着序列的推移，编码信息C对翻译的影响越来越弱。因此，越靠后的内容，翻译效果越差。（其实也是因为长距离衰减问题）
&#8195;&#8195; 3.<font color='red'>解码阶段缺乏对编码阶段各个词的直接利用。</font>简单说就是：机器翻译领域，解码阶段的词和编码阶段的词有很强的映射关系，比如“爱”和“love”。<font color='red'>但是seq2seq模型无法再译“love”时直接使用“爱”这个词的信息，因为在编码阶段只能使用全局信息C。</font>（attention在这点做得很好）
&#8195;&#8195;在 2014——2015年提出并改进了一种叫做注意力attetion的技术，它极大地提高了机器翻译的质量。注意力使得模型可以根据需要，关注到输入序列的相关部分。
#### 2.3 attention的引出（重点内容）
基于上面第3点，需要对模型进行改造。（图不是很好，将就看看）
![循环神经网络的改造——引入attention](https://img-blog.csdnimg.cn/5c28e95fee1a4a388b742b600d0fc9c8.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;编码阶段和前面的模型没有区别，保留了各时刻LSTM1的输出向量v。解码阶段，模型预测的方法有了变化，比如在t=1时刻，预测方式为：
&#8195;&#8195; 1.计算LSTM2在t=1时刻的输出q1，以及v1、v2、v3的相似度，即对q1和v1、v2、v3求内积：
s1=<q1,v1>
s2=<q1,v2>
s3=<q1,v3>
&#8195;&#8195; 2.s1、s2、s3可以理解为未归一化的相似度，通过softmax函数对其归一化，得到a1、a2、a3。满足a1+a2+a3=1。a1、a2、a3就是相似度得分attention score。用于表示解码阶段t=1时刻和编码阶段各个词之间的关系。

&#8195;&#8195; 例如解码器在第一个时刻，翻译的词是“I”，它和编码阶段的“我”这个词关系最近，$a_我$的分数最高（比如0.95）。<font color='red'>由此达到让输出对输入进行聚焦的能力，找到此时刻解码时最该注意的词，这就是注意力机制。</font>比起循环神经网络有更好的效果。
&#8195;&#8195; （<font color='red'>attention score只表示注意力强度，是一个标量，一个系数，不是向量，不含有上下文信息，所以还不是最终输出结果。在此回答一些小伙伴的疑问</font>）

&#8195;&#8195; 3.根据相似度得分对v1、v2、v3进行加权求和，即$h_{1}=a_{1}v_{1}+a_{2}v_{2}+a_{3}v_{3}$。
&#8195;&#8195; 4.向量h1经过softmax函数来预测单词“I”。可以看出，此时的h1由最受关注的向量$v_我$主导。因为$a_我$最高。

&#8195;&#8195;<font color='red'>上述模型就是注意力（Attention）模型</font>)（这里没有用Self-Attention代替LSTM，主要还是讲attention机制是基于什么原因引出的。好的建议可以反馈给我）。(此处的模型没有key向量，是<font color='red'>做了简化，即向量$K=V$</font>)

&#8195;&#8195;注意力模型和人类翻译的行为更为相似。<font color='red'>人类进行翻译时，会先通读“我爱你”这句话，从而获得整体语义（LSTM1的输出C）。而在翻译阶段，除了考虑整体语义，还会考虑各个输入词（“我”、“爱”、“你”）和当前待翻译词之间的映射关系（权重a1、a2、a3来聚焦注意力）</font>

&#8195;&#8195;一个注意力模型不同于经典的（seq2seq）模型，主要体现在 2 个方面：
&#8195;&#8195;1.编码器会把更多的数据传递给解码器。<font color='red'>编码器把所有时间步的 hidden state（隐藏层状态）传递给解码器，而非只传递最后一个 hidden state。</font>
&#8195;&#8195;2.解码器在产生输出之前，做了一个额外的处理。<font color='red'>把注意力集中在与该时间步相关的输入部分：</font>
a. 查看所有接收到的编码器的 hidden state（隐藏层状态）。其中，编码器中每个 hidden state（隐藏层状态）都对应到输入句子中一个单词。
b. 给每个 hidden state（隐藏层状态）一个分数（attention score）。
c. 将每个 hidden state（隐藏层状态）乘以经过 softmax 的对应的分数，分数的高低代表了注意力的强度。分数更大的hidden state更会被关注。

Tips：上面计算相似度s=<q,k>时，s要除以$\sqrt(d_{key})$(Key 向量的长度）。原因是：
&#8195;&#8195;求相似度时，如果特征维度过高（如词向量embedding维度），就会导致计算出来的相似度s过大。s值的过大会导致归一化函数softmax饱和（softmax在s值很大的区域输出几乎不变化）使得归一化后计算出来的结果a要么趋近于1要么趋近于0。即加权求和退化成胜者全拿，则解码时只关注注意力最高的（attention模型还是希望别的词也有权重）而且softmax函数的饱和区导数趋近于0，梯度消失。所以对公式s=<q,k>进行优化：
$$s=\frac{<q,k>}{\sqrt{d_{key}}}$$
&#8195;&#8195;q和k求内积，所以其实key和q的向量长度一样。
#### 2.4 从机器翻译推广到attention的一般模式
（本来不想写的，想到一个问题，还是把这节补了）
&#8195;&#8195;Attention不止是用来做机器翻译，甚至是不止用在NLP领域。换一个更一般点的例子，来说明Attention的一般模式。
&#8195;&#8195;比如有一个场景是家长带小孩取玩具店买玩具，用模型预测最后玩具是否会被购买。每个玩具有两类特征，1-形状颜色功能等，用来吸引孩子；第二类特征是加个、安全、益智性等，用来决定家长是否购买。
&#8195;&#8195;假设孩子喜好用特征向量q表示，玩具第一类特征用向量k表示，第二类特征用向量v表示，模型结果如下：
![attention一般模式](https://img-blog.csdnimg.cn/68568d9356bc4dfb9d34041bf3fa7384.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;首先计算q和k的相似度$s_{1}-s_{n}$，并归一化到$a_{1}-a_{n}$，a反映了孩子对玩具的喜好程度（权重）。接下来a对特征v进行加权求和（家长角度考虑），得到向量h。最后家长是否购买玩具是由向量h决定的。
&#8195;&#8195;上述过程就是Attention的标准操作流程。Attention模型三要素是$q、K、 V$。$K、 V$矩阵分别对应向量序列$k_1$到$k_n$和$v_1$到$v_n$。由于中间涉及到加权求和，所以这两个序列长度一致，而且元素都是对应的。即$k_j$对应$v_j$。但是k和v分别表示两类特征，所以向量长度可以不一致。
&#8195;&#8195;为了运算方便，可以将Attention操作计算为：$h=Attention（q、K、 V）$。q也可以是一个向量序列Q（对应机器翻译中输入多个单词），此时输出也是一个向量序列H。Attention通用标准公式为：$$H=Attention(Q,K,V)=\begin{bmatrix}
Attention(q_{1},K,V)\\ 
...\\ 
Attention(q_{m},K,V)\end{bmatrix}$$
&#8195;&#8195;这里，$Q、K、 V、H$均为矩阵（向量序列）。其中，<font color='red'>$H$和$Q$序列长度一致，各行一一对应（一个输入对应一个输出），$K$和$V$序列长度一致，各行一一对应。
#### 2.5 Attention模型的改进形式
Attention模型计算相似度，除了直接求内积<q,k>，还有很多其它形式。
$$s=A^{T}Tanh（qW+kU) $$
&#8195;&#8195;多层感知机，$A 、 W 、 U$都是待学习参数。这种方法不仅避开了求内积时 q 和 k 的向量长度必须一致的限制,还可以进行不同空间的向量匹配。例如,在进行图文匹配时, q 和 k 分别来自文字空间和图像空间,可以先分别通过 $W 、 U$将它们转换至同一空间,再求相似度。
&#8195;&#8195;上面式子中，$W 、U$是矩阵参数，相乘后可以使q和k的维度一致。比如机器翻译中，中文一词多义情况比较多，中文向量q维度可以设置长一点，英文中一词多义少，k的维度可以设置短一点。$qW+kU$是同长度向量相加，结果还是一个向量，再经过列向量$A^{T}$相乘，得到一个标量，即attention score数值。
&#8195;&#8195;第二个改进式子：
$$s=\frac{<qWk^{T}>}{\sqrt{d_{key}}}$$
&#8195&#8195;其中，W是待学习参数，q和k维度可以不同。
&#8195;&#8195;第三个改进式子：
$$s=\frac{<qW,kU>}{\sqrt{d_{key}}}$$
#### 2.6 Self-Attention
##### 2.6.1 Self-Attention结构
&#8195;&#8195;$H=Attention(Q,K,V)$有一种特殊情况，就是$Q=K=V$,也就是自注意力模型self-attention。（一个输入向量同时承担了三种角色）
![Self-Attention](https://img-blog.csdnimg.cn/9b1819fc6a5d48d6aedab8b961ff66f2.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;如上图所示，self-attention中query、key、value的这三样东西其实是一样的，它们的形状都是：(L,N,E) 
L：输入序列的长度（例如一个句子的长度）
N：batch size（例如一个批的句子个数）
E：词向量长度
&#8195;&#8195;Self-Attention只有一个序列（即只有一种输入特征），比如机器翻译中，输入只有词向量。这应该就是Self-Attention和Attention的区别。
##### 2.6.2 Self-Attention在机器翻译中的优势
机器人第二定律
&#8195;&#8195;**机器人必须服从人给予 ==它== 的命令，当 ==该命令== 与 ==第一定律== 冲突时例外。**
&#8195;&#8195;句子中高亮的3 个部分，用于指代其他的词。如果不结合它们所指的上下文，就无法理解或者处理这些词。当一个模型要处理好这个句子，它必须能够知道：

==它== 指的是机器人
==该命令== 指的是这个定律的前面部分，也就是 人给予 ==它== 的命令
==第一定律== 指的是机器人第一定律
&#8195;&#8195; Self Attention 能做到这一点。它在处理某个词之前，将模型中这个词的相关词和关联词的理解融合起来（并输入到一个神经网络）。它通过对句子片段中每个词的相关性打分（attention score），并将这些词向量加权求和。

&#8195;&#8195;举个例子，下图顶部模块中的 Self Attention 层在处理单词==it==的时候关注到 ==a robot==。它最终传递给神经网络的向量，是这 3 个单词的词向量加权求和的结果。
![self-attention](https://img-blog.csdnimg.cn/42a06b9cd5564b4788d1ff21cc651978.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195; Self Attention机制，会把其他单词的理解融入处理当前的单词。使得模型不仅能够关注这个位置的词，而且能够关注句子中其他位置的词，作为辅助线索，进而可以更好地编码当前位置的词。

##### 2.6.3Self-Attention 过程
&#8195;&#8195;如上一节所讲Self Attention 它在处理某个词之前，通过对句子片段中每个词的相关性进行打分，并将这些词的表示向量加权求和。

Self-Attention 沿着句子中每个 token 的路径进行处理，主要组成部分包括 3 个向量。
&#8195;&#8195; ==Query==：Query 向量是当前单词的表示，用于对其他所有单词（使用这些单词的 key 向量）进行评分。我们只关注当前正在处理的 token 的 query 向量。
&#8195;&#8195; ==Key==：Key 向量就像句子中所有单词的标签。它们就是我们在搜索单词时所要匹配的。
&#8195;&#8195; ==Value==：Value 向量是实际的单词表示，一旦我们对每个词的相关性进行了评分，我们需要对这些向量进行加权求和，从而表示当前的词。
![slef-attention过程](https://img-blog.csdnimg.cn/369d5ca872984b4bbe1b5a6eb0eb084e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
一个粗略的类比是把它看作是在一个文件柜里面搜索
向量     |含义
-------- | -----
Query   |一个==便签==，上面写着你正在研究的主题
Key  | 柜子里的文件夹的==标签==
Value  |文件夹里面的内容

&#8195;&#8195;首先将==便签==与==标签==匹配，会为每个文件夹产生一个分数（attention score）。然后取出匹配的那些文件夹里面的内容 Value 向量。最后我们将每个 Value 向量和分数加权求和，就得到 Self Attention 的输出。（下图是单指计算it的时候）
![self-attention输出](https://img-blog.csdnimg.cn/68f22c3e7b8345ca96b49baeefe47ff0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;这些加权的 Value 向量会得到一个向量，它将 50% 的注意力放到单词==robot== 上，将 30% 的注意力放到单词 ==a==，将 19% 的注意力放到单词 ==it==。最终一个具有高分数的 Value 向量会占据结果向量的很大一部分。
&#8195;&#8195;注意：上面都是展示大量的单个向量，是想把重点放在词汇层面上。而实际的代码实现，是通过巨大的矩阵相乘来完成的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/59b46859a6f540b59416d184e8bda580.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;更详细的self-attention介绍可以参考[《图解GPT》](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.4-%E5%9B%BE%E8%A7%A3GPT?id=%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8%EF%BC%9A%E4%BA%86%E8%A7%A3-gpt2)
##### 2.6.3 Self Attention和循环神经网络对比
&#8195;&#8195;Self Attention是一个词袋模型，对词序不敏感。因为每时刻输出$h_{i}=a_{1}v_{1}+a_{2}v_{2}+a_{3}v_{3}...=\sum a_{i,j}v_{i}$。这是一个加权求和，调换词序对结果不影响。所以对比循环神经网络可以发现：
&#8195;&#8195;1.LSTM、RNN、ELMo等循环神经网络模型考虑了词序，但是正因为如此，每个时刻的输出依赖上一时刻的输入，所以只能串行计算，无法并行，无利用更庞大的算力来加快模型训练，这也是循环神经网络渐渐被attention替代的原因之一。Self Attention模型不考虑词序，所有字是全部同时训练的, 各时刻可以独立计算，可以并行，反而成了它的优点。
&#8195;&#8195;2.循环神经网络的存在长距离衰减问题，也没有注意力机制（前面讲过，不再重复）attention可以无视词的距离，因为每时刻都是加权求和，考虑了每一个词，不存在信息衰减。
&#8195;&#8195;<font color='red'>LSTM:非词袋模型，含有顺序信息，无法解决长距离依赖，无法并行，没有注意力机制
&#8195;&#8195;Self Attention：词袋模型，不含位置信息，没有长距离依赖，可以并行，有注意力机制。</font >
但是这里有个问题，语义和词序是有一定的关联的，为了解决这个问题，有两个办法：
&#8195;&#8195;1.位置嵌入（Position Embeddings)
&#8195;&#8195;2.位置编码（Position Encodings）
&#8195;&#8195;在transformer部分会进一步介绍
##### 2.6.4计算 Self Attention ：
&#8195;&#8195;第 1 步：将输入词向量，映射为三个新的向量：Query 向量，Key 向量，Value 向量。这 3 个向量是词向量分别和 3 个矩阵相乘得到的，而这个矩阵是我们要学习的参数（这应该是改进版的Self Attention）
&#8195;&#8195;第 2 步：计算 Attention Score（注意力分数）。有几种计算方式，常用的是求点积。注意力分数决定了我们在编码时，需要对句子中其他位置的每个词放置多少的注意力。
&#8195;&#8195;第 3 步：把每个分数除以$\sqrt(d_{key})$(Key 向量的长度，也可以是其它数），再进行softmax归一化，得到系数a；
&#8195;&#8195;第 4步：a和向量V加权求和得z。这种做法背后的直觉理解就是：对于分数高的位置，相乘后的值就越大，我们把更多的注意力放到了它们身上
&#8195;&#8195;第 5 步是把上一步得到的向量相加，就得到了 Self Attention 层在这个位置的输出。
&#8195;&#8195;输出向量会输入到前馈神经网络。在实际的代码实现中，Self Attention 的计算过程是使用矩阵来实现的，这样可以加速计算，一次就得到所有位置的输出向量。
##### 2.6.5使用矩阵计算 Self-Attention
&#8195;&#8195; 1.把所有词向量放到一个矩阵 X 中(输入)，然后分别和3 个权重矩阵$W^Q$, $W^K$ $W^V$相乘，得到 Q，K，V 矩阵。 X 中的每一行，表示句子中的每个词的词向量，长度是 512。Q，K，V 矩阵中的每一行表示 Query 向量，Key 向量，Value 向量，向量长度是 64。
&#8195;&#8195;  2.由于我们使用了矩阵来计算，我们可以把上面的第 2 步到第 5 步压缩为一步，直接得到 Self Attention 的输出。
![矩阵计算 Self-Attention](https://img-blog.csdnimg.cn/d955a14eed2b4b339a37c0b27a836083.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
### 3.多头注意力机制（multi-head attention）
#### 3.1 从attention引出multi-head attention
&#8195;&#8195; 在标准的 Attention 模型 Attention ( Q，K , V )中，对 Q 和 K 进行相似度匹配,相当于在特定的角度上计算相似度, Q 和 K 所在的向量空间即为观察角度,并且有且仅有一个观察角度。然而,实际情况要复杂得多。
&#8195;&#8195; 例如,“小明养了一只猫,它特别调皮可爱,他非常喜欢它”。将这句话作为Self - Attention 的输入序列,其中每个词都和其他词有一个对应的相似度得分。<font color='red'>“猫”从指代的角度看,与“它”的匹配度最高,但从属性的角度看,与“调皮”“可爱”的匹配度最高。由于标准的 Attention 模型无法处理这种多语义的情况,所以,需要将向量序列 Q 、 K 、 V 多次转换至不同的语义空间，对标准的 Attention 模型进行多语义匹配改进。</font>(Self - Attention是Attention的特例，所以此种改进方法也适用）
#### 3.2 从Attention到Multi-Head Attention的公式变换：
&#8195;&#8195;1.用矩阵系数$W_{1}^{Q}、W_{1}^{K}、W_{1}^{V}$将Q，K , V 转至语义空间1，公式为：
$$Q_{1}=QW_{1}^{Q}=\begin{bmatrix}
q_{1}W_{1}^{Q}\\ 
...\\ 
q_{m}W_{1}^{Q}\end{bmatrix}$$
$$K_{1}=KW_{1}^{K}=\begin{bmatrix}
q_{1}W_{1}^{K}\\ 
...\\ 
q_{m}W_{1}^{K}\end{bmatrix}$$
$$V_{1}=VW_{1}^{V}=\begin{bmatrix}
q_{1}W_{1}^{V}\\ 
...\\ 
q_{m}W_{1}^{V}\end{bmatrix}$$
&#8195;&#8195;Q，K , V都是向量序列，因此特征变换就是各时序上的向量变换。$W_{1}^{Q}、W_{1}^{K}、W_{1}^{V}$是待学习参数。
&#8195;&#8195;2.在转换后的语义空间进行attention计算，即$head_1= Attention（Q_1，K_1 , V_1）$。$head_1$也是向量序列，长度和Q一致（一个输入对应一个输出）。
&#8195;&#8195;3.用矩阵系数$W_{2}^{Q}、W_{2}^{K}、W_{2}^{V}$将Q 、 K 、 V 转换至语义空间2，同样进行Attention计算，得到$head_2$。同理计算出$head_3....head_c$。
&#8195;&#8195;4.$head_c$是多个不同语义空间注意力计算的结果，将它们串联起来。$head_1....head_c$均为向量序列，串联之后还是向量序列。
总结起来，多头注意力模型公式可以写成：
$$Multi-Head（Q ,K , V )=concat(head_1....head_c)W^{O}=\begin{bmatrix}
concat(h_{1,1}...h_{n,1})W^{O}\\ 
...\\ 
concat(h_{1,m}...h_{n,m}W^{O}\end{bmatrix}$$
&#8195;&#8195;多头注意力结果串联在一起维度可能比较高，所以通过$W^{O}$进行一次线性变换，实现降维和各头信息融合的目的，得到最终结果。
&#8195;&#8195;多头注意力模型中，head数是一个超参数，语料大，电脑性能好就可以设置的高一点。宁可冗余也不遗漏。
&#8195;&#8195;多头注意力从如下两个方面增强了 attention 层的能力：
&#8195;&#8195; 1.它扩展了模型关注不同位置的能力。例如当我们翻译句子：The animal didn’t cross the street because it was too tired时，我们想让机器知道其中的it指代的是什么。这时，多头注意力机制会有帮助。
&#8195;&#8195; 2.多头注意力机制赋予 attention 层多个“子表示空间”，可以表示不同角度下的语义信息。

&#8195;&#8195;在前面的讲解中，我们的 K、Q、V 矩阵的序列长度都是一样的。但是在实际中，<font color='red'>K、V 矩阵的序列长度是一样的（加权求和），而 Q 矩阵的序列长度可以不一样。
&#8195;&#8195;这种情况发生在：在解码器部分的Encoder-Decoder Attention层中，Q 矩阵是来自解码器下层，而 K、V 矩阵则是来自编码器的输出。</font >
#### 3.3 多头注意力模型的可视化表示
&#8195;&#8195; 下面我们会看到，多头注意力机制会有多组$W^Q$, $W^K$ $W^V$的权重矩阵（在 Transformer 的论文中，使用了 8 组注意力（attention heads）。因此，接下来我也是用 8 组注意力头 （attention heads））。多抽头的每一组注意力的 的权重矩阵都是随机初始化的，都是不一样的。经过训练之后，每一组注意力$W^Q$, $W^K$ $W^V$  可以看作是把输入的向量映射到一个”子表示空间“。
![在这里插入图片描述](https://img-blog.csdnimg.cn/f47d80e90096477db2438c321e917d90.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195; 将输入 X 和每组注意力的$W^Q$, $W^K$ $W^V$相乘，得到 8 组 Q, K, V 矩阵。接着，我们把每组 K, Q, V 进行计算，得到每组的 Z 矩阵，一共就得到 8 个 Z 矩阵。（就是上一节的h，这个图写的是Z）
&#8195;&#8195;把8个矩阵拼接起来，然后和另一个权重矩阵$W^O$相乘。（前馈神经网络层接收的也是 1 个矩阵，而不是8个。其中每行的向量表示一个词）![多头注意力模型](https://img-blog.csdnimg.cn/ceb5a78cab9e441babb82561c8f6659f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;总结一下：把 8 个矩阵$Z_{0},Z_{1}...Z_{7}$拼接起来，<font color='red'>拼接后的矩阵和 $W^O$ 权重矩阵相乘，得到最终的矩阵 Z。这个矩阵包含了所有 attention heads（注意力头） 的信息。</font>这个矩阵会输入到 FFNN (Feed orward Neural Network)层。
&#8195;&#8195;这就是多头注意力的全部内容。下面我把所有的内容都放到一张图中，这样你可以总揽全局，在这张图中看到所有的内容。
![多头注意力模型](https://img-blog.csdnimg.cn/adeaa975448f46fbb0b1387c151d8660.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
#### 3.4 代码实现矩阵计算 Attention
&#8195;&#8195;下面我们是用代码来演示，如何使用矩阵计算 attention。首先使用 PyTorch 库提供的函数实现，然后自己再实现。
&#8195;&#8195;PyTorch 提供了 MultiheadAttention 来实现 attention 的计算。(其实应该理解为多头自注意力模型）
##### 3.4.1 定义MultiheadAttention
```python
torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
```
&#8195;&#8195;1.embed_dim最终输出的 K、Q、V 矩阵的维度，这个维度需要和词向量的维度一样
&#8195;&#8195;2.num_heads：设置多头注意力的数量。要求embed_dim%num_heads==0，即要能被embed_dim整除。这是为了把词的隐向量长度平分到每一组，这样多组注意力也能够放到一个矩阵里，从而并行计算多头注意力。
&#8195;&#8195;3.dropout：这个 dropout 加在 attention score 后面
&#8195;&#8195;例如，我们前面说到，8 组注意力可以得到 8 组 Z 矩阵，然后把这些矩阵拼接起来，得到最终的输出。
&#8195;&#8195;如果最终输出的每个词的向量维度是 512，那么每组注意力的向量维度应该是512/8=64 如果不能够整除，那么这些向量的长度就无法平均分配。

##### 3.4.2 forward的输入（引出mask机制）

&#8195;&#8195;定义 MultiheadAttention 的对象后，调用forward时传入的参数如下。
```python
forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None)
```
&#8195;&#8195;1.query：对应于 Query矩阵，形状是 (L,N,E) 。其中 L 是输出序列长度，N 是 batch size，E 是词向量的维度
&#8195;&#8195;2.key：对应于 Key 矩阵，形状是 (S,N,E) 。其中 S 是输入序列长度，N 是 batch size，E 是词向量的维度
&#8195;&#8195;3.value：对应于 Value 矩阵，形状是 (S,N,E) 。其中 S 是输入序列长度，N 是 batch size，E 是词向量的维度

**下面重点介绍.key_padding_mask和attn_mask：**
&#8195;&#8195;**4.key_padding_mask：**
&#8195;&#8195;在$self \ attention$的计算过程中, 我们通常使用$mini \ batch$来计算, 也就是一次计算多句话, , 而一个$mini \ batch$是由多个不等长的句子组成的, 我们就需要按照这个$mini \ batch$中最大的句长对剩余的句子进行补齐长度, 我们一般用$0$来进行填充, 这个过程叫做$padding$.
&#8195;&#8195;但这时在进行$softmax$的时候就会产生问题, 回顾$softmax$函数$$\sigma (\mathbf {z} )_{i}={\frac {e^{z_{i}}}{\sum _{j=1}^{K}e^{z_{j}}}}$$
&#8195;&#8195; $e^0$=1, 这样的话$softmax$中被$padding$的部分就参与了运算, 就等于是让无效的部分参与了运算, 会产生很大隐患, 这时就需要做一个$mask$让这些无效区域不参与运算, 我们一般给无效区域加一个很大的负数的偏置, 也就是:$$z_{illegal} = z_{illegal} + bias_{illegal}$$$$bias_{illegal} \to -\infty$$$$e^{z_{illegal}} \to 0 $$
&#8195;&#8195;经过上式的$masking$我们使无效区域经过$softmax$计算之后还几乎为$0$, 这样就避免了无效区域参与计算.。

 key_padding_mask = ByteTensor，非 0 元素对应的位置会被忽略
 key_padding_mask =BoolTensor， True 对应的位置会被忽略
 （<font color='red'>如果 key_padding_mask对应是0、1张量，那么1表示mask，如果是布尔张量，则true表示mask</font >）
 
 &#8195;&#8195;key_padding_mask形状是 (N,S)。其中 N 是 batch size，S 是输入序列长度，里面的值是1或0。我们先取得key中有padding的位置，然后把mask里相应位置的数字设置为1，这样attention就会把key相应的部分变为"-inf". (为什么变为-inf参考https://blog.csdn.net/weixin_41811314/article/details/106804906)
 
&#8195;&#8195;**5.attn_mask**：表示不计算未来时序的信息。以机器翻译为例，Decoder的Self-Attention层只允许关注到输出序列中早于当前位置之前的单词，在Self-Attention分数经过Softmax层之前，屏蔽当前位置之后的位置。
&#8195;&#8195;attn_mask形状可以是 2D (L,S)，或者 3D (N∗numheads,L,S)。其中 L 是输出序列长度，S 是输入序列长度，N 是 batch size。
 attn_mask =ByteTensor，非 0 元素对应的位置会被忽略（不计算attention，不看这个词）
attn_mask =BoolTensor， True 对应的位置会被忽略

&#8195;&#8195;mask机制更具体内容可以参考[Transformer相关——（7）Mask机制](https://ifwind.github.io/2021/08/17/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%887%EF%BC%89Mask%E6%9C%BA%E5%88%B6/)
##### 3.4.3 forward的输出
&#8195;&#8195;解码（decoding ）阶段的每一个时间步都输出一个翻译后的单词（以英语翻译为例）。
```python
#实例化一个nn.MultiheadAttention
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
```
即输出是：
&#8195;&#8195;attn_output：即最终输出的的注意力Z，形状是为(L,N,E)。 L 是输出序列长度，N 是 batch size，E 是词向量的维度
&#8195;&#8195;attn_output_weights：注意力系数a。形状是 (N,L,S) 
代码示例如下：
```c
# nn.MultiheadAttention 输入第0维为length
# batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
query = torch.rand(12,64,300)
# batch_size 为 64，有 10 个词，每个词的 Key 向量是 300 维
key = torch.rand(10,64,300)
# batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
value= torch.rand(10,64,300)
embed_dim = 299
num_heads = 1
# 输出是 (attn_output, attn_output_weights)
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output = multihead_attn(query, key, value)[0]
# output: torch.Size([12, 64, 300])
# batch_size 为 64，有 12 个词，每个词的向量是 300 维
print(attn_output.shape)
```
#### 3.5 手动实现attention
```c
	def forward(self, query, key, value, mask=None):
		bsz = query.shape[0]
		Q = self.w_q(query)
		K = self.w_k(key)
		V = self.w_v(value)
		1.#计算attention score 
		attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
	    if mask isnotNone:
	         attention = attention.masked_fill(mask == 0, -1e10)# mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
	    2.#计算上一步结果的最后一维做 softmax，再经过 dropout，得attention。
	    attention = self.do(torch.softmax(attention, dim=-1))
	    3.attention结果与V相乘，得到多头注意力的结果
	    # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
	    x = torch.matmul(attention, V)
	    4.转置x并拼接多头结果
	    x = x.permute(0, 2, 1, 3).contiguous()#转置
	    x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))#拼接
	    x = self.fc(x)
	    return x

	# batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
	query = torch.rand(64, 12, 300)
	# batch_size 为 64，有 12 个词，每个词的 Key 向量是 300 维
	key = torch.rand(64, 10, 300)
	# batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
	value = torch.rand(64, 10, 300)
	attention = MultiheadAttention(hid_dim=300, n_heads=6, dropout=0.1)
	output = attention(query, key, value)
	## output: torch.Size([64, 12, 300])
	print(output.shape)
```
### 4. 图解transformer
#### 4.1 自注意力模型的缺点及transformer的提出
&#8195;&#8195;虽然自注意力模型有很多优势，但是,要想真正取代循环神经网络,自注意力模型还需要解决如下问题:.
1.在计算自注意力时,没有考虑输入的位置信息,因此无法对序列进行建模;.
2.输入向量 T ,同时承担了Q、K、V三种角色,导致其不容易学习;
3.只考虑了两个输人序列单元之间的关系,无法建模多个输人序列单元之间更复杂的关系;
4.自注意力计算结果互斥,无法同时关注多个输人

&#8195;&#8195;2017 年，Google 提出了 Transformer 模型，综合解决了以上问题。<font color='red'>Transformer也使用了用 Encoder-Decoder框架。为了提高模型能力，每个编码解码块不再是由RNN网络组成，而是由Self Attention+FFNN 的结构组成(前馈神经网络）。</font> 从本质上讲，transformer是将序列转换为序列，所以叫这个名字。可以翻译为转换器，也有人叫变压器。
&#8195;&#8195;概括起来就是transformer使用了位置嵌入$(positional \ encoding)$来理解语言的顺序, 使用自注意力机制和全连接层来进行计算。
![transformer](https://img-blog.csdnimg.cn/fcbca96864c34a858fd198ea37b06699.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195; Transformer 可以拆分为 2 部分：左边是编码部分(encoding component)，右边是解码部分(decoding component)。

#### 4.2 transformer的改进
##### 4.2.1 融入位置信息
&#8195;&#8195;为了解决Self-Attention词袋模型问题，除了词向量，还应该给输入向量引入不同的位置信息。有两种引人位置信息的方式:
&#8195;&#8195;1.位置嵌入( Position Embeddings )：与词嵌入类似,即为序列中每个绝对位置赋予一个连续、低维、稠密的向量表示。
&#8195;&#8195;2.位置编码( Position Encodings )：使用函数$f:\mathbb{N}\rightarrow \mathbb{R}^{d}$ ,直接将一个整数(位置索引值)映射到一个 d 维向量上。映射公式为:
$$PosEnc(p,i)=
\begin{Bmatrix}
sin(\frac{p}{10000^{\frac{i}{d}}})\\ 
cos(\frac{p}{10000^{\frac{i-1}{d}}})\end{Bmatrix}$$
&#8195;&#8195;其中，p为序列中位置索引值，$0\leqslant i< d$是位置编码向量中的索引值。
##### 4.2.2 输入向量角色信息
&#8195;&#8195;原始的自注意力模型在计算注意力时，直接使用两个输入向量计算注意力系数a，然后使用得到的注意力对同一个输入向量加权,这样导致一个输入向量同时承担了三种角色:査询( Query )键( Key )和值( Value )。（见上面2.6节）
&#8195;&#8195;更好的做法是,对不同的角色使用不同的向量。<font color='red'>即使用不同的参数矩阵对原始的输人向量做线性变换,从而让不同的变换结果承担不同的角色。</font>具体地,分别使用三个不同的参数矩阵$W^Q$, $W^K$, $W^V$，将输入向量$x_{i}$映射为三个新的向量 $q_{i}$、$k_{i}$、$v_{i}$，分别表示查询、键和值对应的向量。
##### 4.2.3多层自注意力（多层编码解码结构）
&#8195;&#8195;原始的自注意力模型仅考虑了序列中任意两个输人序列单元之间的关系,而在实际应用中,往往需要同时考虑更多输入序列单元之间的关系,即更高阶的关系。如果<font color='red'>直接建模高阶关系,会导致模型的复杂度过高。而类似于图模型中的消息传播机制( Message Propogation ),这种高阶关系可以通过堆叠多层自注意力模型实现。

&#8195;&#8195;另一方面,直接堆叠多层注意力模型,由于每层的变换都是线性的(注意力计算一般使用线性函数，只是简单的加权求和),最终模型依然是线性的。因此,为了增强模型的表示能力,往往在每层自注意力计算之后,增加一个非线性的前馈神经网络FFNN。如果将自注意力模型看作特征抽取器,那么FFNN就是最终的分类器。

&#8195;&#8195;同时,为了使模型更容易学习,还可以使用层归一化( Layer Normalization )残差连接( Residual Connections )等深度学习的训练技巧。这些都加在一起，叫Transformer块（Block）
##### 4.2.4多头自注意力( Multi - head Self - attention )
&#8195;&#8195;由于自注意力结果需要经过归一化,导致即使一个输人和多个其他的输人相关,也无法同时为这些输入赋予较大的注意力值,即自注意力结果之间是互斥的,无法同时关注多个输人。
&#8195;&#8195;因此,如果能<font color='red'>使用多组自注意力模型产生多组不同的注意力结果,则不同组注意力模型可能关注到不同的输人上,从而增强模型的表达能力。具体来说，只需要设置多组映射矩阵即可。然后将产生的多个输出向量拼接。</font>为了将输出结果作为下一组的输人,还需要将拼接后的输出向量再经过一个线性映射,映射回 d 维向量。该模型又叫作多头自注意力( Multi - head Self - attention )模型。
&#8195;&#8195;从另一方面理解、多头自注意力机制相当于多个不同的自注意力模型的集成( Ensemble ),也会增强模型的效果。类似卷积神经网络中的多个卷积核,也可以将不同的注意力头理解为抽取不同类型的特征。
#### 4.3 Encoder
##### 4.3.1 Encoder层结构
&#8195;&#8195;Encoder由多层编码器组成，每层编码器在结构上都是一样的，但不同层编码器的权重参数是不同的。每层编码器里面，主要由以下两部分组成：

1.Self-Attention Layer
2.Feed Forward Neural Network（前馈神经网络，缩写为 FFNN）

&#8195;&#8195;输入编码器的文本数据，首先会经过一个 Self Attention 层，这个层处理一个词的时候，不仅会使用这个词本身的信息，也会使用句子中其他词的信息（你可以类比为：当我们翻译一个词的时候，不仅会只关注当前的词，也会关注这个词的上下文的其他词的信息）。

&#8195;&#8195;接下来，Self Attention 层的输出会经过前馈神经网络FFNN。

&#8195;&#8195;<font color='red'>Self-Attention模型的作用是提取语义级别的信息（不存在长距离依赖），而FFNN是在各个时序上对特征进行非线性变换，提高网络表达能力。</font>
##### 4.3.2残差连接和标准化
&#8195;&#8195;编码器的每个子层（Self Attention 层和FFNN）都有一个Add&normalization层。如下如所示：
![编码器残差连接和层标准化](https://img-blog.csdnimg.cn/53e1f9be1f544249bbe0e3de5b08008e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195; Add&normalization的意思是LayerNorm（X+Z)。即残差连接和标准化。

&#8195;&#8195; Add残差连接是 用到Shortcut 技术，解决深层网络训练时退化问题。具体解释可以看文章[《Transformer相关——（5）残差模块》](https://ifwind.github.io/2021/08/17/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%885%EF%BC%89%E6%AE%8B%E5%B7%AE%E6%A8%A1%E5%9D%97/)
&#8195;&#8195; LayerNorm 用于提高网络的训练速度，防止过拟合。具体可以参考[《Transformer相关——（6）Normalization方式》](https://ifwind.github.io/2021/08/17/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%886%EF%BC%89Normalization%E6%96%B9%E5%BC%8F/)
（再写下去感觉这篇文章绷不住了，太长抓不住主线）
##### 4.3.3 Transformer-encoder结构梳理——数学表示.
&#8195;&#8195; 经过前面部分的讲解，我们已经知道了很多知识点。下面用公式把一个$transformer \ block$的计算过程整理一下:
1). 字向量与位置编码:
$$X = EmbeddingLookup(X) + PositionalEncoding \tag{eq.2}$$$$X \in \mathbb{R}^{batch \ size  \ * \  seq. \ len. \  * \  embed. \ dim.} $$2). 自注意力机制:
$$Q = Linear(X) = XW_{Q}$$$$K = Linear(X) = XW_{K} \tag{eq.3}$$$$V = Linear(X) = XW_{V}$$$$X_{attention} = SelfAttention(Q, \ K, \ V) \tag{eq.4}$$3). 残差连接与$Layer \ Normalization$$$X_{attention} = X + X_{attention} \tag{eq. 5}$$$$X_{attention} = LayerNorm(X_{attention}) \tag{eq. 6}$$4). 下面进行$transformer \ block$结构图中的第4部分, 也就是$FeedForward$, 其实就是两层线性映射并用激活函数激活, 比如说$ReLU$:
$$X_{hidden} = Activate(Linear(Linear(X_{attention}))) \tag{eq. 7}$$5). 重复3).:$$X_{hidden} = X_{attention} + X_{hidden}$$$$X_{hidden} = LayerNorm(X_{hidden})$$$$X_{hidden} \in \mathbb{R}^{batch \ size  \ * \  seq. \ len. \  * \  embed. \ dim.} $$
#### 4.4 Decoder（解码器）
&#8195;&#8195;<font color='red'>同理，解码器也具有这两层，但是这两层中间还插入了一个 Encoder-Decoder Attention 层。</font>编码器输出最终向量，将会输入到每个解码器的Encoder-Decoder Attention层，用来帮解码器把注意力集中中输入序列的合适位置。（类似于 seq2seq 模型 中的 Attention）。
&#8195;&#8195;在解码器的子层里面也有层标准化（layer-normalization）。假设一个 Transformer 是由 2 层编码器和两层解码器组成的，如下图所示。
![2层的transformer](https://img-blog.csdnimg.cn/ed6bb16f86ec497eae9c65be754cad7c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;在完成了编码（encoding）阶段之后，我们开始解码（decoding）阶段。解码（decoding ）阶段的每一个时间步都输出一个翻译后的单词（这里的例子是英语翻译）。接下来会重复这个过程，直到输出一个结束符，Transformer 就完成了所有的输出。Decoder 就像 Encoder 那样，从下往上一层一层地输出结果，每一步的输出都会输入到下面的第一个解码器。和编码器的输入一样，<font color='red'>我们把解码器的输入向量，也加上位置编码向量，来指示每个词的位置。</font>

&#8195;&#8195;解码器中的 Self Attention 层，和编码器中的 Self Attention 层不太一样：在解码器里，Self Attention 层只允许关注到输出序列中早于当前位置之前的单词。具体做法是：<font color='red'>在 Self Attention 分数经过 Softmax 层之前，屏蔽当前位置之后的那些位置。</font>所以decoder-block的第一层应该叫**masked -Self Attention**。
![masked-self-attention](https://img-blog.csdnimg.cn/0abb645f43b040ec9331d802a85393cc.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;这个屏蔽（masking）经常用一个矩阵来实现，称为 attention mask。想象一下有 4 个单词的序列（例如，机器人必须遵守命令）。在一个语言建模场景中，这个序列会分为 4 个步骤处理--每个步骤处理一个词（假设现在每个词是一个 token）。由于这些模型是以 batch size 的形式工作的，我们可以假设这个玩具模型的 batch size 为 4，它会将整个序列作（包括 4 个步骤）为一个 batch 处理。
![masked矩阵](https://img-blog.csdnimg.cn/1422b0616fdc49268053715ee1b50eeb.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
<center>图：masked矩阵<center>

&#8195;&#8195;在矩阵的形式中，我们把 Query 矩阵和 Key 矩阵相乘来计算分数。让我们将其可视化如下，不同的是，我们不使用单词，而是使用与格子中单词对应的 Query 矩阵（或Key 矩阵）。
![Query矩阵](https://img-blog.csdnimg.cn/1ace92aea0384b6da21a05a52cc7a168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
<center>图：Query矩阵<center>

&#8195;&#8195;在做完乘法之后，我们加上一个==下三角形的 attention mask矩阵==。它将我们想要屏蔽的单元格设置为负无穷大或者一个非常大的负数（例如 GPT-2 中的 负十亿）：
![加上attetnion的mask](https://img-blog.csdnimg.cn/5dc4004f22ad48e5a305ba8c5fc84f23.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
<center>图：加上attetnion的mask<center>

&#8195;&#8195;然后对每一行应用 softmax，会产生实际的分数，我们会将这些分数用于 Self Attention。
![图：softmax](https://img-blog.csdnimg.cn/9dc933a18c104034a2b96f3abe91e544.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
<center>图：softmax<center>

这个分数表的含义如下：
&#8195;&#8195;当模型处理数据集中的第 1 个数据（第 1 行），其中只包含着一个单词 （robot），它将 100% 的注意力集中在这个单词上。
&#8195;&#8195;当模型处理数据集中的第 2 个数据（第 2 行），其中包含着单词（robot must）。当模型处理单词 must，它将 48% 的注意力集中在 robot，将 52% 的注意力集中在 must。
&#8195;&#8195;诸如此类，继续处理后面的单词。在文末，会加一些更多关于mask 预训练模型的介绍。

&#8195;&#8195;Encoder-Decoder Attention层的原理和多头注意力（multiheaded Self Attention）机制类似，不同之处是：Encoder-Decoder Attention层是使用前一层的输出来构造 Query 矩阵，而 Key 矩阵和 Value 矩阵来自于编码器最终的输出。
#### 4.5 最后的线性层和 Softmax 层
&#8195;&#8195;Decoder 最终的输出是一个向量，其中每个元素是浮点数。输出向量经过Softmax 层后面的线性层（普通的全连接神经网络）映射为一个更长的向量，这个向量称为 logits 向量。

&#8195;&#8195;现在假设我们的模型有 10000 个英语单词（模型的输出词汇表），这些单词是从训练集中学到的。因此logits 向量有 10000 个数字，每个数表示一个单词的得分。经过Softmax 层归一化后转换为概率。概率值最大的词，就是这个时间步的输出单词。
![在这里插入图片描述](https://img-blog.csdnimg.cn/4ba7d7bd4b5f4fc8a9a4b6ff47ccef7e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;Transformer 是深度学习的集大成之作,融合了多项实用技术,不仅在自然语言处理领域的许多问题中得到了应用,在计算机视觉、推荐系统等领域也得到了广泛应用。
&#8195;&#8195;但是Transformer也有一个缺点，就是<font color='red'>参数量过大</font>。三个角色映射矩阵、多头注意力机制，FFNN，以及多个block的堆叠，导致一个实用的Transformer含有巨大的参数量,模型变得不容易训练，尤其是数据集小的时候。基于这种情况，BERT应运而生。

#### 4.6 Transformer 的输入
&#8195;&#8195;和通常的 NLP 任务一样，我们首先会使用词嵌入算法（embedding algorithm），将每个词转换为一个词向量。实际中向量一般是 256 或者 512 维。整个输入的句子是一个向量列表，其中有 n个词向量。

&#8195;&#8195;在实际中，每个句子的长度不一样，我们会取一个适当的值，作为向量列表的长度。如果一个句子达不到这个长度，那么就填充全为 0 的词向量；如果句子超出这个长度，则做截断。<font color='red'>句子长度是一个超参数，通常是训练集中的句子的最大长度。</font>

&#8195;&#8195;编码器中，每个位置的词都经过 Self Attention 层，得到的每个输出向量都单独经过前馈神经网络层，每个向量经过的前馈神经网络都是一样的。第一 个层 编码器的输入是词向量，而后面的编码器的输入是上一个编码器的输出。
### 5. Transformer 的训练过程
&#8195;&#8195;假设输出词汇只包含 6 个单词（“a”, “am”, “i”, “thanks”, “student”, and “<eos>”（“<eos>”表示句子末尾））。我们模型的输出词汇表，是在训练之前的数据预处理阶段构造的。当我们确定了输出词汇表，我们可以用向量来表示词汇表中的每个单词。这个表示方法也称为 one-hot encoding
#### 5.1 损失函数
&#8195;&#8195;用一个简单的例子来说明训练过程，比如：把“merci”翻译为“thanks”。这意味着我们希望模型最终输出的概率分布，会指向单词 ”thanks“（在“thanks”这个词的概率最高）。但模型还没训练好，它输出的概率分布可能和我们希望的概率分布相差甚远。
&#8195;&#8195;由于模型的参数都是随机初始化的。模型在每个词输出的概率都是随机的。<font color='red'>我们可以把这个概率和正确的输出概率做对比，然后使用反向传播来调整模型的权重，使得输出的概率分布更加接近正确输出。比较概率分布的差异可以用交叉熵。</font>
&#8195;&#8195;在实际中，我们使用的句子不只有一个单词。例如--输入是：“je suis étudiant” ，输出是：“i am a student”。这意味着，我们的模型需要输出多个概率分布，满足如下条件：
&#8195;&#8195;每个概率分布都是一个向量，长度是 vocab_size（我们的例子中，向量长度是 6，但实际中更可能是 30000 或者 50000）
第一个概率分布中，最高概率对应的单词是 “i”
第二个概率分布中，最高概率对应的单词是 “am”
以此类推，直到第 5 个概率分布中，最高概率对应的单词是 “”，表示没有下一个单词了。
#### 5.2 贪婪解码和集束搜索
&#8195;&#8195;<font color='red'>贪婪解码（greedy decoding）：模型每个时间步只产生一个输出，可以认为：模型是从概率分布中选择概率最大的词，并丢弃其他词。</font>
&#8195;&#8195;<font color='red'>集束搜索(beam search)：每个时间步保留两个最高概率的输出词</font>，然后在下一个时间步，重复执行这个过程：假设第一个位置概率最高的两个输出的词是”I“和”a“，这两个词都保留，然后根据第一个词计算第二个位置的词的概率分布，再取出 2 个概率最高的词，对于第二个位置和第三个位置，我们也重复这个过程。
&#8195;&#8195;在我们的例子中，<font color='red'>beam_size 的值是 2（含义是：在所有时间步，我们保留两个最高概率），top_beams 的值也是 2（表示我们最终会返回两个翻译的结果）。beam_size 和 top_beams 都是你可以在实验中尝试的超参数。</font>
### 6. 花式Mask预训练（我悟了）
本节选自苏剑林的文章：[《从语言模型到Seq2Seq：Transformer如戏，全靠Mask》](https://spaces.ac.cn/archives/6933#%E8%8A%B1%E5%BC%8F%E9%A2%84%E8%AE%AD%E7%BB%83)

**背景**
&#8195;&#8195;从Bert、GPT到XLNet等等，各种应用transformer结构的模型不断涌现，有基于现成的模型做应用的，有试图更好地去解释和可视化这些模型的，还有改进架构、改进预训练方式等以得到更好结果的。总的来说，这些以预训练为基础的工作层出不穷，有种琳琅满目的感觉
#### 6.1 单向语言模型 
&#8195;&#8195;语言模型可以说是一个无条件的文本生成模型（文本生成模型，可以参考[《玩转Keras之seq2seq自动生成标题》](https://spaces.ac.cn/archives/5861)）。单向语言模型相当于把训练语料通过下述条件概率分布的方式“记住”了：
p(x1,x2,x3,…,xn)=p(x1)p(x2|x1)p(x3|x1,x2)…p(xn|x1,…,xn−1)
&#8195;&#8195;我们一般说的“语言模型”，就是指单向的（更狭义的只是指正向的）语言模型。==语言模型的关键点是要防止看到“未来信息”==。如上式，预测x1的时候，是没有任何外部输入的；而预测x2的时候，只能输入x1，预测x3的时候，只能输入x1,x2；依此类推。
![单向语言模型](https://img-blog.csdnimg.cn/8f13a78a86514fd78cc77b28520d04b3.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;RNN模型是天然适合做语言模型的，因为它本身就是递归的运算；**如果用CNN来做的话，则需要对卷积核进行Mask，即需要将卷积核对应右边的部分置零。如果是Transformer呢？那需要一个下三角矩阵形式的Attention矩阵，并将输入输出错开一位训练：**
![下三角矩阵](https://img-blog.csdnimg.cn/f5c140f2cc6e48a09da84954bd7ec656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;如图所示，Attention矩阵的每一行事实上代表着输出，而每一列代表着输入，而**Attention矩阵就表示输出和输入的关联**。假定白色方格都代表0，那么第1行表示“北”只能跟起始标记$<s>$相关了，而第2行就表示“京”只能跟起始标记$<s>$和“北”相关了，依此类推。（（Mask的实现方式，也可以参考[《“让Keras更酷一些！”：层中层与mask》](https://spaces.ac.cn/archives/6810#Mask)）
####  6.2 Transformer专属 
&#8195;&#8195;事实上，除了单向语言模型及其简单变体掩码语言模型之外，UNILM的Seq2Seq预训练、XLNet的乱序语言模型预训练，基本可以说是专为Transformer架构定制的。说白了，如果是RNN架构，根本就不能用乱序语言模型的方式来预训练。至于Seq2Seq的预训练方式，则必须同时引入两个模型（encoder和decoder），而无法像Transformer架构一样，可以一个模型搞定。

&#8195;&#8195;**这其中的奥妙主要在Attention矩阵之上**。Attention实际上相当于将输入两两地算相似度，这构成了一个$n^2$大小的相似度矩阵（即Attention矩阵，n是句子长度，本节的Attention均指Self Attention），这意味着它的空间占用量是O($n^2$)量级，相比之下，RNN模型、CNN模型只不过是O(n)，所以实际上Attention通常更耗显存。
&#8195;&#8195;然而，有弊也有利，更大的空间占用也意味着拥有了更多的可能性，==我们可以通过往这个O($n^2$)级别的Attention矩阵加入各种先验约束==，使得它可以做更灵活的任务。说白了，也就只有纯Attention的模型，才有那么大的“容量”去承载那么多的“花样”。

而==加入先验约束的方式，就是对Attention矩阵进行不同形式的Mask==，这便是本文要关注的焦点。

#### 6.3 乱序语言模型
&#8195;&#8195;乱序语言模型是XLNet提出来的概念，它主要用于XLNet的预训练上。
&#8195;&#8195;乱序语言模型跟语言模型一样，都是做条件概率分解，但是乱序语言模型的分解顺序是随机的：
$$p(x1,x2,x3,…,xn)
=p(x1)p(x2|x1)p(x3|x1,x2)…p(xn|x1,x2,…,xn−1)
=p(x3)p(x1|x3)p(x2|x3,x1)…p(xn|x3,x1,…,xn−1)
=…
=p(xn−1)p(x1|xn−1)p(xn|xn−1,x1)…p(x2|xn−1,x1,…,x3)$$
&#8195;&#8195;总之，x1,x2,…,xn任意一种“出场顺序”都有可能。原则上来说，每一种顺序都对应着一个模型，所以原则上就有n!个语言模型。而基于Transformer的模型，则可以将这所有顺序都做到一个模型中去！
&#8195;&#8195;**实现某种特定顺序的语言模型，就将原来的下三角形式的Mask以某种方式打乱**。正因为Attention提供了这样的一个n×n的Attention矩阵，我们才有足够多的自由度去以不同的方式去Mask这个矩阵，从而实现多样化的效果。

&#8195;&#8195;以“北京欢迎你”的生成为例，假设随机的一种生成顺序为“$<s>$ → 迎 → 京 → 你 → 欢 → 北 → <e>”，那么我们只需要用下图中第二个子图的方式去Mask掉Attention矩阵，就可以达到目的了：
![l乱序语言模型](https://img-blog.csdnimg.cn/2292a816c044400492613f9e2a105b9b.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;跟前面的单向语言模型类似，第4行只有一个蓝色格，表示“迎”只能跟起始标记$<s>$相关，而第2行有两个蓝色格，表示“京”只能跟起始标记$<s>$和“迎”相关，依此类推。直观来看，这就像是把单向语言模型的下三角形式的Mask“打乱”了。

&#8195;&#8195;有人会问，打乱后的Mask似乎没看出什么规律呀，难道每次都要随机生成一个这样的似乎没有什么明显概率的Mask矩阵？事实上有一种更简单的、数学上等效的训练方案，即==在输入层面进行打乱==。
&#8195;&#8195;==纯Attention的模型本质上是一个无序的模型==，它里边的词序实际上是通过Position Embedding加上去的。也就是说，我们输入的不仅只有token本身，还包括token所在的位置id；再换言之，你觉得你是输入了序列“[北, 京, 欢, 迎, 你]”，实际上你输入的是集合“{(北, 1), (京, 2), (欢, 3), (迎, 4), (你, 5)}”。
&#8195;&#8195;既然只是一个集合，跟顺序无关，那么我们完全可以换一种顺序输入，比如刚才的“$<s>$→ 迎 → 京 → 你 → 欢 → 北 → $<e>$”，我们可以按“(迎, 4), (京, 2), (你, 5), (欢, 3), (北, 1)”的顺序输入，也就是说将token打乱为“迎,京,你,欢,北”输入到Transformer中，但是第1个token的position就不是1了，而是4；依此类推。这样换过来之后，Mask矩阵可以恢复为下三角矩阵。（讲了一通，乱序实现从乱序mask矩阵到乱序输入序列，有点迷。。。）
#### 6.4 Seq2Seq
&#8195;&#8195;原则上来说，任何NLP问题都可以转化为Seq2Seq来做，它是一个真正意义上的万能模型。所以如果能够做到Seq2Seq，理论上就可以实现任意任务了。
&#8195;&#8195;微软的[UNILM](https://arxiv.org/abs/1905.03197)能将Bert与Seq2Seq优雅的结合起来。能够让我们==直接用单个Bert模型就可以做Seq2Seq任务，而不用区分encoder和decoder。而实现这一点几乎不费吹灰之力——只需要一个特别的Mask==。
&#8195;&#8195;UNILM直接将Seq2Seq当成句子补全来做。假如输入是“你想吃啥”，目标句子是“白切鸡”，那UNILM将这两个句子拼成一个：[CLS] 你 想 吃 啥 [SEP] 白 切 鸡 [SEP]。经过这样转化之后，最简单的方案就是训练一个语言模型，然后输入“[CLS] 你 想 吃 啥 [SEP]”来逐字预测“白 切 鸡”，直到出现“[SEP]”为止，即如下面的左图：
![unilm](https://img-blog.csdnimg.cn/61c52c89d2484fce9a5ab71efb6350db.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;不过左图只是最朴素的方案，它把“你想吃啥”也加入了预测范围了（导致它这部分的Attention是单向的，即对应部分的Mask矩阵是下三角），事实上这是不必要的，属于==额外的约束==。真正要预测的只是“白切鸡”这部分，所以我们可以把“你想吃啥”这部分的Mask去掉，得到上面的右图的Mask。

UNILM单个Bert模型完成Seq2Seq任务的思路：
&#8195;&#8195;添加上述形状的Mask，==输入部分的Attention是双向的，输出部分的Attention是单向==，满足Seq2Seq的要求，而且==没有额外约束==。这样做不需要修改模型架构，并且还可以直接沿用Bert的Masked Language Model预训练权重，收敛更快。这符合“一Bert在手，天下我有”的万用模型的初衷，个人认为这是非常优雅的方案。
#### 6.5 实验 
&#8195;&#8195;事实上，上述的这些Mask方案，基本上都已经被集成在原作者写的[bert4keras](https://spaces.ac.cn/archives/6915)，读者可以直接用bert4keras加载bert的预训练权重，并且调用上述Mask方案来做相应的任务。具体UNILM实现例子，可以参考[原文](https://spaces.ac.cn/archives/6933#%E8%8A%B1%E5%BC%8F%E9%A2%84%E8%AE%AD%E7%BB%83)
#### 6.6 总结
&#8195;&#8195;1.原始的seq2seq训练的是一个单向语言模型，语言模型的关键点是要防止看到“未来信息”。这一点可以通过循环神经网络的递归计算来实现，比如RNN。也可以通过CNN来做，只需要对卷积核进行Mask，即需要将卷积核对应右边的部分置零。如果是Transformer呢，那就需要一个下三角矩阵形式的Attention矩阵（表示输入与输出的关联）来实现。
&#8195;&#8195;2.不仅如此，通过Attention矩阵的不同Mask方式，还可以实现乱序语言模型和Seq2Seq。
&#8195;&#8195;前者只需要乱序原来的下三角形式的Masked-Attention矩阵（也等价于乱序输入序列），后者通过句子补全来做（类似输入一个词，预测接下来会输入的词，即输入法预测）。具体做的时候，只需要mask输入部分就行（感觉就是GPT2）。
&#8195;&#8195;3.之所以一个transformer结构能搞出后面那么多花样的玩法（Bert、GPT、XLNet等），==关键在于Attention矩阵==。Attention实际上相当于将输入两两地算相似度，这构成了一个$n^2$大小的相似度矩阵（复杂度O($n^2$)）。比起RNN、CNN模型只是O(n)，Attention通常更耗显存。
&#8195;&#8195;但正因如此，却也有了更多的可能性。==通过往O($n^2$)级别的Attention矩阵加入各种先验约束，使得它可以做更灵活的任务。这种先验约束就是mask玩法==。说白了，也就只有纯Attention的模型，才有那么大的“容量”去承载那么多的“花样”。（读到这里，我悟了）。


