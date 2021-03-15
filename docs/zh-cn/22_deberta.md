## DeBerta: Decoding-Enhanced BERT with Disentangled Attention(分散注意力)

<!-- https://yam.gift/2020/06/27/Paper/2020-06-27-DeBERTa/
https://baijiahao.baidu.com/s?id=1688306953815332609&wfr=spider&for=pc
https://blog.csdn.net/doyouseeman/article/details/114600476
https://blog.csdn.net/zephyr_wang/article/details/113776734?utm_medium=distribute.pc_relevant.none-task-blog-OPENSEARCH-2.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-2.control
https://blog.csdn.net/qq_27590277/article/details/113706443?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242 -->

<!-- https://tech.ifeng.com/c/82rY5GraNHR -->
<!-- https://deberta.readthedocs.io/en/latest/index.html -->

<div align=center>
    <img src="zh-cn/img/deberta/p0.png" /> 
</div>



<!-- 清楚的讲解
https://facico.blog.csdn.net/article/details/114600476?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-2.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-2.control -->

<!-- https://blog.csdn.net/doyouseeman/article/details/114600476 -->

<!-- https://yam.gift/2020/06/27/Paper/2020-06-27-DeBERTa/ -->

<!-- https://blog.csdn.net/zephyr_wang/article/details/113776734?utm_medium=distribute.pc_relevant.none-task-blog-OPENSEARCH-2.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-OPENSEARCH-2.control -->

<!-- https://facico.blog.csdn.net/article/details/114600476?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-2.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-2.control -->

<!-- https://blog.csdn.net/doyouseeman/article/details/108643938 -->

<!-- https://blog.csdn.net/qq_27590277/article/details/113706443?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242 -->

<!-- https://baijiahao.baidu.com/s?id=1688306953815332609&wfr=spider&for=pc -->

<!-- 
实验结果
https://baijiahao.baidu.com/s?id=1688306953815332609&wfr=spider&for=pc -->

<!-- https://deberta.readthedocs.io/en/latest/index.html -->

去年 6 月，来自微软的研究者提出一种新型预训练语言模型 DeBERTa，该模型使用两种新技术改进了 BERT 和 RoBERTa 模型。8 月，该研究开源了模型代码，并提供预训练模型下载。最近这项研究又取得了新的进展。

微软最近通过训练更大的版本来更新 DeBERTa 模型，该版本由 48 个 Transformer 层组成，带有 15 亿个参数。本次扩大规模带来了极大的性能提升，使得单个 DeBERTa 模型 SuperGLUE 上宏平均（macro-average）得分首次超过人类（90.3 vs 89.8），整体 DeBERTa 模型在 SuperGLUE 基准排名中居于第二.

<div align=center>
    <img src="zh-cn/img/deberta/p8.png" /> 
</div>

DeBERTa 是一种基于 Transformer，使用自监督学习在大量原始文本语料库上预训练的神经语言模型。像其他 PLM 一样，DeBERTa 旨在学习通用语言表征，可以适应各种下游 NLU 任务。DeBERTa 使用 3 种新技术改进了之前的 SOTA PLM（例如 BERT、RoBERTa、UniLM），这 3 种技术是：

1. Disentangled Attention（分散注意力）
2. Enhanced Mask Decoder
3. Virtual Adversarial Training

#### 1.Disentangled Attention


BERT加入位置信息的方法是在输入embedding中加入postion embedding, pos embedding与char embedding和segment embedding混在一起，这种早期就合并了位置信息在计算self-attention时，表达能力受限，维护信息非常被弱化了

<div align=center>
    <img src="zh-cn/img/bert/figure_2.png" /> 
</div>

本文的motivation就是将pos信息拆分出来，单独编码后去content 和自己求attention，增加计算 “位置-内容” 和 “内容-位置” 注意力的分散Disentangled Attention

+ 对于序列中的位置$i$，我们使用两个向量来表示它$H_i$ 和$P_{i|j}$ 分别表示内容表示和对于j的相对位置表示.
+ 则$i,j$之间的attention score:

$$A_{ij}=\{H_i,P_{i|j}\}\times\{H_j,P_{j|i}\}=H_iH^T_{j}+H_iP^T_{j|i}+P_{i|j}H^T_{j}+P_{i|j}P^T_{j|i}$$
即，注意力权重由<内容，内容>,<内容，位置>,<位置，内容>,<位置，位置>组合成.因为这里用的是相对位置embedding,所以<位置，位置>的作用不大，将其在上式中移除.

+ 使用attention的QKV模式表示

以单头为例，标准的self-attention如下:

<div align=center>
    <img src="zh-cn/img/deberta/p1.png" /> 
</div>

定义$k$为做大相对距离，$\sigma(i,j)\in [0,2k)$为Token i到Token j的相对距离，文章中默认$k=512$

<div align=center>
    <img src="zh-cn/img/deberta/p2.png" /> 
</div>

即全部编码到$[0,2k-1]$，自己为$k$. 所以，具有相对位置偏差的disentangled self-attention

<div align=center>
    <img src="zh-cn/img/deberta/p3.png" /> 
</div>

注意,$Q$那里的$\sigma$是$(j,i)$，因为这里<位置,内容>是$j$的内容，相对于$i$的位置

<div align=center>
    <img src="zh-cn/img/deberta/p4.png" /> 
</div>


#### 2.Enhanced Mask Decoder(EMD)

+ 用EMD代替原BERT的Softmax层预测遮盖的Token
+ 分散注意力机制只考虑了内容和相对位置，并没有考虑绝对位置，但绝对位置也很重要，比如“a new store opened beside the new mall”中的store和mall，他们局部上下文相似，但在句子中扮演着不同的角色，这种角色的不同很大程度上决定于绝对位置. 因此在decoder(这里的decoder不是说的Transformer中的Decoder)的softmax之前将单词的绝对位置嵌入.
+ 将绝对位置加入有两种方式: 一、放在input layer前（BERT),二、放在所有transformer层之后，softmax层之前.
+ DeBERTa就用的第二种方式，实验发现第二种方式比第一种方式效果好,作者推测，第一种方式可能会影响相对位置的学习.

<div align=center>
    <img src="zh-cn/img/deberta/p5.png" /> 
</div>

另外，EMD可以使其他有用的信息(类似绝对距离那样)在预训练的时候加入. 具体实现的时候，就是使用n个堆叠的EMD，每个EMD输入到下一个EMD。
n层layer共享参数来较少参数，绝对位置是第一个EMD，其他的可以放各种信息.

另外本文还做了一个改动，就是在将 Encoder 的输出喂进 Decoder 时，将 MASK Token 中 10% 不改变的 Token 编码换成了他们的绝对位置 Embedding，然后再用 MLM 预测. 因为这些 Token 虽然不会造成预训练和精调阶段的不匹配，但是却导致了信息泄露——Token 以本身为条件进行预测.


#### 3.Scale-invariant-Fine-Tuning(SiFT)

这就是他新发明的虚拟对抗训练方式，但说的非常简略.

在fine-tunning时，使用layer normalization

+ 先将embedding进行normalize
+ 再将扰动并进normalize之后的embedding
+ 发现这样会大大提高fine-tunning的性能.


#### 4.实验与结论

大模型与集成：

+ 15亿的参数，首次在superGLUE上超过人类水平
+ 集成后又比15亿多了0.4

DeBERTa_1.5B： 48层，hidden 大小等于1536，24个注意力头。训练数据有160G.
不过T5有11 billion参数，DeBERTa_1.5B参数量还是很小的，效果也更好。如下表：

<div align=center>
    <img src="zh-cn/img/deberta/p7.png" /> 
</div>


实验：

+ 效果不多说，目前GLUE榜第一，NLU上基本SOTA
+ 消融实验也做了，主要做的是上面的<位置，内容>，<内容，位置>...和有没有EMD，实验证明目前组合最好

attention可视化:

<div align=center>
    <img src="zh-cn/img/deberta/p6.png" /> 
</div>

RoBERTa: 有明显的对角线效果，说明token会关注自己比较明显
竖直条纹主要是因为高频词（“a”,“the”,标点符号这些）引起的

DeBERTa: 对角线并不明显，主要是EMD起的作用
竖直条纹主要在第一列，说明[CLS]的主导性是可取的，因为[CLS]这个向量经常作为特征载体在下游任务中作为输入序列的上下文表示。（[CLS]）
