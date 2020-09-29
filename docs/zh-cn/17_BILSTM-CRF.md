## Bidirectional LSTM-CRF Models for Sequence Tagging

<!-- https://blog.csdn.net/weixin_40485502/article/details/104071471 -->
<!-- https://blog.csdn.net/weixin_42295205/article/details/105350882 -->

<!-- https://search.bilibili.com/all?keyword=LSTM-CRF&from_source=nav_search&spm_id_from=333.851.b_696e7465726e6174696f6e616c486561646572.10 -->

<!-- https://www.bilibili.com/video/BV1K54y117yD?from=search&seid=15091890416270583666 -->

### 1.序列标注的方式

序列标记包括词性标注(POS)、分块和命名实体识别(NER)等，一直是经典的NLP任务。几十年来，引起了研究界的广泛关注。序列标注任务的输出可支撑下游任务。

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p1.png" /> 
</div>

我们着重介绍命名实体识别，命名实体识别有常用的4种标注方式：

**1.BMES(四位序列标注法)**

B表示一个词的首位词，M表示一个词的中间位置，E表示一个词的末尾位置，S表示一个单独的字词
```
我/S 是/S 中/B国/M 人/E
我/是/中国人/(标注上分出来的实体块)
```

**2.IO**

I-X代表实体X，O代表不属于任何类型

**3.BIO(三位序列标注法)**

B-begin,I-inside,O-outside, B-X代表实体X的开头，I-X代表实体的结尾，O代表不属于任何类型

**4.BIOES(四位序列标注法)**

B-begin,I-inside,O-outside,E-end,S-single，B表示开始，I表示内部，O代表非实体，E代表实体尾部，S代表该词本身就是一个实体。

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p2.png" /> 
</div>


### 2.命名实体识别的发展历程

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p3.png" /> 
</div>

+ 统计模型

	- Hidden Markov Models (HMM),
	- Maximum entropy Markov models (MEMMs) (McCallum et al.,2000)
	- Conditional Random Fields (CRF)(Lafferty et al.， 2001)

+ 神经网络
	- 基于卷积网络的模型(Collobert et al.， 2011)
		- Conv-CRF等模型,因为它包含一个卷积网络和CRF层输出(这个词的句子级别loglikelihood (SSL)是用于原始论文)。
		- Conv-CRF模型产生了有前景的结果序列标记任务。
	- 递归神经网络(Mesnil et al ., 2013;Yao et al ., 2014)
	- 基于卷积网(Xu and Sarikaya, 2013)


### 3.HMM在NER中的应用

+ HMM的数学定义

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p4.png" /> 
</div>

+ HMM的初始隐状态概率

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p5.png" /> 
</div>

+ HMM的概率转移

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p6.png" /> 
</div>

+ NMM的发射概率

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p7.png" /> 
</div>

+ HMM的3个基本问题

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p8.png" /> 
</div>

+ 极大似然估计进行监督学习

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p9.png" /> 
</div>

+ 用HMM解决序列标注的问题

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p10.png" /> 
</div>

> 关于HMM我们有详细的细节推导版本的教程，如果需要，我们将共享给读者.

### 4.CRF在NER中的应用

> 关于CRF的细节请参考本系列教程的[CRF章节](zh-cn/09_CRF.md).

+ 线性链条件随机场的定义

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p11.png" /> 
</div>

+ 特征函数

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p12.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p13.png" /> 
</div>

+ CRF参数化的形式

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p14.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p15.png" /> 
</div>

!> CRF VS HMM

<div align=center>
    <img src="zh-cn/img/bilstm_crf/p16.png" /> 
</div>


### 5.Bi-LSTM-CRF在NER中的应用







### 6.Code Example