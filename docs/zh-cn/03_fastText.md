## fastText

<!-- fastText分类 -->
<!-- https://www.jiqizhixin.com/articles/2018-12-03-6 -->
<!-- https://www.jianshu.com/p/e8be2a27233b  paper的翻译-->
<!-- https://mp.weixin.qq.com/s/F_6liXbQYOsdeBwGzgJrEw -->
<!-- https://mp.weixin.qq.com/s/sis0AOACxTuqX5AYTKOwJA -->
<!-- https://www.cnblogs.com/DjangoBlog/p/7904039.html -->
<!-- https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650716942&idx=3&sn=0d48c0218131de502ac5e2ef9b700967 -->

<!-- https://github.com/facebookresearch/fastText -->
<!-- https://github.com/salestock/fastText.py  封装好的py API-->

<!-- https://github.com/d2l-ai/d2l-zh/tree/master/chapter_natural-language-processing -->

<!-- fastText 词向量 -->

<!-- https://blog.csdn.net/sinat_26917383/article/details/54850933 -->
<!-- https://blog.csdn.net/qq_32023541/article/details/80899874 -->
<!-- https://blog.csdn.net/ACM_hades/article/details/105258695 -->

fastText是Facebook, 2016-2017年开发的一款快速文本分类器，提供简单而高效的文本分类和词表征学习的方法，不过这个项目其实是有两部分组成的，一部分是这篇文章介绍的
fastText 文本分类（paper：A. Joulin, E. Grave, P. Bojanowski, T. Mikolov,
Bag of Tricks for Efficient Text
Classification（高效文本分类技巧）），
另一部分是词嵌入学习（paper:P.
Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors
with Subword
Information（使用子词(字)信息丰富词汇向量））。
按论文来说只有文本分类部分才是 fastText(所以基本网上资源在介绍fastText仅是指文本分类部分），但也有人把这两部分合在一起称为
fastText。笔者，在这即认为词嵌入学习属于fastText项目。

github链接：https://github.com/facebookresearch/fastText


### 1. Enriching Word Vectors with Subword Information(使用子词信息丰富词汇向量)

fastText词向量模型是在skip-gram模型的基础上提出来的，关于skip-gram模型请参考word2vec章节的介绍。在skip-gram中对词典中的每个词$w$对应两个向量：
+ 输入向量$u_w$：是输入层到隐藏层的链接矩阵$W\in R^{V\times N}$的行向量
+ 输出向量$v_t$：是隐藏层到输出层的链接矩阵$W^{'}\in R^{N\times V}$的列向量

fastText模型与skip-gram模型相同的部分：

+ fastTex模型与skip-gram模型隐藏层到输出层部分(即后半部分) 是一样的结构，都是一个将隐藏层状态向量$h_t$输出到softmax层得到词汇表各词的预测概率。
+ 训练目标是一样的都是用当前词$w_t$预测其上下文词集$C_t$
+ softmax层也都是使用负采样softmax层或者分层softmax层进行优化。

fastText模型与skip-gram模型不相同的部分：

+ fastTex模型与skip-gram模型区别在于：输出层到隐藏层部分(前部)，即得到隐藏层状态向量$h_t$方式
+ skip-gram模型：将当前词$w_t$的one-hot编码与连接矩阵$W∈R^{V\times N}$相乘，得到词$w_t$的输入向量$u_{w_t }$作为隐藏层状态向量$h_t$ ，即$h_t=u_{w_t }$
+ fastTex模型:将当前词的$w_t$和该词的字符级的n-grams的one-hot编码相加，再将这个和与连接矩阵$W∈R^{V×N}$相乘，得到隐藏层状态向量$h_t$，该向量就是我们最终得到词$w_t$的词向量(即fastTex模型的词向量)。计算隐藏层状态向量$h_t$的细节下面进行详细解释。

​字符级n-grams:

下面举例子来说明符级n-grams(character n-grams):求词 **where** 的n-grams

+ 在**where**前后加上 开始符`<` 和 结束符`>`,于是得到 `<where>`
+ 我们取n-grams中`n=3`,得到 where 5个字符级tri-gram如下： `<wh,whe,her,ere,re>`
+ 那么 where 对应6个一个where自身词和5个子词(Subword)：`<where>,<wh,whe,her,ere,re>`

他们都有自己对应输入向量u,将它们的输入向量求和就得到了词where的隐藏层状态向量$h_{where}$,也就是词 where 的词向量。

fastText模型的隐藏层计算方法:

论文n-grams中n不是简单的取3，而是分别取3，4，5，6；这样可以得到更多的字符级n-grams(也叫子词),下面讲述fastTex模型输出层到隐藏层的结构:

+ 输入层词汇表$D_{in}$（输入层使用的词汇表）:对于词汇表(词典)D的每个词我们分别对其进行字符级n-grams提取并将这个字符级n-grams和原词一起加入输入层词汇表$D_{in}$.
+ 比如对于词汇表D中的where词：原词：`<where>`;3-grams: ` <wh,whe,her,ere,re>`;4-grams: `<wh,whe,her,ere,re>`;5-grams: `<wher,where,here>`;6-grams: `<where,where>`
+ 然后将他们都加入输入层词汇表$D_{in}$
+ 显然$D_{in}$比$D$要大，我们将元词汇表$D$叫做输出层词汇表$D_{out}$
+ 因为输入层词汇表的改变所以输入层到隐藏层的连接矩阵由$W\in R^{|D|\times N}$变为$W\in R^{|D_{in}|\times N}$,$W$的行有些是某个词（比如where)的时输入向量（$u_{where}$)，有些是字符级n-grams（子词）(比如`<wh`)的输入向量（$u_{< wh}$)

隐藏状态$h$的计算方式：

+ 首先获取当前输入出 where 的字符级n-grams：`<wh,whe,her,ere,re>,<wh,whe,her,ere,re>,<wher,where,here>,<wher,where,here>`
+ 然后将原词的和字符级n-grams的one-hot编码进行累加得到输入向量$x_{where}$,即将`<where>`和`<wh,whe,her,ere,re>,<wh,whe,her,ere,re>,<wher,where,here>,<wher,where,here>`的one-hot向量相加得到$x_{where}$
+ 将$x_{where}$与连接矩阵$W\in R^{|D_{in}|\times N}$想成得到隐藏章台向量$h_{where}$,其实是将`<where>,<wh,whe,her,ere,re>,<wh,whe,her,ere,re>,<wher,where,here>,<wher,where,here>`的输入向量$u(W\in R^{|D_{in}|\times N})$中对应的行进行相加得到$h_{where}$
+ 当fastTex模型训练完成后，where的词向量就是将其输入后得到隐藏层状态向量$h_{where}$
+ 因为fastTex模型的使用字符级n-grams所以对于没有在训练集中出现的词也可以得到该词对词向量，因为这个词的字符级n-grams出现过。

训练词向量的损失函数定义：

<div align=center>
    <img src="zh-cn/img/fasttext/p1.png" /> 
</div>
<div align=center>
    <img src="zh-cn/img/fasttext/p2.png" /> 
</div>


### 2.Bag of Tricks for Efficient Text Classification（高效文本分类技巧）

这里有一点需要特别注意，一般情况下，使用fastText进行文本分类的同时也会产生词的embedding(上文介绍的过程），即embedding是fastText分类的产物。除非你决定使用预训练的embedding来训练fastText分类模型，这另当别论。


字符级别的n-gram：

word2vec把语料库中的每个单词当成原子的，它会为每个单词生成一个向量。这忽略了单词内部的形态特征，比如：`“apple”` 和`“apples”`，“达观数据”和“达观”，这两个例子中，两个单词都有较多公共字符，即它们的内部形态类似，但是在传统的word2vec中，这种单词内部形态信息因为它们被转换成不同的id丢失了。

为了克服这个问题，fastText使用了字符级别的n-grams来表示一个单词。对于单词“apple”，假设n的取值为3，则它的trigram有:

`“<ap”,  “app”,  “ppl”,  “ple”, “le>”`

其中，`<`表示前缀，`>`表示后缀。于是，我们可以用这些trigram来表示“apple”这个单词，进一步，我们可以用这5个trigram的向量叠加来表示“apple”的词向量。
这带来两点好处：

1. 对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。
2. 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。


模型架构:

fastText模型架构和word2vec的CBOW模型架构非常相似。下面是fastText模型架构图:

<div align=center>
    <img src="zh-cn/img/fasttext/p3.png" /> 
</div>

**注意**：此架构图没有展示词向量的训练过程。可以看到，和CBOW一样，fastText模型也只有三层：输入层、隐含层、输出层（Hierarchical Softmax），输入都是多个经向量表示的单词，输出都是一个特定的target，隐含层都是对多个词向量的叠加平均。

不同的是，CBOW的输入是目标单词的上下文，fastText的输入是多个单词及其n-gram特征，这些特征用来表示单个文档；CBOW的输入单词被one-hot编码过，fastText的输入特征是被embedding过；CBOW的输出是目标词汇，fastText的输出是文档对应的label。

值得注意的是，fastText在输入时，将单词的字符级别的n-gram向量作为额外的特征；在输出时，fastText采用了分层Softmax，大大降低了模型训练时间。这两个知识点在前文中已经讲过，这里不再赘述。fastText相关公式的推导和CBOW非常类似，这里也不展开了。


fastText核心思想： 现在抛开那些不是很讨人喜欢的公式推导，来想一想fastText文本分类的核心思想是什么？

仔细观察模型的后半部分，即从隐含层输出到输出层输出，会发现它就是一个softmax线性多类别分类器，分类器的输入是一个用来表征当前文档的向量；模型的前半部分，即从输入层输入到隐含层输出部分，主要在做一件事情：生成用来表征文档的向量。那么它是如何做的呢？叠加构成这篇文档的所有词及n-gram的词向量，然后取平均。叠加词向量背后的思想就是传统的词袋法，即将文档看成一个由词构成的集合。

于是fastText的核心思想就是：将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。这中间涉及到两个技巧：字符级n-gram特征的引入以及分层Softmax分类。

fastText的分类效果：为何fastText的分类效果常常不输于传统的非线性分类器？

假设我们有两段文本：

> 我 来到 达观数据

> 俺 去了 达而观信息科技

这两段文本意思几乎一模一样，如果要分类，肯定要分到同一个类中去。但在传统的分类器中，用来表征这两段文本的向量可能差距非常大。传统的文本分类中，你需要计算出每个词的权重，比如tf-idf值， “我”和“俺” 算出的tf-idf值相差可能会比较大，其它词类似，于是，VSM（向量空间模型）中用来表征这两段文本的文本向量差别可能比较大。

但是fastText就不一样了，它是用单词的embedding叠加获得的文档向量，词向量的重要特点就是向量的距离可以用来衡量单词间的语义相似程度，于是，在fastText模型中，这两段文本的向量应该是非常相似的，于是，它们很大概率会被分到同一个类中。

使用词embedding而非词本身作为特征，这是fastText效果好的一个原因；另一个原因就是字符级n-gram特征的引入对分类效果会有一些提升 。

fastText的应用：

fastText作为诞生不久的词向量训练、文本分类工具，比较深入的应用。主要被用在以下两个系统：

1. 同近义词挖掘。Facebook开源的fastText工具也实现了词向量的训练，基于各种垂直领域的语料，使用其挖掘出一批同近义词；
2. 文本分类系统。在类标数、数据量都比较大时，会选择fastText来做文本分类，以实现快速训练预测、节省内存的目的。

### 3.代码实现fastText

有很多对于fastText的实现，包括tensorflow, keras, pytorch,gensim等，本节我们介绍facebook官网提供的fastText的实现。


**词向量**

```shell
# 这里我们参考了fastText的官方文档的介绍： https://github.com/facebookresearch/fastText
# https://fasttext.cc/docs/en/support.html

# 安装
# pip install fasttext

# word representation model
import fasttext
# skipgram model
model = fasttext.train_unsupervised("data.txt",model='skipgram')
model = fasttext.train_unsupervised('data/fil9', minn=2, maxn=5, dim=300)
model = fasttext.train_unsupervised('data/fil9', epoch=1, lr=0.5)
model = fasttext.train_unsupervised('data/fil9', thread=4)

# or cbow model
model = fasttext.train_unsupervised("data.txt",model='cbow')

# data.txt utf-8编码的训练语料（中文的话分词空格隔开，英文不需要分词...）

print(model.words) # 词典
print(model['king']) # king的词向量
model.get_word_vector("the")

# 保存和加载模型
model.save_model("model_filename.bin")
# 加载模型
model = fasttext.load_model("model_filename.bin")

# 或得相近词
model.get_nearest_neighbors('asparagus')
model.get_analogies("berlin", "germany", "france")

# 新词词向量计算， gearshift不在词典中
model.get_nearest_neighbors('gearshift')
```

**文本分类**

```python
import fasttext

model = fasttext.train_supervised("data.train.txt")
model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2)
model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs') # -loss hs(分层softmax)
model = fasttext.train_supervised(input="cooking.train", lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova') # -loss one-vs-all
# data.train.txt is a text file containing a training sentence per line along with the labels.
# By default, we assume that labels are words that are prefixed by the string __label__

print(model.words)
print(model.labels)

# 模型评价
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test('test.txt'))

# 模型预测
model.predict("Which baking dish is best to bake a banana bread ?")
model.predict("Which baking dish is best to bake a banana bread ?", k=3)
model.predict(["Which baking dish is best to bake a banana bread ?", "Why not put knives in the dishwasher?"], k=3)

# 压缩模型文件
# with the previously trained `model` object, call :
model.quantize(input='data.train.txt', retrain=True)

# then display results and save the new model :
print_results(*model.test(valid_data))
model.save_model("model_filename.ftz")
# model_filename.ftz will have a much smaller size than model_filename.bin.
```