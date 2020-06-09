## LSA(LSI),pLSA(pLSI)

<!-- https://blog.csdn.net/pipisorry/article/details/42560693 -->
<!-- https://www.cnblogs.com/sancallejon/p/4963630.html -->
<!-- https://www.sohu.com/a/234584362_129720 -->

<!-- https://blog.csdn.net/fkyyly/article/details/84665361 -->
<!-- https://zhuanlan.zhihu.com/p/84788824-->

<!-- https://www.cnblogs.com/yalphait/articles/8685586.html -->
<!-- https://www.cnblogs.com/datalab/p/3163692.html -->

<div align=center>
    <img src="zh-cn/img/lsa/p1.png" /> 
</div>

我们将着重给读者详细介绍LSA,pLSA和LDA，关于HDP读者可以根据自己兴趣自行学习。关于本节介绍的LSA和pLSA算法主要参考李航老师的《统计学习方法》第二版中的第17章，18章及相关网络配图。

潜在语义分析（LSA）是一种无监督学习的方法，主要用于文本的主题分析，其特点是通过矩阵分解发现文本与单词之间的基于主题的语义关系。潜在语义分析由Deerwester等于1990年提出，最初应用于文本信息检索，所以也被称之为潜在语义索引（LSI),在推荐系统，图像处理，生物信息学等领域也有广泛的应用。

文本信息处理中，传统的方法以单词向量表示文本的语义内容，以单词向量空间的度量表示文本的相似度。潜在语义分析意在解决这种方法不能准确表示语义的问题。试图从大量的文本数据中发现潜在的主题，以主题向量表示文本的语义内容。以主题向量的空间度量更准确的表示文本之间的语义相似度。这也是主题分析（topic model)的基本思想。

潜在语义分析使用的是非概率的主题分析模型。具体的，将文本集合表示为单词-文本矩阵，对单词-文本矩阵进行奇异值分解，从而得到主题向量空间，以及文本在主题向量空间的表示。奇异值分解（SVD）分解，其特点是分解的矩阵正交。

非负矩阵分解（NMF)是另一种矩阵因子分解额方法，其特点是分解的矩阵非负。1999年Lee和Sheung的论文发表之后，非负矩阵分解引起了高度的重视和广泛的应用，非负矩阵分解也可以用于主题分析。


### 1.单词向量空间和主题向量空间

#### 1.单词向量空间

文本信息处理，比如文本信息检索，文本数据挖掘中的一个核心是对文本语义内容进行表示，并进行文本之间的语义相似度的计算。最简单的方法是利用向量空间模型（vector space model VSM),也就是单词空间向量模型（word vector space model).向量空间模型的基本思想是，给定一个文本，用一个向量表示该文本的“语义”，向量的每一维对应一个单词，其数值为该单词在该文本中出现的频数或权重；基本假设是文本中所有单词的出现情况表示了文本的语义内容；文本集合中的每个文本都表示为一个向量，存在于一个向量空间；向量空间的度量，如內积或标准化內积表示文本之间的“语义相似度”

例如，文本信息检索的任务是，用户提出查询时，帮助用户找到与查询最相关的文本，以排序的方式展示给用户。一个最简单的做法是次用单词向量空间模型，将查询与文本表示为为单词向量，计算查询向量与文本向量的内机，作为语义相似度，以这个相似度的高低对文本进行排序。在这里查询被看成伪文本，查询与文本的相似度表示查询与文本的相关性。

下面给出严格的定义。给定一个含$n$个文本的结合$D={d_1,d_2,...,d_n}$,以及在所有文本中出现的$m$个单词的集合$W={w_1,w_2,...,w_m}$.将单词在文本中出现的数据用一个单词-文本矩阵（word-document matrix)表示，记作$X$

<div align=center>
    <img src="zh-cn/img/lsa/p2.png" /> 
</div>

这是一个$m\times n$ 矩阵，元素$x_{ij}$表示单词$w_i$在文本.$d_j$内中出现的频数或权值。由于单 词的种类很多，而每个文本中出现单词的种类通常较少，所以单词-文本矩阵是一个稀疏矩阵。权值通常用单词频率-逆文本频率（term frequency-inverse document frequency, TF-IDF）表示，其定义是可以参考词向量的相关章节。

单词向量空间模型直接使用单词-文本矩阵的信息。单词-文本矩阵的第$j$列向量$x_j$表示文本$d_j$

<div align=center>
    <img src="zh-cn/img/lsa/p3.png" /> 
</div>

其中$x_{ij}$表示单词$w_i$在文本$d_j$的权值，权值越大，该单词在该文本中的重要度就越高。矩阵$X$也可以写作$X=[x_1,x_2,...,x_n]$。

两个单词向量的内积或标准化内积（余弦）表示对应的文本之间的语义相似度，因此，文本$d_i$ 与$d_j$之间的相似度为
$$x_i. x_j=\frac{x_i. x_j}{||x_i||||x_j||}$$
直观上，在两个文本中共同出现的单词越多，其语义内容就越相近，对应的单词向量同不为零的维度就越多，内积就越大（单词向量元素的值都是非负的），表示两个文本在语义内容上越相似.

单词向量空间的优缺点：

+ 模型简单
+ 计算效率高
+ 局限性，内积相似度未必能准确表达两个文本的语义相似度，特别是中文会存在一词多义，多词一义。

#### 2.主题向量空间

两个文本的相似度可以体现在两者的主题相似度上，所谓主题(topic)并没有严格的定义，就是指文本所讨论的中心内容。有个文本一般含有若干个主题，如果两个文本的主题相似，那么两者的语义也应该是相似的。主题可以由若干个语义相关的单词表示，同义词比如“airplane”和“aircraft”可以表示同一个主题，而多义词“apple”可以表示不同的主题。这样基于主题的模型就可以解决上述基于单词的模型存在的问题。

可以设想定义一种主题向量空间模型（topic vector space model)。给定一个文本，用主题空间的一个向量表示该文本，该向量的每一分量对应一个主题，其数值为该主题在文本中出现的权值。用两个向量的内积表示对应两个文本的语义相似度。注意主题的个数通常远远小于单词的个数，主题空间向量更加抽象。事实上潜在语义分析（LSA)正是构建主题向量空间的方法,单词向量空间模型和主题向量空间模型可以互补，现实中，两者可以同时使用。

给定一个文本集合$D={d_1,d_2,...,d_n}$和一个相应的单词集合$W={w_1,w_2,...,w_m}$。可以获得其单词-文本矩阵$X$，$X$构成原始的单词向量空间，每一列是一个文本 在单词向量空间中的表示.矩阵$X$也可以写作$X=[x_1,x_2,...,x_n]$
<div align=center>
    <img src="zh-cn/img/lsa/p2.png" /> 
</div>

假设所有文本共含有$k$个主题题。假设每个主题由一个定义在单词集合$W$上的$m$维向量表示，称为主题向量，即 

<div align=center>
    <img src="zh-cn/img/lsa/p4.png" /> 
</div>

$t_{il}：单词$w_i$在主题$t_l$的权值，权值越大，该单词在该话题中的重要度就越高。$k$个话题向量张成一个话题向量空间(topic vector space)，维数为$k$
话题向量空间$T$是单词向量空间$X$的一个子空间,话题向量空间T也可以表示为一个矩阵，称为单词-话题矩阵（word-topic matrix)，记作

<div align=center>
    <img src="zh-cn/img/lsa/p5.png" /> 
</div>

矩阵$T$也可写作$T=[t_1,t_2,...,t_k]$

**文本在话题向量空间的表示：**

现在考虑文本集合D的文本$d_j$，在单词向量空间中由一个向量$x_j$表示，将$x_j$投影到话题向量空间$T$中，得到在话题向量空间的一个向量$y_j$, $y_j$是一个$k$维向量， 其表达式为

<div align=center>
    <img src="zh-cn/img/lsa/p6.png" /> 
</div>

$y_{lj}：文本$d_j$在话题$t_l$的权值， 权值越大，该话题在该文本中的 重要度就越高.矩阵$Y$表示话题在文本中出现的情况，称为话题-文本矩阵(topic-document matrix) ，记作

<div align=center>
    <img src="zh-cn/img/lsa/p7.png" /> 
</div>

矩阵$Y$可一个写作$Y=[y_1,...,y_n]$

**从单词向量空间到话题向量空间的线性变换：**

这样一来，在单词向量空间的文本向量$x_j$可以通过它在话题空间中的向量$y_j$近似表示，具体地由$k$个话题向量以$y_j$为系数的线性组合近似表示
$$x_j\simeq y_{1j}t_1+y_{2j}t_2+...+y_{kj}t_k,j=1,2,...,n$$
所以，单词-文本矩阵$X$可以近似的表示为单词-话题矩阵$T$与话题一文本矩阵$Y$的乘积形式。这就是潜在语义分析
$$X\simeq TY$$

直观上，潜在语义分析是将文本在单词向量空间的表示通过线性变换转换为在话题向量空间中的表示

<div align=center>
    <img src="zh-cn/img/lsa/p8.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/lsa/p9.png" /> 
</div>


### 2.LSA(LSI)

潜在语义分析利用矩阵奇异值分解,具体的潜在语义对单词-文本矩阵进行奇异值分解， 将其左矩阵作为话题向量空间，将其对角矩阵与右矩阵的乘积作为文本在话题向量空间的表示。
首先给定单词-文本矩阵：

<div align=center>
    <img src="zh-cn/img/lsa/p2.png" /> 
</div>

潜在语义分析根据确定的话题个数$k$对单词-文本矩阵$X$进行截断奇异值分解

<div align=center>
    <img src="zh-cn/img/lsa/p10.png" /> 
</div>

$U_k$是$m\times k$矩阵，它的列由$X$的前$k$个相互正交的左奇异向量组成，每一列表示一个话题，称为**话题向量空间**，$\sum_k$是$k$阶对角矩阵，它的对角元素为前$k$个最大奇异值，$V_k$是$n\times k$矩阵，它的列由$X$的前$k$个相互正交的右奇异向量组成。

有了话题向量空间，接着考虑文本在话题空间的表示

<div align=center>
    <img src="zh-cn/img/lsa/p11.png" /> 
</div>

上式中，矩阵$X$的第$j$列向量 $x_j$满足

<div align=center>
    <img src="zh-cn/img/lsa/p12.png" /> 
</div>

$(\sum_kV^T_k)_ j$是矩阵$(\sum_kV^T_k)$的第$j$列向量，矩阵的每一列元素

<div align=center>
    <img src="zh-cn/img/lsa/p13.png" /> 
</div>

是一个文本在话题向量空间的表示.

综上，可以通过对单词一文本矩阵的奇异值分解进行潜在语义分析，得到话题空间$U_k$，以及文本在话题空间的表示$(\sum_kV^T_k)$.



### 3.NMF(非负矩阵分解算法)

非负矩阵分解也可以用于话题分析。对单词一文本矩阵进行非负矩阵分解，将其左矩阵作为话题向量空间，将其右矩阵作为文本在话题向量空间的表示。(注意通常单词-文本矩阵是非负的)

给定一个非负矩阵$X≥0$，找到两个非负矩阵$W≥0$和$H≥0$，使得
$$X\simeq WH$$
即将非负矩阵$X$分解为两个非负矩阵$W$和$H$的乘积的形式，称为**非负矩阵分解**。因为$WH$与$X$完全相等很难实现，所以只要求$WH$与$X$近似相等。

+ 假设非负矩阵$X$是$m\times n$矩阵，非负矩阵$W$和$H$分别为$m\times k$矩阵和$k\times n$矩阵。
+ 假设$k< min(m, n)$，即$W$和$H$小于原矩阵$X$，所以非负矩阵分解是对原数据的压缩。

由$X \simeq WH$知，矩阵$X$的第$j$列向量$x_j$满足

<div align=center>
    <img src="zh-cn/img/lsa/p14.png" /> 
</div>

矩阵$X$的第$j$列$x_j$可以由矩阵$W$(基矩阵）的k个列$w_l$的线性组合逼近，线性组合的系数是矩阵$H$（系数矩阵）的第$j$列$h_j$的元素。非负矩阵分解旨在用较少的基向量、系数向量来表示较大的数据矩阵。

在LSA中，$W=[w_1,w_2,...,w_k]$为话题向量空间，每个列向量表示$k$个话题，$H=[h_1,h_2,...,h_n]$为文本在话题空间向量的表示，每一列表示文本集和的$n$个文本。

**非负矩阵分解的形式化**

非负矩阵分解可以形式化为最优化问题求解。首先定义损失函数或代价函数。

+ 第一种损失函数是平方损失。设两个非负矩阵$A=[a_{ij}]_ {m\times n}$，和$B=[b_{ij}]_ {m\times n}$，平方损失函数定义为 
$$||A-B||=\sum_{i,j}(a_{ij}-b_{ij})^2$$
+ 另一种损失函数是散度（divergence)。设两个非负矩阵$A=[a_{ij}]_ {m\times n}$，和$B=[b_{ij}]_ {m\times n}$,散度损失函数定义为
$$D(A||B)=\sum_{i,j}(a_{ij}log\frac{a_{ij}}{b_{ij}}-a_{ij}+b_{ij})$$
散度损失函数退化为Kuliback-Leiber散度或相对嫡，这时$A$和$B$是概率分布.

<div align=left>
    <img src="zh-cn/img/lsa/p15.png" /> 
</div>

使用梯度下降求解。


### 4.pLSA(pLSI)

概率潜在语义分析（pLSA)，也成为概率潜在索引（pLSI),是一种利用概率生成模型对文本集合进行主题分析的无监督学习方法，模型的最大特点是用隐变量表示话题；整个模型表示文本生成话题，话题生成单词，从而得到单词-文本共现矩阵的过程；假设每个文本由一个话题分布决定，每个话题由一个单词分布决定。

概率潜在语义分析受潜在语义分析的启发，1999年由Hofmann提出，前者基于概率模型，后者基于非概率模型。概率潜在语义分析最初用于文本数据挖掘，后来扩展到其他领域。

#### 1.基本思想

+ 给定一个文本集合，每个文本讨论若干个话题，每个话题由若干个单词表示。
+ 对文本集合进行概率潜在语义分析，就能够发现每个文本的话题，以及每个话题的单词。 
+ 话题是不能从数据中直接观察到的，是潜在的。

+ 文本集合转换为文本-单词共现数据，具体表现为单词-文本矩阵。
+ 文本数据基于如下的概率模型产生（共现模型）： 
    - 首先有话题的概率分布，然后有话题给定条件下文本的条件概率分布，以及话题给定条件下单词的条件概率分布。
    - 概率潜在语义分析就是发现由隐变量表示的话题，即潜在语义。
    - 直观上，语义相近的单词、语义相近的文本会被聚到相同的“软的类别”中，而话题所表示的就是这样的软的类别。
    - 假设有3个潜在的话题，图中三个框各自表示一个话题。

<div align=center>
    <img src="zh-cn/img/lsa/p16.png" /> 
</div>


#### 2.生成模型

+ 假设有单词集合$W={w_1,w_2,...,w_M}$,其中M是单词个数
+ 文本（指标）集合$D={d_1,d_2,...,d_N}$，其中N是文本个数
+ 话题集合$Z={z_1,z_2,...,z_K}$，其中$K$是预先设定的话题个数
+ 随机变量$w$取决于单词集合
+ 随机变量$d$取决于文本集合
+ 随机变量$z$取决于话题集合

+ 概率分布$P(d)$，条件概率分布$P(z|d)$,条件概率分布$P(w|z)$皆属于**多项分布**
+ $P(d)$：生成文本$d$的概率
+ $P(z|d)$:文本$d$生成话题$z$的概率
+ $P(w|z)$：话题$z$生成单词$w$的概率

一个文本的内容由其相关话题决定，一个话题的内容由其相关单词决定。生成模型通过以下步骤生成文本-单词共现数据：

1. 依据概率分布$P(d)$，从文本（指标）集合中随机选取一个文本$d$，共生成$N$个文本；针对每个文本，执行以下操作
2. 在文本$d$给定条件下，依据条件概率分布$P(z|d)$，从话题集合随机选取一个话题$z$，共生成$L$个话题，这里$L$是文本长度
3. 在话题$z$给定条件下，依据条件概率分布$P(w|z)$，从单词集合中随机选取一个单词$w$

+ 生成模型中，单词变量$w$与文本变量$d$是观测变量，话题变量$z$是隐变量
+ 模型生成的是单词-话题-文本三元组$(w, z, d)$的集合，但观测到的是单词-文本二元组$(w, d)$的集合
+ 观测数据表示为单词-文本矩阵$T$的形式,矩阵$T$的行表示单词，列表示文本，元素表示单词-文本对$(w, d)$的出现次数

+ 从数据的生成过程可以推出，文本-单词共现数据$T$的生成概率为所有单词-文本 对$(w, d)$的生成概率的乘积

<div align=center>
    <img src="zh-cn/img/lsa/p17.png" /> 
</div>

这里$n(w, d)$表示$(w, d)$的出现次数，单词-文本对出现的总次数是$N\times L$

+ 每个单词-文本对$(w, d)$的生成概率由以下公式决定
<div align=center>
    <img src="zh-cn/img/lsa/p18.png" /> 
</div>
即生成模型的定义,生成模型假设在话题$z$给定条件下，单词$w$与文本$d$条件独立，即
<div align=center>
    <img src="zh-cn/img/lsa/p19.png" /> 
</div>

+ 生成模型属于概率有向图模型，可以用有向图(directed graph)表示
<div align=center>
    <img src="zh-cn/img/lsa/p20.png" /> 
</div>
图中实心圆表示观测变量，空心圆表示隐变量，箭头表示概率依存关系，方框表示多次重复，方框内数字表示重复次数。

#### 3.共现模型

可以定义与以上的生成模型等价的共现模型。 文本-单词共现数据T的生成概率为所有单词-文本对$(w, d)$的生成概率的乘积：

<div align=center>
    <img src="zh-cn/img/lsa/p21.png" /> 
</div>

每个单词-文本对$(w, d)$的概率由以下公式决定：

<div align=center>
    <img src="zh-cn/img/lsa/p22.png" /> 
</div>
即共现模型的定义。

+ 共现模型假设在话题$z$给定条件下，单词$w$与文本$d$是条件独立的，即 
<div align=center>
    <img src="zh-cn/img/lsa/p23.png" /> 
</div>

+ 图中所示是共现模型。图中文本变量$d$是一个观测变量，单词变量$w$是一个观测变量，话题变量$z$是一个隐变量

<div align=center>
    <img src="zh-cn/img/lsa/p24.png" /> 
</div>


#### 4.模型性质

虽然生成模型与共现模型在概率公式意义上是等价的，但是拥有不同的性质。

+ 生成模型
    - 刻画文本-单词共现数据生成的过程
    - 单词变量$w$与文本变量$d$是非对称的
    - 非对称模型

+ 共现模型
    - 描述文本-单词共现数据拥有的模式
    - 单词变量$w$与文本变量$d$是对称的
    - 对称模型

图中显示模型中文本、话题、单词之间的关系。

<div align=center>
    <img src="zh-cn/img/lsa/p25.png" /> 
</div>

与潜在语义分析(LSA)的关系:

+ 概率潜在语义分析模型（共现模型）可以在潜在语义分析模型的框架下描述

<div align=center>
    <img src="zh-cn/img/lsa/p26.png" /> 
</div>
图中显示潜在语义分析，对单词-文本矩阵进行奇异值分解得到。

+ 共现模型也可以表示为三个矩阵乘积的形式

<div align=center>
    <img src="zh-cn/img/lsa/p27.png" /> 
</div>

+ 概率潜在语义分析模型中的矩阵$U^{'}$,和$V^{'}$是非负的、规范化的，表示条件概率分布。
+ 潜在语义分析模型中的矩阵$U$和$V$是正交的，未必非负，并不表示概率分布。 


#### 5.模型学习的EM算法

```
EM算法是一种迭代算法，每次迭代包括交替的两步：
 E步，求期望
 M步，求极大
```
+ E步是计算Q函数，即完全数据的对数似然函数对不完全数据的条件分布的期望
+ M步是对Q函数极大化，更新模型参数

>思考EM算法和极大似然估计的区别和联系？

<div align=left>
    <img src="zh-cn/img/lsa/p28.png" /> 
</div>

<div align=left>
    <img src="zh-cn/img/lsa/p29.png" /> 
</div>

<div align=left>
    <img src="zh-cn/img/lsa/p30.png" /> 
</div>

<div align=left>
    <img src="zh-cn/img/lsa/p31.png" /> 
</div>

<div align=left>
    <img src="zh-cn/img/lsa/p32.png" /> 
</div>

<div align=left>
    <img src="zh-cn/img/lsa/p33.png" /> 
</div>

>pLSA的EM算法：

<div align=left>
    <img src="zh-cn/img/lsa/p34.png" /> 
</div>
<div align=left>
    <img src="zh-cn/img/lsa/p35.png" /> 
</div>


### 5.Example

```python

'''
下面利用sogou提供的已经分好类的预料,尝试建立一个LSI模型。
文档解压之后，每个类别一个目录，每个文档一个文件，共17900多个文件，类别大致有财经、教育等等。
中文分词采用jiema提供的api，加载gensim完成相应的工作。
'''

import os
import re
import jieba as ws
import pandas as pd
from gensim import models,corpora
import logging
 
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
documents=[]
labels=[]
class_dir=os.listdir('/home/xujing/Reduced/')
 
#读取语料库
for i in class_dir:
    if i.find('C')>-1:
        currentpath='/home/kim/Reduced/'+i+'/'
#        print(currentpath)
        files=os.listdir(currentpath)
        for f in files:
            tmp_list=[]  
            tmp_str=''
            try:            
                file=open(currentpath+f,encoding='gbk')
                file_str=file.read()
                file_str=re.sub('(\u3000)|(\x00)|(nbsp)','',file_str)#正则处理，去掉一些噪音
                doc=''.join(re.sub('[\d\W]','',file_str))
                tmp_str='|'.join(ws.cut(doc))
                tmp_list=tmp_str.split('|')
                labels+=[i]
            except:
                print('read error: '+currentpath+f)
            documents.append(tmp_list)
            file.close()
            
             
#------------------------------------------------------------------------------
#LSI model: latent semantic indexing model
#------------------------------------------------------------------------------
#https://en.wikipedia.org/wiki/Latent_semantic_analysis
#http://radimrehurek.com/gensim/wiki.html#latent-semantic-analysis
dictionary=corpora.Dictionary(documents)
corpus=[dictionary.doc2bow(doc) for doc in documents]#generate the corpus
tf_idf=models.TfidfModel(corpus)#the constructor
 
#this may convert the docs into the TF-IDF space.
#Here will convert all docs to TFIDF
corpus_tfidf=tf_idf[corpus]
 
#train the lsi model
lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=9)
topics=lsi.show_topics(num_words=5,log=0)
for tpc in topics:
    print(tpc)

'''
设定参数num_topics=9,即指定主题有9个，每个词语前面的数字大致可以理解为该词与主题的相关程度。可以明显地看到，
topic[2]大致可以对应到教育主题，
topic[3]大致可以对应到财经主题，
topic[5]可以对应到体育的主题，etc。
事实上，由于在预处理阶段，没有对停用词进行处理，导致到有很多例如“我”、“你”等词语出现在最后结果之中。但从这个结果来看还勉强过得去。

实际上，在spark mllib里面，也提供了相应的方法，例如TF-IDF的转换，等等，后面将对word2vec模型、以及spark上的实现进行探索。
'''
```

gensim好像没有关于pLSA的实现，具体的可以参考gensim的官网：<https://radimrehurek.com/gensim/index.htmls>