## CRF(条件随机场)

条件随机场（CRF）是给定一组输入随机变量条件下另一组随机变量的条件概率分布模型，其特点是假设输出随机变量构成马尔科夫随机场。条件随机场可以用于不同的预测问题，这里主要讲解线性链条件随机场，这时，问题变成了由输入序列对输出序列的判别模型，形式上为对数线性模型。线性链条件随机场应用于标注问题是由Lafferty等人于2001年提出的。

### 1.概率无向图模型（马尔可夫随机场）

**1.模型定义**

图（graph)是由结点（node)及连接结点的边（edge)组成的集合。结点和边分别记做$v$和$e$,结点和边的集合分别记做$V$和$E$，图记作$G=(V,E)$,无向图是指边没有方向的图。

概率图模型是由图表示的概率分布。设由概率分布$P(Y)$,$Y\in y$是一组随机变量。由无向图$G=(V,E)$表示概率分布$P(Y)$，即在图$G$中，结点$v\in V$表示一个随机变量$Y_v$，$Y=(Y_v)_ {v \in V}$,边$e\in E$表示随机变量之间的概率依赖关系。

给定一个联合概率分布$P(Y)$和表示它的无向图$G$。首先定义无向图表示的随机变量之间存在的成对马尔可夫，局部马尔可夫，全局马尔可夫。

+ 成对马尔可夫性： 设$u$和$v$是无向图$G$中任意两个没有边连接的结点，结点$u$和$v$分别对应随机变量$Y_u$和$Y_v$。其他所有结点为$O$，对应的随机变量组是$Y_o$.成对马尔可夫性是指给定随机变量组$Y_o$的条件下随机变量$Y_u$和$Y_v$是条件独立的，即
$$P(Y_u,Y_v|Y_o)=P(Y_u|Y_o)P(Y_v|Y_o)$$

+ 局部马尔可夫性：设$v\in V$是无向图$G$中任意一个结点，$W$是与$v$有边连接的所有结点，$O$是$v$和$W$以外的其他所有结点。$v$表示的随机变量是$Y_v$,$W$表示的随机变量组是$Y_W$，$O$表示的随机变量组$Y_o$。局部马尔可夫性是指在给定随机变量组$Y_W$的条件下随机变量$Y_v$与随机变量组$Y_o$是独立的，即
$$P(Y_v,Y_o|Y_W)=P(Y_v|Y_W)P(Y_o|Y_W)$$

在$P(Y_o|Y_W)>0$时，等价的
$$P(Y_v|Y_W)=P(Y_v|Y_W,Y_o)$$

<div align=center>
    <img src="zh-cn/img/crf/p1.png" /> 
</div>


+ 全局马尔可夫性： 设结点集合A和B是在无向图$G$中被结点集合$C$分开的任意结点集合。结点集合$A,B,C$所对应的随机变量组分别为$Y_A,Y_B,Y_C$。全局马尔可夫性是指给定速记变量组$Y_C$条件下随机变量组$Y_A$和$Y_B$是条件独立的即
$$P(Y_A,Y_B|Y_C)=P(Y_A|Y_C)P(Y_B|Y_C)$$

<div align=center>
    <img src="zh-cn/img/crf/p2.png" /> 
</div>

**定义(概率无向图模型)**设在联合概率分布$P(Y)$,由无向图$G=(V,E)$表示，在图$G$中，结点表示随机变量，边表示随机变量之间的依赖关系。如果联合概率分布$P(Y)$满足成对，局部，或全局马尔可夫性，就称此联合概率分布为**概率无向图模型或马尔可夫随机场**。

以上是概率无向图模型的定义，实际上，我们更关心的是如何求其联合概率分布。对给定的概率无向图模型，我们希望将整体的联合概率写成若干个子联合概率的乘积的形式，也就是将联合概率进行因子分解，这样便于模型的学习与计算。事实上，概率无向图模型的最大特点就是异于因子分解。


### 2.条件随机场的定义和形式

**1.条件随机场的定义**

给定无向图中的团与最大团的定义

**定义（团与最大团)** 无向图$G$中任何两个结点均有边连接的结点子集称为团(clique)。若$C$是无向图$G$的一个团，并且不能再加进去任何一个$G$的结点使其成为一个更大的团，则称此$C$为最大团.

<div align=center>
    <img src="zh-cn/img/crf/p3.png" /> 
</div>

+ 2个结点的团：$\\{Y_1,Y_2\\},\\{Y_2,Y_3\\},\\{Y_3,Y_4\\},\\{Y_4,Y_2\\},\\{Y_1,Y_3\\}$
+ 最大团：$\\{Y_1,Y_2,Y_3\\},\\{Y_2,Y_3,Y_4\\}$
+ $\\{Y_1,Y_2,Y_3,Y_4\\}$不是一个团，因为$Y_1$和$Y_4$没有边连接。

将概率无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积的形式的操作，成为概率无向图模型的因子分解。

给定概率无向图模型，设其无向图为$G$,$C$为$G$上的最大团，$Y_C$表示$C$对应的随机变量。那么概率无向图模型的联合概率分布$P(Y)$可写作图中所有最大团$C$上的函数$\Psi_C(Y_C)$的乘积形式，即
$$P(Y)=\frac{1}{Z} \prod_C \psi_C(Y_C)$$

其中，$Z$是规范化因子，如下：
$$Z=\sum_Y\prod_C\Psi_C(Y_C)$$

规范化因子保证$P(Y)$构成一个概率分布。函数$\Psi_C(Y_C)$成为势函数(potential function).这路要求势函数$\Psi_C(Y_C)$是严格正的，通常定义为指数函数
$$\Psi_C(Y_C)=exp(-E(Y_C))$$

概率无向图模型的因子分解由下述定理保证：

**定理（Hammersley-Clifford定理）**概率无向图模型的联合概率分布$P(Y)$可以表示为如下形式：
$$P(Y)=\frac{1}{Z}\prod_C \psi_C(Y_C)$$
$$Z=\sum_Y\prod_C\Psi_C(Y_C)$$

其中，$C$是无向图的最大团，$Y_C$是$C$的结点对应的随机变量，$\Psi_C(Y_C)$是$C$上定义的严格函数，乘积是在无向图所有的最大团上进行的。

**2.CRF的参数化形式**

**定理（线性链条件随机场的参数化形式）** 设$P(Y|X)$为线性链条件随机场，则在随机变量$X$取值为$x$的条件下，随机变量$Y$取值为$y$的条件概率具有如下形式:

<div align=center>
    <img src="zh-cn/img/crf/p4.png" /> 
</div>
其中：
<div align=center>
    <img src="zh-cn/img/crf/p5.png" /> 
</div>

+ $t_k$ 定义在边上的特征函数，转移特征，依赖于前一个和当前位置，$\lambda_k$为其权重
+ $s_l$ 定义在结点上的特征函数，状态特征，依赖于当前位置，$\mu_l$为其权重
+ 通常，特征函数$t_k$和$s_l$取值为1或0，当满足特征条件时取值1，否则0

!> 例题

<div align=center>
    <img src="zh-cn/img/crf/p6.png" /> 
</div>


**3.CRF的简化形式**

注意到条件随机场中同一特征在各个位置都有定义，可以对同一个特征在各个位置求和，将局部特征函数转化为一个全局特征函数，这样就可以将条件随机场写成权值向量和特征向量的内积形式，即条件随机场的简化形式。首先将转移特征和状态特征及其权值用统一的符号表示，设有$K_1$个转移特征，$K_2$个状态特征，$K=K_1+K_2$,记

<div align=center>
    <img src="zh-cn/img/crf/p7.png" /> 
</div>

然后，对转移与状态特征在各个位置i求和，记作

<div align=center>
    <img src="zh-cn/img/crf/p8.png" /> 
</div>

权值：

<div align=center>
    <img src="zh-cn/img/crf/p9.png" /> 
</div>

则条件随机场可表示为:

<div align=center>
    <img src="zh-cn/img/crf/p10.png" /> 
</div>

若$w$表示权值向量:$w=(w_1,w_2,...,w_K)^T$,以$F(y,x)$表示全局特征向量，即

<div align=center>
    <img src="zh-cn/img/crf/p11.png" /> 
</div>
则，条件随机场写成内积：


<div align=center>
    <img src="zh-cn/img/crf/p12.png" /> 
</div>
<div align=center>
    <img src="zh-cn/img/crf/p13.png" /> 
</div>

**3.CRF的矩阵形式**

线性链条件随机场，引进特殊的起点和终点状态标记$Y_0= start$， $Y_{n+1} = stop$，这时$P_w(y|x)$可以通过矩阵形式表示。对观测序列$x$的每一个位置$i=1,2,...,n+1$，定义一个$m$阶矩阵($m$是标记$Y_i$取值的个数)

<div align=center>
    <img src="zh-cn/img/crf/p14.png" /> 
</div>

矩阵随机变量的元素为：

<div align=center>
    <img src="zh-cn/img/crf/p15.png" /> 
</div>
<div align=center>
    <img src="zh-cn/img/crf/p16.png" /> 
</div>

给定观测序列$x$，标记序列$y$的非规范化概率可以通过$n+l$个矩阵的乘积表示：

<div align=center>
    <img src="zh-cn/img/crf/p17.png" /> 
</div>


条件概率$P_w(y|x)$

<div align=center>
    <img src="zh-cn/img/crf/p18.png" /> 
</div>

$Z_w(x)$为规范化因子，是$n+1$个矩阵的乘积的(start, stop)元素

<div align=center>
    <img src="zh-cn/img/crf/p19.png" /> 
</div>

!> 例题

<div align=center>
    <img src="zh-cn/img/crf/p20.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/crf/p21.png" /> 
</div>


### 3.条件随机场的概率计算问题

条件随机场的概率计算问题包括：

+ 给定条件随机场$P(Y|X)$，输入序列$x$和输出序列$y$
+ 计算条件概率$P(Y=y_i|x)$,$P(Y_{i-1}=y_{i-1},Y_i=y_i|x)$
+ 及相应的数学期望
+ 使用类似于HMM的前向-后向算法，递归计算

**1.前向-后向算法**


对每一个指标$i=0,1,2,...,n+1$,定义前向向量$\alpha_i(x)$,

<div align=center>
    <img src="zh-cn/img/crf/p22.png" /> 
</div>

递推公式为：

<div align=center>
    <img src="zh-cn/img/crf/p23.png" /> 
</div>

又可以表示为：

<div align=center>
    <img src="zh-cn/img/crf/p24.png" /> 
</div>

即表示在位置$i$的标记是$y_i$，且到位置$i$的前部分标记序列的非规范化概率，$y_i$可取的值$m$个，所以$\alpha_i(x)$是$m$维列向量。

同样，对每个指标$i=0,1,2,3,...,n+1$,定义后向向量$\beta_i(x)$

<div align=center>
    <img src="zh-cn/img/crf/p25.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/crf/p26.png" /> 
</div>

又可以表示为：

<div align=center>
    <img src="zh-cn/img/crf/p27.png" /> 
</div>


即表示在位置$i$的标记是$y_i$，且从位置$i+1$到$n$的后部分标记序列的非规范化概率

前向-后向得：
<div align=center>
    <img src="zh-cn/img/crf/p28.png" /> 
</div>

**2.概率计算**

按照前向-后向向量的定义，可计算标记序列在位置$i$是标记$y_i$的条件概率

<div align=center>
    <img src="zh-cn/img/crf/p29.png" /> 
</div>

在位置$i-1$与$i$是标记$y_{i-1}$和$y_i$的条件概率

<div align=center>
    <img src="zh-cn/img/crf/p30.png" /> 
</div>

其中

<div align=center>
    <img src="zh-cn/img/crf/p31.png" /> 
</div>

**3.期望的计算**

假设经验分布为$\tilde{P}(X)$,特征函数$f_k$关于联合分布$P(X,Y)$的数学期望为

<div align=center>
    <img src="zh-cn/img/crf/p32.png" /> 
</div>


### 4.条件随机场的学习算法









### 5.条件随机场的预测算法