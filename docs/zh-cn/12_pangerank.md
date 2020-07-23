## PageRank


+ 在实际应用中许多数据都以图（graph)的形式存在，比如，互联网、社交网络都可以看作是一个图
+ 图数据上的机器学习具有理论与应用上的重要意义
+ PageRank算法是图的链接分析（link analysis）的代表性算法，属于图数据上的无监督学习方法。
+ PageRank可以定义在任意有向图 上，后来被应用到社会影响力分析、文本摘要等多个问题。
+ PageRank算法的基本想法是在有向图上定义一个随机游走模型，即一阶马尔可夫链，描述随机游走者沿着有向图随机访问各个结点的行为
+ 在一定条件下，极限情况访问每个结点的概率收敛到平稳分布，这时各个结点的平稳概率值就是其PageRank值，表示结点的重要度。
+ PageRank是递归定义的，PageRank的计算可以 通过迭代算法进行。


PageRank最初作为互联网网页重要度的计算方法，1996年由Page和Brin提出，并用于谷歌搜索引擎的网页排序。事实上，PageRank可以定义在任意有向图上，后来被应用到社会影响力分析，文本摘要等多个问题。

PageRank算法的基本思想是在有向图上定义的一个随机游走模型，即一阶马尔科夫链，描述随机游走者沿着有向图随机访问各个节点的行为。在一定条件下，极限情况访问每个节点的概率收敛到平稳分布，这时各个节点的平稳概率值就是其PageRank值，表示节点的重要度。PageRank是递归定义的，计算可以通过迭代算法完成。


### 1.PageRank的定义

**1.基本思想**

历史上，PageRank算法作为计算互联网网页重要度的算法被提出
PageRank是定义在网页集合上的一个函数，它对每个网页给出一个正实数，表示网页的重要程度， 整体构成一个向量
PageRank值越高，网页就越重要，在互联网搜索的排序中可能就 被排在前面。

假设互联网是一个有向图，在其基础上定义随机游走模型，即一阶马尔可夫链， 表示网页浏览者在互联网上随机浏览网页的过程
假设浏览者在每个网页依照连接出去的超链接以等概率跳转到下一个网页，并在网上持续不断进行这样的随机跳转，这个过程形成一阶马尔可夫链
PageRank表示这个马尔可夫链的平稳分布
每个网页的PageRank值就是平稳概率。


下图表示一个有向图，假设是简化的互联网例，结点A,B,C和D表示网页， 结点之间的有向边表示网页之间的超链接，边上的权值表示网页之间随机跳转的概率


<div align=center>
    <img src="zh-cn/img/pagerank/p1.png" /> 
</div>


+ 假设有一个浏览者，在网上随机游走
+ 如果浏览者在网页A，
	- 则下一步以1/3的概率转移到网页B,C和D
+ 如果浏览者在网页B，
	- 则下一步以1/2的概率转移到网页A和D
+ 如果浏览者在网页C，
	- 则下一步以概率1转移到网页A
+ 如果浏览者在网页D，
	- 则下一步以1/2的概率转移到网页B和C


+ 直观上，一个网页，如果指向该网页的超链接越多，随机跳转到该网页的概率也就越高，该网页的PageRank值就越高，这个网页也就越重要
+ 一个网页，如果指向该网页的PageRank值越高，随机跳转到该网页的概率也就越高，该网页的PageRank 值就越高，这个网页也就越重要
+ PageRank值依赖于网络的拓扑结构，一旦网络的拓扑（连接关系）确定，PageRank值就确定
+ PageRank的计算可以在互联网的有向图上进行，通常是一个迭代过程
+ 先假设一 个初始分布，通过迭代，不断计算所有网页的PageRank值，直到收敛为止


**2.有向图和随机游走模型**

1.有向图

**定义(有向图)** 有向图(directed graph)记作$G=(V,E)$，其中$V$和$E$分别表示节点和有向边的集合。

+ 从一个结点出发到达另一个结点，所经过的边的一个序列称为一条路径(path), 路径上边的个数称为路径的长度
+ 如果一个有向图从其中任何一个结点出发可以到达 其他任何一个结点，就称这个有向图是**强连通图（strongly connected graph)**
+ 假设$k$是一个大于1的自然数，如果从有向图的一个结点出发返回到这个结点的路径的长度都是$k$的倍数，那么称这个结点为周期性结点
+ 如果一个有向图不含有周期性结点，则称这个有向图为**非周期性图（aperiodic graph)**，否则为**周期性图**
+ 下图是一个周期性有向图的例子

<div align=center>
    <img src="zh-cn/img/pagerank/p2.png" /> 
</div>

+ 从结点A出发返回到A，必须经过路径 `A一B一C一A`，所有可能的路径的长度都是3的倍数，所以结点A是周期性结点。 这个有向图是周期性图

2.随机游走模型

<div align=center>
    <img src="zh-cn/img/pagerank/p3.png" /> 
</div>

+ 注意转移矩阵具有性质

<div align=center>
    <img src="zh-cn/img/pagerank/p4.png" /> 
</div>

+ 即每个元素非负，每列元素之和为1，即矩阵M为随机矩阵（stochastic matrix)。
+ 在有向图上的随机游走形成马尔可夫链。也就是说，随机游走者每经一个单位时间转移一个状态
+ 如果当前时刻在第$j$个结点（状态），那么下一个时刻在第$i$个结点（状态）的概率是$m_{ij}$
+ 这一概率只依赖于当前的状态，与过去无关，具有马尔可夫性

+ 在下图的有向图上可以定义随机游走模型

<div align=center>
    <img src="zh-cn/img/pagerank/p1.png" /> 
</div>

+ 结点A到结点B,C和D存在有向边，可以以概率1/3从A分别转移到B,C和D，并以概率0转移到A，于是可以写出转移矩阵的第1列
+ 结点B到结点A和D存在有向边，可以以概率1/2从B分别转移到A和D，并以概率0分别转移到B和C，于是可以写出矩阵的第2列
+ 于是得到转移矩阵

<div align=center>
    <img src="zh-cn/img/pagerank/p5.png" /> 
</div>

+ 随机游走在某个时刻$t$访问各个结点的概率分布就是马尔可夫链在时刻$t$的状态分布，可以用一个$n$维列向量$R_{t}$表示，那么在时刻$t+1$访问各个结点的概率分布$R_{t+1}$满足
$$R_{t+1}=MR_{t}$$


**3.PageRank的基本定义**

给定一个包含$n$个结点的强连通且非周期性的有向图，在其基础上定义随机游走模型。假设转移矩阵为$M$，在时刻$0,1,2,...,t,...$访问各个结点的概率分布为
$$R_{0},MR_{0},M^{2}R_{0},...,M^{t}R_{0},...$$
则极限$\lim_{t\to \infty}M^{t}R_{0}=R$存在，极限向量$R$表示马尔可夫链的平稳分布，满足 
$$MR=R$$

下面给出PageRank的基本定义

<div align=center>
    <img src="zh-cn/img/pagerank/p6.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/pagerank/p7.png" /> 
</div>


<div align=center>
    <img src="zh-cn/img/pagerank/p8.png" /> 
</div>

!> 例子：

求下图的PageRank

<div align=center>
    <img src="zh-cn/img/pagerank/p1.png" /> 
</div>

转移矩阵

<div align=center>
    <img src="zh-cn/img/pagerank/p5.png" /> 
</div>

取初始分布向量$R_{0}$为

<div align=center>
    <img src="zh-cn/img/pagerank/p9.png" /> 
</div>


以转移矩阵M连乘初始向量$R_{0}$得到向量序列

<div align=center>
    <img src="zh-cn/img/pagerank/p10.png" /> 
</div>

最后我们就得到了最后的PageRank值！


!> 一般的有向图未必满足强连通且非周期性的条件。所以PageRank 的基本定义不适用。

!> 例子 

下图的有向图的转移矩阵$M$是 

<div align=center>
    <img src="zh-cn/img/pagerank/p11.png" /> 
</div>

这时$M$不是一个随机矩阵，因为随机矩阵要求每一列的元素之和是1，这里第3列的和是0，不是1，如果仍然计算在各个时刻的各个结点的概率分布，就会得到如下结果

<div align=center>
    <img src="zh-cn/img/pagerank/p12.png" /> 
</div>

可以看到，随着时间推移，访问各个结点的概率皆变为0


**4.PageRank的一般定义**

+ PageRank一般定义的想法是在基本定义的基础上导入平滑项
+ 给定一个含有$n$个结点$v_i, i=1,2,… ,n$，的任意有向图
+ 假设考虑一个在图上 随机游走模型，即一阶马尔可夫链，其转移矩阵是$M$，从一个结点到其连出的所有结点的转移概率相等
+ 这个马尔可夫链未必具有平稳分布
+ 假设考虑另一个完全随机游走的模型，其转移矩阵的元素全部为$1/n$，也就是说从任意一个结点到任意一个结点的转移概率都是$1/n$
+ 两个转移矩阵的线性组合又构成一个新的转移矩阵，在其上可以定义一个新的马尔可夫链。
+ 容易证明这个马尔可夫链一定具有平稳分布，且平稳分布满足
$$R=dMR+\frac{1-d}{n}1$$

式中$d(0≤d ≤ 1)$是系数，称为**阻尼因子（damping factor)**，
$R$是$n$维向量,$1$是所有分量为1的$n$维向量,$R$表示的就是有向图的一般PageRank

上式中，第一项表示状态分布是平稳分布时依照转移矩阵$M$访问各个节点的概率，第二项表示完全随机访问各个结点的概率，阻尼因子$d$的取值由经验决定，例如$d=0.85$,当$d$接近1时随机游走主要依照转移矩阵$M$进行,当$d$接近0时， 随机游走主要以等概率随机访问各个结点。 

可以由上式写出每个结点的PageRank，这是一般PageRank的定义
$$PR(v_i)=d(\sum_{v_j\in M(v_j)}\frac{PR(v_j)}{L(v_j)})+\frac{1-d}{n},i=1,2,...,n$$
第二项称为平滑项，由于采用平滑项，所有结点的PageRank值都不会为0，具有以下性质： 


<div align=center>
    <img src="zh-cn/img/pagerank/p13.png" /> 
</div>


<div align=center>
    <img src="zh-cn/img/pagerank/p14.png" /> 
</div>

一般PageRank的定义意味着互联网浏览者，按照以下方法在网上随机游走：

+ 在任意一个网页上，浏览者或者以概率$d$决定按照超链接随机跳转，这时以等概率从连接出去的超链接跳转到下一个网页
或者以概率$(1-d)$决定完全随机跳转，这时以等概率$1/n$跳转到任意一个网页
+ 第二个机制保证从没有连接出去的超链接的网页也可以跳转出。这样可以保证平稳分布，即一般PageRank的存在，因而一般PageRank适用于任何结构的网络。

### 2.PageRank的计算

PageRank的定义是构造性的，即定义本身就给出了算法，本节列出PageRank的计算方法包括迭代算法，幂法，袋鼠算法，其中常用的方法是幂法。

**1.迭代算法**

给定一个含有$n$个结点的有向图，转移矩阵为$M$，有向图的一般PageRank由迭代公式 
$$R_{t+1}=dmR_{t}+\frac{1-d}{n}1$$
的极限向量$R$确定。

PageRank的迭代算法就是按照这个一般定义进行迭代，直至收敛

<div align=center>
    <img src="zh-cn/img/pagerank/p15.png" /> 
</div>

!> 例子

图中所示的有向图，取$d=0.8$，求图的PageRank

<div align=center>
    <img src="zh-cn/img/pagerank/p16.png" /> 
</div>

可得转移矩阵为	

<div align=center>
    <img src="zh-cn/img/pagerank/p17.png" /> 
</div>


按照迭代算法计算


<div align=center>
    <img src="zh-cn/img/pagerank/p18.png" /> 
</div>


迭代公式

<div align=center>
    <img src="zh-cn/img/pagerank/p19.png" /> 
</div>

令初始向量

<div align=center>
    <img src="zh-cn/img/pagerank/p20.png" /> 
</div>


进行迭代

<div align=center>
    <img src="zh-cn/img/pagerank/p21.png" /> 
</div>


最后得到

<div align=center>
    <img src="zh-cn/img/pagerank/p22.png" /> 
</div>

计算结果表明，结点C的PageRank值超过一半，其他结点也有相应的 PageRank值。




**2.幂法**

+ 幂法（(power method)是一个常用的PageRank计算方法，通过近似计算矩阵的主特征值和主特征向量求得有向图的一般PageRank
+ 幂法主要用于近似计算矩阵的主特征值（dominant eigenvalue）和 主特征向量（dominant eigenvector)
+ 主特征值是指绝对值最大的特征值
+ 主特征向量是其对应的特征向量
+ 注意特征向量不是唯一的，只是其方向是确定的，乘上任意系数还是特征向量

假设要求$n$阶矩阵$A$的主特征值和主特征向量，采用下面的步骤。

1.首先，任取一个初始$n$维向量$x_0$，构造如下的一个$n$维向量序列
$$x_0,x1=Ax_0,x2=Ax_1,...,x_k=Ax_{k-1}$$

2.然后，假设矩阵$A$有$n$个特征值，按照绝对值大小排列
$$|\lambda_1|>=|\lambda_2|>=...>=|\lambda_n|$$

3.对应的$n$个线性无关的特征向量为
$$\mu_1,\mu_2,...,\mu_n$$
这n个特征向量构成n维空间的一组基

4.于是，可以将初始向量$x_0$表示为$\mu_1,\mu_2,...,\mu_n$的线性组合
$$x_0=a_1\mu_1+a_2\mu_2+...+a_n\mu_n$$
得到
<div align=center>
    <img src="zh-cn/img/pagerank/p23.png" /> 
</div>

5.接着，假设矩阵$A$的主特征值$\lambda_1$是特征方程的单根，由上式得

<div align=center>
    <img src="zh-cn/img/pagerank/p24.png" /> 
</div>

由于特征值存在的大小关系，当$k$充分大时，
$$x_k=a_1\lambda^{k}_ {1}[\mu_1+\varepsilon_k]$$

这说明当$k$充分大时，向量$x_k$与tezhengxiangliang$\mu_1$只相差一个系数
<div align=center>
    <img src="zh-cn/img/pagerank/p25.png" /> 
</div>

于是主特征值$\lambda_1$可表示为

<div align=center>
    <img src="zh-cn/img/pagerank/p26.png" /> 
</div>

其中$x_{k,j}$和$x_{k+1,j}$分别是$x_k$和$x_{k+1}$的第$j$个分量

在实际计算时，为了避免出现绝对值过大或过小的情况，通常在每步迭代后即进 行规范化，将向量除以其范数，即

<div align=center>
    <img src="zh-cn/img/pagerank/p27.png" /> 
</div>
这里的范数是向量的无穷范数，即向量各分量的绝对值的最大值.

现在回到计算一般PageRank。转移矩阵可以写作
$$R=(dM+\frac{1-d}{n}E)R=AR$

其中
+ $d$是阻尼因子
+ $E$是所有元素为1的n阶方阵
+ 根据Perron-Frobenius定理， 一般PageRank的向量$R$是矩阵$A$的主特征向量，主特征值是1

所以可以使用幂法 近似计算一般PageRank 

<div align=center>
    <img src="zh-cn/img/pagerank/p28.png" /> 
</div>

!> 例子

给定一个如图所示的有向图，取$d=0.85$，求有向图的一般 PageRank
<div align=center>
    <img src="zh-cn/img/pagerank/p29.png" /> 
</div>

由图可知转移矩阵

<div align=center>
    <img src="zh-cn/img/pagerank/p30.png" /> 
</div>

令 $t=0$

<div align=center>
    <img src="zh-cn/img/pagerank/p31.png" /> 
</div>

计算有向图的一般转移矩阵$A$

<div align=center>
    <img src="zh-cn/img/pagerank/p32.png" /> 
</div>

迭代并规范化

<div align=center>
    <img src="zh-cn/img/pagerank/p33.png" /> 
</div>

如此继续迭代规范化得到$x_t,t=1,2,3,...,22$的向量序列

<div align=center>
    <img src="zh-cn/img/pagerank/p34.png" /> 
</div>

得到了满足精度的向量

<div align=center>
    <img src="zh-cn/img/pagerank/p35.png" /> 
</div>

规范化使其作为PageRank的概率分布

<div align=center>
    <img src="zh-cn/img/pagerank/p36.png" /> 
</div>

**3.代数算法**

代数算法通过一般转移矩阵的逆矩阵计算求有向图的一般PageRank
按照一般PageRank的定义式
$$R=dMR+\frac{1-d}{n}1$$
于是
$$(I-dM)R=\frac{1-d}{n}1$$
$$R=(I-dM)^{-1}\frac{1-d}{n}1$$

这里$I$是单位矩阵。当$0< d<1$ 时，上述线性方程组的解存在且唯一
这样，可以通过求逆矩阵$(I-dM)^{-1}$得到有向图的一般PageRank.


### 3.总结

PageRank的原论文可以参考The PageRank citation ranking: vring order to the Web.其详细介绍可以参考Mining of massive datasets.和Web data mining: exploring hyperlinks,contents, and usage data.与PageRank同样著名的连接分析算法还有HITS算法，可以发现网络中的枢纽和权威。PageRank有不少扩展和变形，原始的PageRank是基于离散时间马尔可夫链的，BrowseRank是基于连续时间马尔可夫链的推广，可以更好的防范网页排名欺诈。Personalized PageRank是个性化的PageRank, Topic Sensitive PageRank是基于话题的PageRank,TrustRank是防范网页排名欺诈的PageRank等等。

参考文献： 本章内容参考李航老师的《统计机器学习 第二版》