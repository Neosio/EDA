# EDA 
###  What & Why EDA?

Exploratory Data Analysis, 数据探索性分析，通过了解数据集，了解变量间的相互关系以及变量与预测值之间的关系，从而帮助我们后期更好地进行特征工程和建立模型，是数据挖掘中十分重要的一步。

数据科学，数据建模，甚至对于发一篇高质量的pape来说都是基础且重要的一环，挖掘这些数据的特点，选取合适的feature，甚至创造新的(magic） feature, 比直接上来生搬硬套模型有用得多。其次，数据量大的时候，training花费的时间是很多的，能早早发现数据的特点，有的放矢地training，才是高效之道。

### What we need?

* 数据科学库（Pandas、Numpy、Scipy）
* 可视化库（Matplotlib、Seabon）

### Prepare & Plan for EDA



**基本准备**
1. Domain Knowledge 领域知识
像数学建模比赛一样, 了解这个领域的背景知识，能帮助你进行下一步的操作而不是抓瞎
2. Checking if the data is intuitive
数据的均值是多少？方差多少？极值大概多少？有多少missing values？
3. How the data was generated?
数据取自于真实环境，但是既有可能是随机sample的，也有可能是over-sample, 也就是不均匀的采样。 搞懂数据是怎么来的，对validation流程的设立至关重要。

**例如**: Training set 和Test set假如用不同的方法生成，那么不应该用Training set的一部分作validation set..否则会带偏。。

> Coursera: competitive data science  学习一下



### How to EDA?

> talk is cheap, show me the code.

**具体步骤：**

1. 数据总览
2. 查看数据缺失和异常
3. 查看预测值的分布
4. 把特征分成类别特征、数值特征，然后对这两种种特征进行更细致的分析





