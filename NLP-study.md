# NLP

# Python基础与数学

## python基础

闭包：一个返回值是函数的函数

## 数学

拉格朗日函数

贝叶斯定理

# 算法

viterbi (维特比) 算法 ：解决多步骤每步多选择模型的最优选择问题。

排序算法

动态规划

# NLP

Pipeline：原始文本->分词->清洗->标准化->特征提取->建模

## 模型

**N-gram模型** 

N-gram模型采用马尔科夫假设。**一个词的出现仅与它之前的若干个词有关**。

一次只考虑一个词称之为 **Unigram**。

如果一个词的出现仅依赖于它前面出现的一个词，那么我们就称之为 **Bi-gram**。

如果一个词的出现仅依赖于它前面出现的两个词，那么我们就称之为 **Tri-gram。**

**N-gram中的数据平滑方法**

N-gram最大的问题就是**稀疏问题（Sparsity）**。

拉普拉斯平滑：

+ Add-one

  ![image-20210604171906517](images/image-20210604171906517-1626749063079.png)

+ Add-K

  ![image-20210604171926408](images/image-20210604171926408-1626749065804.png)

Good-Turning 平滑



**朴素贝叶斯**

朴素贝叶斯法对条件概率分布做了条件独立性的假设。

朴素贝叶斯将句子处理为一个**词袋模型（Bag-of-Words, BoW）**，以至于不考虑每个单词的顺序。

极大似然估计（Maximum Likelihood Estimate，MLE）

最大后验概率估计（Maximum a posteriori estimation, MAP）

## 句子相似度

计算距离

- 欧氏距离
- 余弦相似度

Tf-IDF 算法  参考[TF-IDF与余弦相似度的应用](http://www.ruanyifeng.com/blog/2013/03/cosine_similarity.html)

降低计算相似度时的时间复杂度：倒排表

## 正则化

**L1和L2是正则化项，又叫做罚项，是为了限制模型的参数，防止模型过拟合而加在损失函数后面的一项。**

L1正则：

L1-norm有能产生许多零值或非常小的值的系数的属性，很少有大的系数。

L2正则

正则化项的超参数λ可以通过交叉验证选取。

## **分词方法**

中文分词：

- 前向最大匹配
- 向后最大匹配
- 双向最大匹配

英文分词：

- 英文分词算法(Porter stemmer)

## 评价指标

混淆矩阵

## 超参数搜索方法

**试错法（Babysitting）**：

**网格搜索（Grid search）**：遍历的思路，可行的参数范围都搜一遍。

**随机搜索（Random Search）**：

**贝叶斯优化（Bayesian Optimization）**：

**进化算法优化**：遗传算法



**特征选择**

- **穷举搜索（Exhaustive）**
- **启发式搜索 (Heuristic Search)**：利用当前与问题有关的信息作为启发式信息，这些信息是能够提升查找效率以及减少查找次数的。
  - 序列前向选择( SFS , Sequential Forward Selection )是一种贪心算法
  - 序列后向选择( SBS , Sequential Backward Selection )是一种贪心算法

