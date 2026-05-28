## OptIForest: Optimal Isolation Forest for Anomaly Detection

## Haolong Xiang^1 , Xuyun Zhang^1 ∗, Hongsheng Hu^2 , Lianyong Qi^3 , Wanchun Dou^4 , Mark

## Dras^1 , Amin Beheshti^1 and Xiaolong Xu^5 ∗

(^1) Macquarie University
(^2) CSIRO’s Data
(^3) Qufu Normal University
(^4) Nanjing University
(^5) Nanjing University of Information Science and Technology

## haolong.xiang@hdr.mq.edu.au,{xuyun.zhang, mark.dras, amin.beheshti}@mq.edu.au,

## Hongsheng.Hu@data61.csiro.au, lianyongqi@gmail.com, douwc@nju.edu.cn, xlxu@nuist.edu.cn

## Abstract

```
Anomaly detection plays an increasingly impor-
tant role in various fields for critical tasks such as
intrusion detection in cybersecurity, financial risk
detection, and human health monitoring. A vari-
ety of anomaly detection methods have been pro-
posed, and a category based on the isolation forest
mechanism stands out due to its simplicity, effec-
tiveness, and efficiency, e.g., iForest is often em-
ployed as a state-of-the-art detector for real de-
ployment. While the majority of isolation forests
use the binary structure, a framework LSHiForest
has demonstrated that the multi-fork isolation tree
structure can lead to better detection performance.
However, there is no theoretical work answering
the fundamentally and practically important ques-
tion on the optimal tree structure for an isolation
forest with respect to the branching factor. In this
paper, we establish a theory on isolation efficiency
to answer the question and determine the optimal
branching factor for an isolation tree. Based on the
theoretical underpinning, we design a practical op-
timal isolation forest OptIForest incorporating clus-
tering based learning to hash which enables more
information to be learned from data for better iso-
lation quality. The rationale of our approach relies
on a better bias-variance trade-off achieved by bias
reduction in OptIForest. Extensive experiments on
a series of benchmarking datasets for comparative
and ablation studies demonstrate that our approach
can efficiently and robustly achieve better detection
performance in general than the state-of-the-arts in-
cluding the deep learning based methods.
```
## 1 Introduction

```
Detection of anomalies (also known as outliers) is an im-
portant machine learning task to capture abnormal patterns
or sparse observations in data that collide with the major-
ity showing the expected behaviours [Panget al., 2021a],
and has been deployed in a broad range of fields for criti-
```
```
cal applications such as intrusion detection in cybersecurity,
financial risk detection, and human or device health moni-
toring[Ahmedet al., 2016; Chen and Tsourakakis, 2022;
Fernandoet al., 2021]. While anomalies often take a very
small portion of the data or appear infrequently, failing to
catch them in a timely manner for further actions can result
in severe consequences such as cascading failures in man-
ufacture and deaths in healthcare. Therefore, a variety of
unsupervised anomaly detection methods[Ruffet al., 2021],
from shallow to deep, have been proposed for different types
of anomalies and data types. Note that since collecting la-
bels for data is usually difficult and expensive, especially for
anomalies, we herein focus on unsupervised anomaly detec-
tion which is more commonly-used in practice. Recently, sev-
eral deep learning based detection methods, e.g., RDP[Wang
et al., 2021]and REPEN[Panget al., 2018], have shown
the advantages of having feature representation learning for
anomaly detection [Panget al., 2021a]. But this does not
mean that the traditional (shallow) detection methods will be
obsolete because deep neural networks have their own in-
trinsic limitations such as high computational cost, poor ex-
plainability, and difficulty in hyperparameter tuning[Liet al.,
2022 ], or a shallow model can have comparable performance
but cost much less, e.g., a very recent work ECOD[Liet al.,
2022 ]just use the statistic analysis of the tail event of a distri-
bution to achieve the good performance over various bench-
mark datasets. Instead, traditional models can be preferred
in specific scenarios, e.g., edge computing where computa-
tional resources are limited and medical research where high
explainability is in demand. Moreover, these methods can
work together with the feature representations learned from
deep learning for better performance, e.g., a recent work[Xu
et al., 2022]has shown such an example that iForest is suc-
cessfully used together with deep learning.
```
```
Benefiting from ensemble learning[Aggarwal and Sathe,
2015 ], a category of detection methods based on the isola-
tion forest mechanism[Haririet al., 2019]stands out of the
shallow models due to its excellent simplicity, effectiveness,
and efficiency, therefore being really promising for big data
anomaly detection. The basic idea is to randomly and recur-
sively partition relatively small samples drawn from a dataset
```
# arXiv:2306.12703v2 [cs.LG] 23 Jun 2023


until all data instances are isolated, which produces a for-
est of isolation trees. Anomaly scores can be derived from
an isolation forest based on an observation that anomalies
often have shorter path lengths due to the ease of isolating
them from others. As the first instance, iForest[Liuet al.,
2008 ]has been widely recognised in academia and deployed
in real applications, e.g., it has been included inscikit-learn,
a commonly-used machine learning library in Python. While
most of isolation forests following iForest use the binary tree
structure for data isolation, a framework LSHiForest[Zhang
et al., 2017]producing multi-fork isolation trees with the use
of similarity hash functions has demonstrated better detec-
tion performance. An interesting question arises naturally:
What is the optimal branching factor for an isolation tree?
However, there is little theoretical work answering this fun-
damentally important question, and practically this wide gap
can hamper the further development of isolation forest based
anomaly detection methods.

In this paper, we investigate this interesting problem and
establish a theory on the structure optimality of an isola-
tion tree with respect to the branching factor by introducing
the notion of isolation efficiency. A constrained optimisa-
tion problem is formulated to solve the optimality problem,
and an interesting finding is that the optimal branching fac-
tor ise(Euler’s number), rather than 2 which is commonly-
used in existing methods. Based on the theoretical founda-
tion, a practical optimal isolation forest named OptIForest is
proposed for efficient and robust anomaly detection. Specif-
ically, we adapt clustering based learning to hash in OptI-
Forest to let the isolation process optimised with more in-
formation learned from data. With two key observations on
ensemble learning and anomaly distribution in an isolation
tree, we design controllable initialisation for agglomerative
hierarchical clustering, enabling OptIForest to achieve a bet-
ter bias-variance trade-off via bias reduction from learning
and improve the computation efficiency. Extensive exper-
iments are performed on a suite of benchmarking datasets
in our ablation and comparative studies. The results val-
idate our proposed theory on optimal isolation forest, and
also show that with respect to detection performance, OptI-
Forest can generally outperform the state-of-the-arts includ-
ing the deep anomaly detection methods while maintaining a
high computation efficiency. The source code is available at
https://github.com/xiagll/OptIForest.

The main contributions of our work are threefold, sum-
marised as follows: (1) We are the first to formally investigate
the optimality problem of isolation tree structure with respect
to the branching factor and establish a theory on the optimal
isolation forest, offering a theoretical understanding for the
effectiveness of the isolation forest mechanism. (2) We in-
novatively propose a practical optimal isolation forest OptI-
Forest that can enhance both detection performance and com-
putational efficiency by designing a tailored clustering based
learning to hash for a good bias-variance trade-off. (3) Re-
sults from extensive experiments support our theory well and
validate the effectiveness and efficiency of our approach, as
well as the advantages over the state-of-the-arts.

## 2 Related Work

```
Various methods for unsupervised anomaly detection have
emerged, including distance-based, density-based, statistical,
ensemble-based, and deep learning-based methods[Chandola
et al., 2009; Panget al., 2021a]. This section delves into the
crucial methods related to our study.
Deep Learning-based Methods. Recently, deep neural net-
works are widely explored for anomaly detection, particu-
larly on complex data types[Zonget al., 2018; Zavrtanik
et al., 2021]. Techniques like generative adversarial net-
works (GANs)[Liuet al., 2019], AutoEncoders[Chenet al.,
2017 ], and reinforcement learning have been leveraged to en-
hance the detection performance[Panget al., 2021b]. For
example, a random distance-based anomaly detection method
called REPEN is proposed in[Panget al., 2018], where learn-
ing low-dimensional representation in random subsample is
optimised. While these deep methods can have high accu-
racy with learning feature representation[Zhaet al., 2020;
Zhanget al., 2021], they often suffer from issues like expen-
sive computations, complex hyperparameter tuning, etc.
Ensemble-based Methods. To achieve robust detection,
classical anomaly detection methods are integrated with en-
semble learning, e.g., ensemble LOF[Zimeket al., 2013],
isolation using Nearest Neighbour Ensemble (iNNE)[Ban-
daragodaet al., 2014], and averagek-NN distance ensem-
ble[Aggarwal and Sathe, 2015]. These methods suffer from
heavy computational cost when handling big data. A sim-
pler but more effective method is iForest[Liuet al., 2012]
which leverages the principle that anomalies are more likely
to be isolated from others. This pioneering work shows the
strong ability of the isolation forest mechanism and has been
widely adopted in real applications. A sequence of work on
isolation forests have been proposed to mitigate the shortcom-
ings in iForest, e.g., SCiForest[Liuet al., 2010]addresses the
failure of detecting axis-parallel anomalies and local anoma-
lies by using a optimsation strategy. Another interesting work
is LSHiForest which produces multi-fork isolation trees with
the use of LSH (locality-sensitive hashing) functions that can
hash similar data into the same hash value. The work for-
mally understands the distance metric underlying iForest and
enables the isolation forest mechanism widely applicable to
any data types where an LSH family can be defined. But why
a multi-fork isolation forest performs better has not been tack-
led in LSHiForest. Deep isolation forest[Xuet al., 2022]has
been recently proposed to incorporate isolation forest with
random feature representation, extending iForest to complex
data types. However, these works still fail to learn informa-
tion properly from data to reduce the bias of a base detector
for a better bias-variance trade-off.
```
## 3 Preliminaries and Problem Statement

```
LSHiForest Framework[Zhanget al., 2017].Our approach
is built on top of the LSHiForest Framework which is an ef-
fective and generic anomaly detection framework with the
forest isolation mechanism. While the algorithmic procedu-
ral and the way of deriving anomaly scores in LSHiForest
are quite similar to that in iForest, a key difference is that
LSHiForest makes use of a hash function to determine the
```

```
(a) (b) (c)
```
```
Figure 1: Isolating 9 data instances with different tree structures.
```
branching when isolating data in the recursive tree construc-
tion process, i.e., data instances with the same hash value go
into the same branch. As a result, the isolation trees can be
multi-fork, which is significantly different from the binary
case in iForest. Since the hash functions are drawn from an
LSH (Locality-Sensitive Hashing) family[Bawaet al., 2005],
the data instances falling into the same branch are similar to
each other with provably high probability and the isolation
process is more natural than binary splitting. Moreover, the
LSHiForest framework has high applicability to work with
any distance metric with an LSH family. In fact, iForest
and its variant have been proved to be two specific instances
of the framework with less commonly-used distance metrics.
Besides, other instances of the framework with specific dis-
tance metrics like Angular distance, Manhattan (ℓ 1 ) distance,
and Euclidean (ℓ 2 ) distance have been implemented as well,
and the instance with Euclidean distance (denoted as L2SH)
shows a very efficient, robust, and accurate detection perfor-
mance. Thus, we are motivated to use LSHiForest and its
L2SH instance as the foundation of our approach given their
prominent features and excellent performance.
Problem Statement.Although multi-fork isolation trees can
be elegantly constructed in LSHiForest and better detection
performance can be achieved, the theoretical understanding
of this phenomenon is still missing. Accordingly, an interest-
ing question arises naturally: What is the optimal branching
factor for an isolation tree? Since the branching factor influ-
ences the tree structure which can be regarded as a parameter
of the detection model, answering this question is fundamen-
tally important to understand how the branching factor affects
the performance of isolation forest based anomaly detection
methods, but intrinsically a challenging task. Fig. 1 shows an
example of isolating nine data instances with three different
tree structures, including 9 -fork tree, binary tree, and ternary
tree. It is hard to intuitively tell which structure is the best for
isolation, so how to address this problem is non-trivial.
Besides the tree structure, another challenge imposed on
existing isolation forest based anomaly detection methods is
how much information they should learn from the data to
facilitate data partition at each internal node. For instance,
iForest learns the minimum and maximum of the selected
feature to determine the random splitting value, and its vari-
ant SCiForest[Liuet al., 2010]learns more information to
determine the splitting hyperplane and gains better perfor-
mance. But it is worth noting that more learning does not
means better detection performance while the bias of a single
base detector can be reduced, according to the bias-variance
trade-off theory in ensemble learning[Aggarwal and Sathe,
2015 ]. For example, the recent work[Xuet al., 2022]ac-
tually reports that full learning for the neural networks has
poorer detection performance than the random weight initi-

```
ation. Because LSH hash functions used in LSHiForest are
data-independent, no learning is incorporated in LSHiForest
for data isolation. Thus, it is conjectured that the detection
performance could be further improved if an isolation tree
is constructed with the learning to hash technique[Wanget
al., 2017]which produces hash values based on learning in-
formation from the data. But learning to hash usually incurs
higher computational cost than LSH. Therefore, non-trivial
effort is required for the design of an anomaly detector based
on isolation forest with learning to hash, to target both high
computational efficiency and a good bias-variance trade-off.
```
## 4 Methodology

### 4.1 Optimal Isolation Forest

```
To answer the research question stated above about the op-
timal branching factor of an isolation tree, this section for-
mulates the problem to derive the solution and discusses its
practical implementation. LetTrepresents an isolation tree,
with the branching factorv, the tree depthd, and the num-
ber of leaf nodesψ. For the purpose of theoretical analy-
sis, we can reasonably allowv,d, andψto take real values
rather than just integers. Actually,vanddcan be regarded
as average branching factor of the internal nodes and the av-
erage height of the leaf nodes, respectively. To study how
the branching factor influences the isolation performance of
an isolation tree, we assume thatTis a perfect tree where all
internal nodes havevchildren and all leaves have the same
depthd. Note that the isolation tree structure may be totally
different in practice, but the assumption is the average case
which is reasonable in the ensemble learning setting. Given
Tis perfect, we can have the following relationship:
```
```
ψ=vd. (1)
```
```
Definition 1(Isolation Capacity). The maximum number of
data instances an isolation tree can isolate is defined as the
isolation capacity of the tree, denoted asψ, which is also the
number of leaf nodes.
```
```
Given an isolation capacity, it can be interestingly observed
that the width (controlled byv) and the depth (controlled by
d) of the tree compete with each other, i.e., if the branching
factor is smaller, the tree has to be deeper, or vice versa. So,
we can have the following definition to capture the overall
effect of branching factor and tree depth.
Definition 2(Isolation Area). The isolation area of an isola-
tion tree, denoted asφ, is defined as the product of the branch-
ing factorvwhich controls the width of a tree and the depth
d, i.e.,
φ=v·d. (2)
```
```
Note that isolation trees of the same isolation area can
achieve different isolation capacities. For example, given an
isolation areaφ= 6, a perfect binary tree (i.e.,v= 2and
d= 3) has the isolation capacityψ= 2^3 = 8, but a perfect
ternary tree (i.e.,v= 3andd= 2) has the isolation capacity
ψ= 3^2 = 9. The latter case seems more efficient, and we can
further define the concept of isolation efficiency as follows.
```

Definition 3(Isolation Efficiency).The isolation efficiency of
an isolation tree, denoted asη, is defined as the quotient of
the isolation capacity divided by the isolation area, i.e.,

```
η=
```
```
ψ
φ
```
#### . (3)

Isolation efficiency fundamentally affects the anomaly de-
tection performance of isolation forests because it is associ-
ated with the hardness of distinguishing data instances iso-
lated by a tree. Specifically, higher isolation efficiency leads
to stronger distinguishability. Continuing the example above,
the isolation efficiency of the binary tree isη= 8/ 6 ≃ 1. 33
and that of the ternary tree isη= 9/6 = 1. 5. Fig. 1(c) shows
that the ternary tree can isolate all the data, while Fig. 1(b)
shows that a binary tree with only 3 layers fails to do so.
As the branching factor indeed affects the detection perfor-
mance in terms of the analysis presented above, it is of both
theoretical and practical importance to study the problem of
the optimal branching factor. With the concepts introduced
above, we can formulate the problem of identifying the opti-
mal branching factorv∗as a constrained optimisation prob-
lem as follows.

```
v∗= argmax
v
```
```
η(v,d) =
```
```
vd
vd
```
#### ,

```
s.t.vd= Φ,
```
#### (4)

whereη(v,d)is the function of isolation efficiency with re-
spect to branching factorvand tree depthd, andΦis a con-
stant number representing the fixed isolation area. By solving
the optimisation problem, we can have the following theorem.

Theorem 1.An isolation treeThas the highest isolation effi-
ciency when its branching factorv=e, whereeis the Euler’s
number with numerical values around 2. 718.

The theorem can be proved with solving the optimisation
problem in (4). Please see Appendix A.1 for mode details,
where how the isolation efficiency changes with respect to
branching factor with a fixed isolation area is also illustrated.
This interesting result reveals that the commonly-used binary
tree is not the optimal for the forest isolation mechanism, and
explains the better detection accuracy and robustness gained
by the multi-fork LSHiForest instances like L2SH.

Definition 4(Optimal Isolation Tree). An isolation tree with
branching factorv=eis defined as an optimal isolation tree.

Definition 5(Optimal Isolation Forest). A forest consisting
of a set of optimal isolation trees is defined as an optimal
isolation forest.

Although the optimal isolation tree is theoretically promis-
ing, unfortunately there is no reale-fork branching can be
constructed directly in reality. As a practically viable way, we
can build an isolation tree with the average branching factor
equal toe. To achieve this, we need to generate the branching
factors during the tree construction. Let a random variableV
denote the branching factor for an isolation treeT, with the
sample space{v|v∈Z&v≥ 2 }. LetDbe a distribution
with the probability of taking valuevbeing: Pr(V=v) =pv.

```
We can generate the branching factors from the distributionD
if it satisfies the following condition:
```
#### E(V) =

#### +X∞

```
v=
```
```
v·pv=e. (5)
```
```
As the branching factor 2 is the only one less thane, there
should be a lower bound forp 2 and upper bounds forpv,v >
2 , to satisfy the condition mentioned above.
Theorem 2.To satisfy the condition in Eq.(5), the probability
of having a branching factorV≥v,v > 2 , should have the
following upper bound:
```
```
Pr(V≥v)≤
```
```
(e−2)
(v−2)
```
#### . (6)

```
The theorem can be proved by lettingpi= 0for 2 < i < v
and leveraging the inequality
```
#### P+∞

```
i=vipi≥v
```
#### P+∞

```
i=vpi. Please
see Appendix A.2 for more details. The upper bound re-
sult can give us an intuition on how the probability decreases
when the branching factor increases, e.g., the probability is
e− 2 ≃ 0. 718 forV≥ 3 , (e− 2 2) ≃ 0. 359 forV≥ 4 , and
(e−2)
3 ≃^0.^239 forV≥^5.
Corollary 1. To satisfy the condition in Eq.(5), the proba-
bility of sampling the branching factorV = 2should follow
p 2 ≥ 3 −e≃ 0. 282.
This corollary can be simply derived from Theorem 2 when
V≥ 3 , i.e.,p 2 = 1−Pr(V≥3)≤ 1 −(e−2) = 3−e.
There are many possible concrete distributions forDto be
specified, either finite or infinite. For example, by just using
binary and ternary branches, we can have a simple finite dis-
tribution as follows:p 2 = 3−e,p 3 =e− 2 , andpi= 0,
i≥ 4. For the infinite case, the probability is often a func-
tion of the branching factor. The work in[Russell, 1991]
derives a distributionpv=(v(v−!)1),v≥ 2 , from a stochastic
technique. Besides, we can have another infinite distribution
pv=(e−1)
```
```
2
2 e− 1 e
```
```
2 −v,v≥ 2 , to satisfy the condition in Eq.(5),
as proved in Appendix A.3. Understanding the distributions
of branching factors and the probability bounds can facilitate
the decision-makings when one tries to design a more practi-
cal optimal isolation forest, as shown in the following section.
```
### 4.2 OptIForest: Practical Detector Design with

### Clustering based Learning to Hash

```
In this section, we investigate how to implement a practical
optimal isolation tree. Since the data-independent LSH hash
functions cannot exactly produce a specified number of hash
values, we have to leverage learning to hash to achieve this for
a specified branching factor. Moreover, learning to hash can
capture more information from the data to benefit anomaly
detection potentially. There are many types of learning to
hash[Wanget al., 2017]. Given the higher accuracy and
less quantification loss, the non-parametric hash functionh(·)
based on nearest vector assignment is adopted herein, i.e.,
```
```
h(x) = argmin
k∈{ 1 ,···,v}
```
```
||x−ck||, (7)
```

where{c 1 ,···,cv}is a set of centres of data partitions. Clus-
tering algorithms are employed to compute the centres.
As our goal is to arrange the clusters into a natural hier-
archy to form an isolation tree, agglomerative hierarchical
clustering is adopted in our approach. The clustering method
treats each data instance as a cluster initially and merges simi-
lar clusters sequentially until a single cluster is left, forming a
hierarchical tree in a bottom-up fashion. While good cluster-
ing quality with less data distortion can be obtained, the high
computational complexities are often regarded as a downside.
With a cubic time complexity, clustering on a small dataset of
a fixed size (e.g., 256 in iForest, and 1024 in LSHiForest) still
seems too time-consuming. Straightforward adoption of the
clustering technique is neither effective nor efficient.
Fortunately, we find that the clustering technique can be
optimised based on two key observations in an isolation for-
est and design a more efficient clustering method. One obser-
vation is that anomalies are often located at upper levels of
an isolation tree while normal data at lower levels. The qual-
ity of clustering at upper levels is more sensitive to anomaly
detection than that at lower levels. The other observation is
that no learning or full learning in existing methods can lead
to poorer detection performance. Therefore, we can let the
upper levels of an isolation tree learn more from data to have
better clustering quality, while letting the lower levels learn
less, aiming to improve detection performance and save com-
putational cost simultaneously.
The basic idea of our approach is to use an isolation tree
efficiently produced in LSHiForest to initialise the clusters
for agglomerative hierarchical clustering so that the cluster-
ing process begins with bigger initial clusters rather than the
ones with a single data instance. This can maintain the high
clustering quality for upper levels of the hierarchy and save
much computational cost because the majority of computa-
tion occurs at lower levels. The specific steps for constructing
an optimal treeTOpt, are outlined in Algorithm 1, and details
are discussed subsequently.

Algorithm 1Constructing an Optimal Isolation Tree

Input:A dataset (sample)Dof sizeψ, and cut thresholdε.
Output:TOpt-an optimal isolation tree.
1:Train LSHiForest to getTLSH; ▷Pre-training
2:TraverseTLSHto getΓ(ε);
3:C 0 ←CΓ(ε);Γ 0 ←Γ(ε) ▷Initialisation
4:while|Ci|> 1 do
5: Generate a branching factorvfromD;
6: if|Ci|≤vthen
7: returnTOptwith(Nroot,C′)←merge(Ci);
8: Jcur←+∞,Ccur←NULL
9: for allCv={Ci 1 ,···,Civ}⊂Cido
10: ifdist(Cv)< Jcurthen
11: Jcur←dist(Cv),Ccur←Cv

12: (N′,C′)←merge(Ccur), ; ▷Optimal identified
13: Ci+1←Ci\Ccur,Ci+1←Ci+1∪{C′};
14: Γi+1←Γi\{Ni 1 ,···,Niv},Γi+1←Γi+1∪{N′};

```
LetTLSHdenote an isolation tree in LSHiForest. It can be
```
```
horizontally partitioned into two disjoint parts by a cut which
can be represented as a set of node. LetΓdenote a cut, and
Γ ={N 1 ,···,Nn 0 }, whereNi, 1 ≤i≤n 0 represents a
node, andn 0 is the number of the nodes inΓ. The nodeNiis
either a subtree ofTLSHor a leaf (a trivial subtree), and con-
tains a dataset which can be regarded as a cluster denoted as
Ci. Thus, we can obtain a set of clustersCΓ={C 1 ,···,Cn 0 }
from a cutΓ. Then, we can define the concept ofε-cut below.
Definition 6(ε-Cut). Given a cutΓand its associated clus-
tersCΓ={C 1 ,···,Cn 0 }, for any nodeNi∈Γ, if its cluster
size satisfies|Ci|≤ε <|Cp|, whereεis a threshold andCpis
the cluster associated with its parent node,Γis calledε-cut
and denoted asΓ(ε).
We can constructΓ(ε)by simply traversing the tree, and
use the associated clustersCΓ(ε)as the initial ones for ag-
glomerative clustering. By tuningε∈[1,ψ], whereψis the
sample size in isolation forests, we can control the number
and size of the clusters to further, and further manipulate the
degree of learning. A higherεleads to fewer initial clusters
and implies less learning. The initialisation degenerates into
the conventional case ifε= 1, while ifε=ψ, the resultant
isolation forest is stillTLSHwithout learning anything.
Unlike traditional agglomerative hierarchical clustering
where a binary tree is generated, we need to achieve multi-
fork branches for an optimal isolation tree. The distortion
measure is employed to capture the learning loss caused by
merging multiple clusters. LetμCdenote the merged centre
(mean) of a set of clustersC={C 1 ,···,Cv}, calculated by
```
```
μC=
```
```
Pv
Pi=1vμCi·ni
i=1ni
```
#### , (8)

```
whereμCis the centre of clusterC={x 1 ,···,xn}, calcu-
lated byμC=^1 n
```
```
Pn
i=1xi. Then, the distortion of merging
the clusters inCcan be calculated as
```
```
dist(C) =
```
```
Xv
```
```
i=
```
```
||μCi−μC||·ni. (9)
```
```
After generating the branching factorv(Line 5 in Algo-
rithm 1), Line 8-14 shows the details of performing cluster
merging. We need to try all the combinations ofvclusters
and evaluate their distortions. The one with minimum dis-
tortion is selected for merging, and a new node is created for
the resultant cluster. This process repeats until the number of
candidate clusters is not greater than the given branching fac-
tor. Finally, all the remaining clusters are merged as the root
node of the resultant isolation tree.
It is worth noting the produced tree is not an exact optimal
isolation tree ifεis not 1 , because this practical algorithm
takes the trade-off between accuracy and efficiency into ac-
count. The most time-consuming part is the evaluation of all
the combinations ofvclusters, whose time complexity grows
exponentially with respect tov. Thus, to be efficient, our
approach only leverages binary and ternary branches for the
tree construction, i.e., we adopt a finite distribution, which
is technically reasonable because the probability drops con-
siderably when the branching factor increases. But note that
other distributions can be used if other clustering techniques
like K-means are used.
```

```
Table 1: AUC-ROC and AUC-PR performance (mean±standard deviation) of all methods. OurOptIForestmethod outperforms others.
```
```
AUC-ROC (%) AUC-PR (%)
Datasets iForest LSHiForest ECOD REPEN RDP DiForest OptIForestiForest LSHiForest ECOD REPEN RDP DiForest OptIForest
AD 69.3±1.977.9±0.4 69.8±0 70.0±2.2 88.7±0.376.8±0.777.4±0.7 40.1±4.9 46.2±0.6 48.3±0 37.7±4.0 72.6±0.7 51.8±2.743.8±3.
campaign 70.9±1.0 67.7±0.6 77.5± 0 61.1±4.4 76.3±0.869.3±0.974.7±0.4 28.5±1.3 24.7±1.0 35.6± 0 17.8±3.9 37.2±0.927.5±1.332.0±0.
Arrhythmia79.7±1.077.5±0.5 82.3± 0 73.7±3.6 75.5±0.5 76.3±1.179.6±0.8 47.5±1.438.6±0.7 49.4± 0 37.4±3.5 32.0±0.6 38.2±1.245.1±1.
cardio 93.0±0.790.4±0.6 95.0± 0 91.5±2.9 88.1±0.693.0±0.592.8±1.3 57.0±2.8 48.9±1.0 67.6± 0 53.3±10.6 53.9±1.5 58.7±1.958.9±3.
backdoor 72.7±2.9 89.2±0.9 84.9±0 86.8±1.6 91.0±2.192.0±0.592.7±0.5 4.5±0.7 27.3±2.6 9.6± 0 12.5±1.7 3.5±0.8 39.4±3.351.7±8.
KDDCup9997.0±0.696.4±0.2 91.1±0 95.9±0.6 41.0±3.1 88.5±0.797.4±0.1 48.6±7.032.6±1.1 48.5± 0 44.1±1.9 15.4±0.9 16.7±0.543.0±0.
Celeba 69.4±2.4 72.5±0.7 72.3± 0 84.3±2.2 86.0±0.667.9±1.779.2±1.9 6.3±0.9 6.8±0.3 8.5± 0 10.7±1.8 10.4±0.65.5±0.6 8.1±1.
mnist 80.2±1.885.3±0.6 83.8±0 67.6±10.6 85.1±1.6 83.7±1.385.5±1.0 27.7±3.238.3±1.0 30.5±0 20.4±10.2 36.7±2.4 33.0±2.140.7±1.
Census 60.1±1.8 62.6±0.4 66.8± 0 62.7±1.5 65.3±0.4 59.4±1.167.8±1.0 7.1±0.3 7.5±0.1 8.6± 0 7.7±0.2 8.6±0.1 6.9±0.2 8.8±0.
Donors 76.6±1.0 74.5±0.7 74.0± 0 83.2±1.7 96.2±1.167.8±1.277.1±3.2 11.9±0.8 10.5±0.6 13.6± 0 15.5±1.3 43.2±6.18.0±0.3 11.3±1.
Cover 88.0±2.193.6±0.5 93.3± 0 86.6±5.7 51.2±1.3 76.4±4.090.9±0.8 6.4±0.9 9.0±0.8 11.6± 0 5.3±2.0 2.0±1.1 4.0±0.8 6.5±0.
http 99.9± 0 93.3± 0 97.9± 0 99.4±0.1 99.3±0.1 99.3±0.199.4±0.1 90.2±7.934.2±0.6 14.5± 0 39.5±2.4 36.2±0.8 35.1±0.435.4±1.
smtp 90.5±0.8 86.9±0.9 88.0± 0 90.9±1.3 69.8±1.1 84.6±0.592.4±0.6 0.4± 0 56.9±3.1 50.7±0 26.9±10.8 21.7±3.955.2±7.036.9±13.
Ionosphere 84.5±0.5 91.2±0.2 76.8±0 86.5±2.3 82.7±0.694.3±0.593.4±0.3 80.8±0.5 89.6±0.4 66.3±0 81.0±3.6 80.7±0.693.0±0.692.3±0.
Satellite 70.3±1.876.7±0.5 74.6±0 74.0±3.0 60.9±0.6 69.2±1.378.6±0.7 65.0±2.1 63.5±0.6 66.1± 0 70.3±3.9 61.3±0.7 47.5±1.371.5±0.
Shuttle 99.7±0.197.2±0.8 99.7± 0 99.4±0.1 99.6±0.9 96.3±1.298.0±0.4 97.6±0.540.1±2.7 95.0± 0 91.9±0.6 89.7±1.1 56.5±6.564.0±4.
Spam 61.7±2.9 70.7±0.2 65.6± 0 73.4±0.3 74.7±0.565.0±0.971.1±0.2 46.8±3.0 59.1±0.5 51.8± 0 62.7±0.5 63.0±0.259.6±0.558.1±0.
Vowel 75.3±1.290.8±0.7 40.8±0 77.9±5.5 67.2±1.294.4±1.490.0±1.2 12.6±1.7 29.5±2.1 2.8± 0 14.5±6.5 7.6±3.2 41.1±9.032.4±4.
Waveform 69.6±2.3 70.7±0.8 71.5± 0 63.2±9.5 66.9±0.9 63.2±3.474.4±1.3 9.7±0.9 9.8±0.4 9.4± 0 7.6±2.4 8.7±1.5 11.2±3.311.3±1.
Wine 63.9±0.767.0±0.3 62.5±0 64.6±2.2 61.7±0.4 65.9±1.266.4±0.5 7.8±0.4 8.9±0.2 8.1± 0 8.1±0.7 8.1±0.2 8.5±0.5 8.1±0.
Average 74.3±1.4 81.9±0.5 78.8±0 79.6±3.1 81.2±0.8 79.2±1.283.9±0.9 34.8±2.5 34.1±1.0 34.8±0 33.3±3.6 37.4±1.4 34.9±2.238.0±3.
```
Once a forest of optimal isolation trees are built as an opti-
mal isolation forest for OptIForest, it can perform anomaly
detection with the same way to derive anomaly scores as
LSHiForest. Although a practical isolation tree can consists
of two parts with the upper layers resulting from clustering
and the lower layers from LSHiForest, the hash function in-
terface makes the two implementations (LSH functions and
the non-parametric learning to hash function in Eq.(7)) no
difference as to anomaly score computation.

## 5 Experiments

### 5.1 Experiment Setting

Baselines. Our method OptIForest is compared with six
state-of-the-art anomaly detection methods: iForest, LSHi-
Forest, ECOD, REPEN, RDP, and DiForest. We evaluate all
methods on 20 widely-used benchmark datasets[Panget al.,
2019; Hanet al., 2022; Liet al., 2022]. We refer the reader
to Appendix B for more details of the baselines and datasets.
Metrics. We conventionally use the Area Under Receiver
Operating Characteristic Curve (AUC-ROC) and Area Un-
der the Precision-Recall Curve (AUC-PR) as the performance
metrics[Kurtet al., 2020; Wanget al., 2021]. To ensure a fair
comparison, we follow the optimal parameter settings of the
baseline methods. Please see Appendix C for more details
about the metrics and the parameter settings. All experiments
are run 15 times and averaged results are reported.

### 5.2 Comparison Study Results and Discussion

AUC-ROC Results.Table 1 illustrates that OptIForest is the
most robust method that achieves the best performance on
most datasets. Specifically, OptIForest has the highest aver-
age AUC-ROC score among all the compared methods, sur-
passing the second best method by 2%. Additionally, RDP
and LSHiForest perform well and are stable on most datasets.
ECOD produces exceptional results on certain datasets, but
performs poorly on others like “Vowel” and “CD” due to its
requirement on the data distribution. Note that ECOD has

```
zero standard deviation because it is a deterministic method.
The rest methods iForest, REPEN, and DiForest only per-
form well on a few datasets, indicating their lack of robust-
ness across all datasets. Thus, it can be seen that no learning
in iForest or DiForest fail to achieve a robust performance
due to the lack of knowledge learned from data. LSHiFor-
est and OptIForest with better tree structures can achieve bias
reduction, and learning from data further makes OptIForest
perform better and even outperform the deep detector.
AUC-PR Results.The results in Table 1 show that OptIFor-
est consistently performs best across various datasets with the
highest average AUC-PR score, showing the superiority of
our method. Specifically, our method outperforms the second
best one by about 0.6%. The top two results of each dataset
are highlighted in bold. Although iForest and ECOD have
exceptional results on a few datasets, they perform poorly on
others. REPEN has the lowest average AUC-PR score and
performs poorly on most datasets. The AUC-PR results show
the same trend as the AUC-ROC results, supporting that our
OptIForest performs the best performance in a robust way.
Execution Time. We use execution time as the efficiency
metric to compare OptIForest with other methods, and the re-
sults are reported in Table 3 in Appendix D.1. OptIForest has
much shorter execution time than the deep learning methods
of REPEN and RDP for most datasets. As expected, OptIFor-
est reasonably has longer execution time than other isolation
forests where no learning is conducted. So, it can be seen
that our approach strikes a good balance between execution
efficiency and detection performance. It is appealing that Op-
tIForest can achieve better performance than deep learning
methods but takes much less execution cost.
```
### 5.3 Ablation Study Results and Discussion

```
To understand how the branching factor, the cut threshold,
and the sampling size influence the performance of OptIFor-
est, we performed detailed ablation studies on eight datasets
with a range of data types and sizes.
```

```
AD cardio
backdoorShuttlesmtp VowelWaveformWine
Datasets
```
```
60
```
```
70
```
```
80
```
```
90
```
```
100
```
```
AUC-ROC (%)
```
```
v=
v e
v 3
v 4
```
Figure 2: Detection performance changes w.r.t. branching factorv.
A branching factor closer toeleads to a better performance.

Branching Factor. To eliminate the effects of feature
learning, we use data-independent baseline (i.e.,ε=ψ) to
analyze the impact of the branching factor. Besides, it is dif-
ficult to implement the average branch ofein practical ex-
periment because many of the underlying branches just pro-
duce 2 branch forks. However, we can still analyze the AUC-
ROC results for branches that are either near or far frome.
The AUC-ROC results for different branching factors are pre-
sented in Fig. 2. When the branching factor is close toe, the
AUC-ROC results are the best on almost all datasets. This
result means that the best anomaly detection accuracy can be
achieved if the branching factor satisfies the condition:v≈e.
These experimental results validate Theorem 1 in Section 4.1.

Cut Thresholdε. The cut threshold is studied by raising
the branching factor to the power ofe, as small changes of the
threshold do not greatly affect detection accuracy. We also
study the boundary condition of the cut threshold (ε= 512).
Fig. 3 displays AUC-ROC results and standard deviations for
differentε. It can be seen that the curve increases as the
threshold increases in four large datasets with data sizes larger
than 10,000 or dimension sizes larger than 1,000, indicating
little learning is required for large datasets (withε=e^6 as
a reference). Conversely, more learning is necessary in four
small datasets, andε=e^4 serves as a good reference point to
balance accuracy and time efficiency. A comparison between
the results ofε= 512with others illustrates that appropri-
ate learning results in better outcomes than not learning on
most datasets. But it is exceptional in the “AD” and “vowel”
datasets, where not learning yields better results. This could
be attributed to the fact that the isolation forest without learn-
ing exactly has the average branch ofe, which achieves a fa-
vorable bias-variance trade-off.

Sampling Size. To facilitate the observation of the re-
sults, the sampling size is set from 26 to 211 with exponential
increase, as the results in this range can display significant
change. The cut threshold of the same sampling size will im-
pact the AUC-ROC results, so we determine the best results
for each sampling group as the final outcomes. It can be con-
cluded from Fig. 4 that the AUC-ROC results improve with
the increase of the sampling size on most datasets. The AUC-

```
e^2 e^3 e^4 e^5 e^6512
Cut threshold
```
```
60
```
```
70
```
```
80
```
```
90
```
```
100
```
```
AUC-ROC (%)
```
```
AD
cardio
```
```
backdoor
Shuttle
```
```
smtp
Vowel
```
```
Waveform
Wine
```
```
Figure 3: Detection performance changes w.r.t. cut thresholdε. In
general, a certain level of learning outperforms the case of no learn-
ing (ε= 512means no learning).
```
```
26 27 28 29 210 211
Sampling size
```
```
60
```
```
70
```
```
80
```
```
90
```
```
100
```
```
AUC-ROC (%)
```
```
AD
cardio
```
```
backdoor
Shuttle
```
```
smtp
Vowel
```
```
Waveform
Wine
```
```
Figure 4: Detection performance changes w.r.t. sampling sizeψ. In
general, the performance is saturated whenψreaches a threshold
( 29 = 512herein).
```
```
ROC results remain stable once the sampling size exceed 29 ,
except for the “cardio” and “Shuttle” datasets, which keep
stable on any sampling size.
```
## 6 Conclusion and Future Work

```
Isolation Forest is an appealing anomaly detection method
given its salient characteristics, but it lacks a theoretical foun-
dation about the structure optimality of an isolation tree. In
this paper, we have introduced the concept of isolation effi-
ciency and formulated a constrained optimisation problem to
derive the optimal branching factor for an isolation tree. We
have shown that an optimal isolation tree or forest is theoret-
ically with the branching factore. Furthermore, we have de-
veloped a practical optimal isolation forest OptIForest which
can achieve both high computational efficiency and a good
bias-variance trade-off by designing a novel clustering based
learning to hash for data isolation. We have conducted exten-
sive experiments on a variety of benchmarking datasets for
both ablation and comparative studies, and the results have
confirmed the effectiveness and efficiency of OptIForest. In
the future, we plan to design a version in the context of feder-
ated learning where data are scattered across multiple clients.
```

## References

[Aggarwal and Sathe, 2015]Charu C Aggarwal and Saket
Sathe. Theoretical foundations and algorithms for outlier
ensembles.Acm sigkdd explorations newsletter, 17(1):24–
47, 2015.

[Ahmedet al., 2016]Mohiuddin Ahmed, Abdun Naser
Mahmood, and Jiankun Hu. A survey of network anomaly
detection techniques. Journal of Network and Computer
Applications, 60:19–31, 2016.

[Bandaragodaet al., 2014] Tharindu R Bandaragoda,
Kai Ming Ting, David Albrecht, Fei Tony Liu, and
Jonathan R Wells. Efficient anomaly detection by
isolation using nearest neighbour ensemble. InIEEE
International conference on data mining workshop (ICDM
workshop), pages 698–705. IEEE, 2014.

[Bawaet al., 2005]Mayank Bawa, Tyson Condie, and
Prasanna Ganesan. Lsh forest: self-tuning indexes for sim-
ilarity search. InProceedings of the 14th international
conference on World Wide Web (WWW), pages 651–660,
2005.

[Chandolaet al., 2009]Varun Chandola, Arindam Banerjee,
and Vipin Kumar. Anomaly detection: A survey. ACM
computing surveys, 41(3):1–58, 2009.

[Chen and Tsourakakis, 2022] Tianyi Chen and Charalam-
pos Tsourakakis. Antibenford subgraphs: Unsupervised
anomaly detection in financial networks. InProceedings
of the 28th ACM SIGKDD Conference on Knowledge Dis-
covery and Data Mining (SIGKDD), pages 2762–2770,
2022.

[Chenet al., 2017]Jinghui Chen, Saket Sathe, Charu Aggar-
wal, and Deepak Turaga. Outlier detection with autoen-
coder ensembles. InProceedings of the 2017 SIAM inter-
national conference on data mining (SDM), pages 90–98.
SIAM, 2017.

[Fernandoet al., 2021] Tharindu Fernando, Harshala Gam-
mulle, Simon Denman, Sridha Sridharan, and Clinton
Fookes. Deep learning for medical anomaly detection–a
survey.ACM Computing Surveys, 54(7):1–37, 2021.

[Hanet al., 2022] Songqiao Han, Xiyang Hu, Hailiang
Huang, Minqi Jiang, and Yue Zhao. Adbench: Anomaly
detection benchmark. Advances in Neural Information
Processing Systems (NeurIPS), 2022.

[Haririet al., 2019] Sahand Hariri, Matias Carrasco Kind,
and Robert J Brunner. Extended isolation forest.
IEEE Transactions on Knowledge and Data Engineering,
33(4):1479–1489, 2019.

[Kurtet al., 2020]Mehmet Necip Kurt, Yasin Yilmaz, and
Xiaodong Wang. Real-time nonparametric anomaly de-
tection in high-dimensional settings. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 2020.

[Liet al., 2022] Zheng Li, Yue Zhao, Xiyang Hu, Nicola
Botta, Cezar Ionescu, and George Chen. Ecod: Unsuper-
vised outlier detection using empirical cumulative distri-
bution functions. IEEE Transactions on Knowledge and
Data Engineering, 2022.

```
[Liuet al., 2008] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua
Zhou. Isolation forest. InIEEE international conference
on data mining (ICDM), pages 413–422, 2008.
[Liuet al., 2010] Fei Tony Liu, Kai Ming Ting, and Zhi Hua
Zhou. On detecting clustered anomalies using scifor-
est. InJoint European Conference on Machine Learning
and Knowledge Discovery in Databases (ECML-PKDD),
pages 274–290. Springer, 2010.
[Liuet al., 2012] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua
Zhou. Isolation-based anomaly detection. ACM Trans-
actions on Knowledge Discovery from Data, 6(1):1–39,
2012.
[Liuet al., 2019] Yezheng Liu, Zhe Li, Chong Zhou,
Yuanchun Jiang, Jianshan Sun, Meng Wang, and Xiang-
nan He. Generative adversarial active learning for unsu-
pervised outlier detection.IEEE Transactions on Knowl-
edge and Data Engineering, 32(8):1517–1528, 2019.
[Panget al., 2018] Guansong Pang, Longbing Cao, Ling
Chen, and Huan Liu. Learning representations of
ultrahigh-dimensional data for random distance-based out-
lier detection. InProceedings of the 24th ACM SIGKDD
international conference on knowledge discovery & data
mining (SIGKDD), pages 2041–2050, 2018.
[Panget al., 2019] Guansong Pang, Chunhua Shen, and An-
ton van den Hengel. Deep anomaly detection with devia-
tion networks. InProceedings of the 25th ACM SIGKDD
international conference on knowledge discovery & data
mining (SIGKDD), pages 353–362, 2019.
[Panget al., 2021a]Guansong Pang, Chunhua Shen, Long-
bing Cao, and Anton Van Den Hengel. Deep learning for
anomaly detection: A review. ACM Computing Surveys,
54(2):1–38, 2021.
[Panget al., 2021b] Guansong Pang, Anton van den Hengel,
Chunhua Shen, and Longbing Cao. Toward deep super-
vised anomaly detection: Reinforcement learning from
partially labeled anomaly data. InProceedings of the 27th
ACM SIGKDD conference on knowledge discovery & data
mining (SIGKDD), pages 1298–1308, 2021.
[Ruffet al., 2021]Lukas Ruff, Jacob R Kauffmann,
Robert A Vandermeulen, Gr ́egoire Montavon, Woj-
ciech Samek, Marius Kloft, Thomas G Dietterich, and
Klaus-Robert Muller. ̈ A unifying review of deep and
shallow anomaly detection. Proceedings of the IEEE,
109(5):756–795, 2021.
[Russell, 1991]KG Russell. Estimating the value of e by
simulation.The American Statistician, 45(1):66–68, 1991.
[Wanget al., 2017]Jingdong Wang, Ting Zhang, Nicu Sebe,
Heng Tao Shen, et al. A survey on learning to hash.IEEE
transactions on pattern analysis and machine intelligence,
40(4):769–790, 2017.
[Wanget al., 2021]Hu Wang, Guansong Pang, Chunhua
Shen, and Congbo Ma. Unsupervised representation learn-
ing by predicting random distances. InInternational Joint
Conference on Artificial Intelligence (IJCAI), pages 2950–
2956, 2021.
```

[Xuet al., 2022]Hongzuo Xu, Guansong Pang, Yijie Wang,
and Yongjun Wang. Deep isolation forest for anomaly de-
tection.arXiv preprint arXiv:2206.06602, 2022.

[Zavrtaniket al., 2021]Vitjan Zavrtanik, Matej Kristan, and
Danijel Skocaj. Reconstruction by inpainting for visualˇ
anomaly detection. Pattern Recognition, 112:107706,
2021.

[Zhaet al., 2020]Daochen Zha, Kwei-Herng Lai, Mingyang
Wan, and Xia Hu. Meta-aad: Active anomaly detection
with deep reinforcement learning. InIEEE International
conference on data mining (ICDM), pages 771–780. IEEE,
2020.

[Zhanget al., 2017]Xuyun Zhang, Wanchun Dou, Qiang
He, Rui Zhou, Christopher Leckie, Ramamohanarao Ko-
tagiri, and Zoran Salcic. Lshiforest: A generic frame-
work for fast tree isolation based ensemble anomaly analy-
sis. InIEEE international conference on data engineering
(ICDE), pages 983–994, 2017.

[Zhanget al., 2021]Huayi Zhang, Lei Cao, Peter VanNos-
trand, Samuel Madden, and Elke A Rundensteiner. Elite:
Robust deep anomaly detection with meta gradient. In
Proceedings of the 27th ACM SIGKDD Conference on
Knowledge Discovery & Data Mining (SIGKDD), pages
2174–2182, 2021.

[Zimeket al., 2013]Arthur Zimek, Matthew Gaudet, Ri-
cardo JGB Campello, and Jorg Sander. Subsampling for ̈
efficient and effective unsupervised outlier detection en-
sembles. InProceedings of the 19th ACM SIGKDD in-
ternational conference on Knowledge discovery and data
mining (SIGKDD), pages 428–436, 2013.

[Zonget al., 2018]Bo Zong, Qi Song, Martin Renqiang
Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, and
Haifeng Chen. Deep autoencoding gaussian mixture
model for unsupervised anomaly detection. In Inter-
national conference on learning representations (ICLR),
2018.


## A Appendix A

### A.1 Proof of Theorem 1

Theorem 1.An isolation treeThas the highest isolation effi-
ciency when its branching factorv=e, whereeis the Euler’s
number with numerical values around 2. 718.

Proof.According to the Definition of the isolation efficiency,
to obtain the optimal isolation tree is to maximise the isola-
tion efficiency, which is formulated as:

```
η(v,d) =
```
```
ψ
φ
```
#### =

```
vd
vd
```
#### , (A.1)

wherevrepresents the branching factor,drepresents the tree
depth. After fixing the isolation area by a constraint num-
berΦ, we can obtain the function of isolation efficiency with
respect tov:

```
η(v) =
```
#### 1

#### Φ

```
v
```
```
Φv
```
. (A.2)

The derivative ofη(v)can be derived by:

```
η′(v) =v(
```
```
Φ
v−2)(1−lnv). (A.3)
```
Becausev(

Φv−2)
> 0 , if 1 −lnv > 0 , we can haveη′(v)> 0 ,
and if 1 −lnv < 0 , we can haveη′(v)< 0. Thus,η(v)is a
convex function and has a maximum when 1 −lnv= 0, i.e.,
the optimal branching factionv∗=e.
Also, we can visualize the relationship betweenvandη(v).
Without loss of generality, Fig. A.1 demonstrates an example
whereΦis fixed at 6 (other values should show the same trend
ofη(v)with respect tov). It can be seen that the highest
isolation efficiency is attained when the branch factor is equal
toe.

```
0 2 4 6 8 10 12 14
Branching factor ( v )
```
```
0
```
```
0.
```
```
0.
```
```
1.
```
```
1.
```
```
Isolation efficiency ( )
```
```
( e ,^16 e
```
(^6) _e_
) ( _v_ ) =^1
6 _v_
(^6) _v_
Figure A.1: The relationship between isolation efficiencyη(v)and
branching factorv. We useΦ = 6as an example for an illustration
without loss of generality.

### A.2 Proof of Theorem 2

Theorem 2.To satisfy the condition in Eq.(5), the probabil-
ity of having a branching factorV≥v,v≥ 2 , should have the
following upper bound:

```
Pr(V≥v)≤
```
```
e− 2
v− 2
```
#### . (A.4)

```
Proof.When the average branching factor is equal toe, we
can formalise the branching factor and the corresponding
probability by:
 P+∞
Pi+=2∞i·pi =e,
i=2pi = 1,
```
#### (A.5)

```
wherepicorresponds to the probability to produceibranches
and the sum of the probabilities of producing different
branches is equal to 1. To calculate the upper bond of
Pr(V≥v), we should makepi= 0for 2 < i < v, and then
leverage the inequality:
+X∞
```
```
i=v
```
```
ipi≥v
```
#### +X∞

```
i=v
```
```
pi. (A.6)
```
```
Combining the condition thatpi= 0for 2 < i < v, we can
rewrite Eq.(A.5) as:

2 p 2 +
```
#### P+∞

```
i=vi·pi =e,
p 2 +
```
#### P+∞

```
i=vpi = 1.
```
#### (A.7)

```
Then, we can have the following equation:
```
```
e−2 =
```
#### +X∞

```
i=v
```
```
ipi− 2
```
#### +X∞

```
i=v
```
```
pi,v > 2. (A.8)
```
```
By substituting Eq.(A.6) into Eq.(A.8), we can obtain:
```
```
e− 2 ≥
```
#### X+∞

```
i=v
```
```
ipi− 2
```
#### +X∞

```
i=v
```
```
pi= (v−2)
```
#### X+∞

```
i=v
```
```
pi,v > 2. (A.9)
```
```
Finally, we can have the following inequality relationship:
+X∞
```
```
i=v
```
```
pi≤
```
```
e− 2
v− 2
```
```
,v > 2. (A.10)
```
```
Thus, the upper bound of Pr(V≥v)isve−−^22 ,v > 2.
```
### A.3 An Instantiation of the Probability

### Distribution ofV

```
Let a random variableVdenote the branching factor for an
isolation treeTwith the sample space{v|v∈Z&v≥ 2 }.
LetDbe a distribution with the probability of taking valuev
being: Pr(V=v) =pv. We can generate the branching fac-
tors from the distributionD. Then, we can have the following
theorem with a proof.
Proposition A.1.If the distributionDis instantiated by as-
signing the probability of taking the branching factorvas
pv=(e−1)
```
```
2
2 e− 1 e
```
```
2 −v,v≥ 2 , the resultant distribution satisfies
```
```
the condition:E(V) =
```
#### P+∞

```
v=2v·pv=e.
Proof.
```
```
E(V) =
```
#### +X∞

```
v=
```
```
v·Pv
```
#### =

#### +X∞

```
v=
```
```
v·
```
```
e(e−1)^2
2 e− 1
```
```
·e^1 −v
```
#### =

```
e(e−1)^2
2 e− 1
```
#### +X∞

```
v=
```
```
v·e^1 −v.
```
#### (A.11)


For convenience, letZdenote the quantity

#### P+∞

```
v=2v·e
```
```
1 −v, i.e.,
```
#### Z≜

#### +X∞

```
v=
```
```
v·e^1 −v. (A.12)
```
Multiplying both sides of the Eq.(A.12) bye−^1 can result in:

```
e−^1 ·Z=
```
#### +X∞

```
v=
```
```
v·e−v. (A.13)
```
Subtracting Eq.(A.13) from Eq.(A.12), we can derive the fol-
lowing result by using the formula of computing the sum of a
geometric series with a common ratioe−^1 < 1.

```
Z−e−^1 Z= 2e−^1 +e−^2 +e−^3 +...+e−v+...
=e−^1 + (e−^1 +e−^2 +e−^3 +...+e−v+...)
```
```
=e−^1 +
```
```
e−^1
1 −e−^1
```
```
=e−^1 +
```
#### 1

```
e− 1
```
#### .

#### (A.14)

According to Eq.(A.14), the value ofZcan be calculated by:

#### Z=

```
e−^1 +e−^11
1 −e−^1
```
#### =

```
1 +e−e 1
e− 1
```
#### =

```
2 e− 1
(e−1)^2
```
#### . (A.15)

Finally, the expectationE(V)can be calculated by Eq.(A.11),
(A.12) and (A.15):

#### E(V) =

```
e(e−1)^2
2 e− 1
```
#### ·Z=

```
e(e−1)^2
2 e− 1
```
#### ·

```
2 e− 1
(e−1)^2
```
```
=e. (A.16)
```
## B Appendix B

### B.1 Baselines

Our method OptIForest is compared with six state-of-the-
art anomaly detection methods: iForest, LSHiForest, ECOD,
REPEN, RDP, and DiForest, which are conventionally used
for empirical study in many prior works. While there are
many other anomaly detection methods, these six state-of-
the-arts are chosen with the rationale that they are either very
fresh or have shown superior performance in most cases.
These methods are categorised into shallow anomaly detec-
tion methods and deep anomaly detection methods, briefly
described as follows:

- Shallow Anomaly Detection Methods. iForest[Liu
    et al., 2008]is a seminal anomaly detection method,
    which can isolate data instances very efficiently with
    an ensemble of binary trees and usually achieves good
    performance. iForest has been widely recognised in
    academia and deployed in real applications, e.g., it has
    been included inscikit-learn^1 , a commonly-used ma-
    chine learning library in Python. LSHiForest[Zhang
    et al., 2017]is a generic framework, which generalise
    the forest isolation mechanism with the multi-fork tree

(^1) https://scikit-learn.org
structure and achieves higher performance and applica-
bility. We select its instance L2SH which in general
has the best performance with respect to accuracy and
robustness for the comparison study. The source code
is available in a public GitHub repository^2. ECOD[Li
et al., 2022]is a very fresh anomaly detection method,
which is parameter-free and easy to interpret. ECOD
uses the empirical distributions of the input data to esti-
mate tail probabilities per dimension for each data point.
This method is simple but effective on many benchmark
datasets. The source code is available in a public GitHub
repository^3.

- Deep Anomaly Detection Methods.REPEN[Panget
    al., 2018]is a random distance-based anomaly detection
    method, which utilises deep representation learning and
    distance calculation to learn low-dimensional represen-
    tations. REPEN has been widely accepted as a bench-
    mark for deep anomaly detection and its source code is
    available in a public GitHub repository^4. RDP[Wang
    et al., 2021]trains neural networks to predict the ab-
    normalities of data distances in a randomly projected
    space, where the genuine class structures are learned and
    implicitly embedded in the randomly projected space.
    RDP is a state-of-the-art deep anomaly detection method
    and achieves good performance on many datasets. The
    source code can be found in a public GitHub repository^5.
    DiForest[Xuet al., 2022]is a very recent anomaly de-
    tection approach that utilises neural networks to learn
    representations and these representations are then used
    to construct isolation forests for anomaly detection. The
    source code is available in a public GitHub repository^6.

### B.2 Datasets

```
We conduct experiments on 20 real-world datasets from dif-
ferent fields including finance, healthcare, network, etc. All
datasets are available in public repositories like UCI Machine
Learning Repository^7 , Kaggle Repository^8 , and ADReposi-
tory^9. The basic information about the datasets is summarised
in Table 2.
```
## C Appendix C

### C.1 Parameter Settings

```
Our method uses 100 isolation trees as the base detector for
the isolation forest, the same as the existing isolation forest
based methods. In section 5.3, we study the relationship be-
tween the sampling size and the detection performance and
observe the performance of our method is saturated when the
sampling size reaches a threshold ( 29 = 512herein). Thus,
```
(^2) https://github.com/xuyun-zhang/LSHiForest
(^3) https://github.com/yzhao062/pyod
(^4) https://github.com/Minqi824/ADBench/tree/main/baseline
(^5) https://git.io/RDP
(^6) https://github.com/xuhongzuo/deep-iforest
(^7) https://archive.ics.uci.edu/ml/datasets.php
(^8) https://www.kaggle.com/datasets
(^9) https://github.com/GuansongPang/
ADRepository-Anomaly-detection-datasets


Table 2: A summary of datasets used in the experiments. Here,
“n” in the table denotes the size of the dataset, “m” denotes the
dimension of the dataset, and “Rate” represents the proportion of
anomalous data in total data instances.

```
Dataset n m Rate(%) Category
AD 3,279 1,555 13.79 Finance
campaign 41,188 62 11.27 Finance
Arrhythmia 452 274 14.60 Healthcare
cardio 1,831 21 9.61 Healthcare
backdoor 95,329 196 2.44 Network
KDDCup99 494,021 38 1.77 Network
Celeba 202,599 39 2.24 Image
mnist 7,603 100 9.21 Image
Census 299,285 500 6.20 Sociology
Donors 619,326 10 5.92 Sociology
Cover 286,048 10 0.96 botany
http 567,498 3 0.39 Web
smtp 95,156 3 0.03 Web
Ionosphere 351 34 35.90 Oryctognosy
Satellite 6,435 36 31.60 Astronautics
Shuttle 49,097 9 7.15 Astronautics
Spam 4,207 57 39.91 Document
Vowel 1,456 12 3.43 Linguistics
Waveform 3,505 21 4.62 Physics
Wine 5,318 11 4.53 Chemistry
```
the size of the sample used for constructing each tree is 512
in our method. Besides, we study how the cut threshold in-
fluence the detection performance in section 5.3 and observe
that the selection of cut thresholdεis influenced by the size
of dataset, i.e., the large datasets perform well with a big cut
threshold (e.g.,ε= 403), while the small datasets perform
well with a small cut threshold (e.g.,ε= 55). To make a fair
comparison, our method will use the above optimal settings to
get the results. In the compared methods, all the parameters
are set as the optimal settings in the original papers.

### C.2 Evaluation Metrics

We use the Area Under Receiver Operating Characteris-
tic Curve (AUC-ROC) and Area Under the Precision-Recall
Curve (AUC-PR) as the accuracy evaluation criterion[Kurt
et al., 2020; Wanget al., 2021]. Before explaining two eval-
uation metrics, we need to introduce four indicators derived
from the confusion matrix, including Recall, Precision, True
Positive Rate (TPR), and False Positive Rate (FPR). These
four indicators can be calculated as:

```
Recall=
```
#### TP

#### TP+FN

#### ,

```
Precision=
```
#### TP

#### TP+FP

#### ,

```
True Positive Rate=
```
#### TP

#### TP+FN

#### ,

```
False Positive Rate=
```
#### FP

#### FP+TN

#### ,

whereTrepresents the original object or event is true,Frep-
resents the original object or event is false,Prepresents the
object is predicted as a positive example by the classifier,N

```
Table 3: Comparing execution time (s) of all methods. It is worth
noting that OptIForest has much shorter execution time than the deep
learning methods of REPEN and RDP for most datasets.
```
```
Dataset iForest LSHiForest ECOD REPEN RDP DiForest OptIForest
AD 10 26 4 23 7,327 14 83
campaign 7 294 3 584 7,384 111 637
Arrhythmia 0.3 6 0.2 16 6,037 2 58
cardio 0.3 26 0.1 9 6,508 6 68
backdoor 26 758 18 2,647 5,267 245 1,
KDDCup99 40 2,409 18 38,561 6,991 634 5,
celeba 22 1,055 7 14,334 8,026 579 2,
mnist 2 35 1 43 7,482 22 132
census 205 1,911 185 30,794 8,991 883 4,
donors 24 3,901 7 51,496 7,655 1,367 6,
Cover 14 1,474 6 27,930 6,472 732 3,
http 19 3,557 4 91,980 6,098 1,322 7,
smtp 4 501 1 7,115 6,422 203 1,
Ionosphere 0.3 6 0.1 3 3,787 2 69
Satellite 1 36 0.4 213 4,649 17 209
Shuttle 2 313 1 1,060 4,928 124 662
Spam 1 15 0.3 123 6,316 10 83
Vowel 0.3 12 0.1 60 3,743 5 50
Waveform 0.5 17 0.2 112 3,839 10 109
Wine 0.4 29 0.1 156 4,613 15 126
```
```
represents the object is predicted as a negative example by the
classifier. Then,TPcan be explained as the number of true
objects that are predicted to be positive by the classifier. The
total number of data instances, denoted asS, is formalised as
S=TP+FP+TN+FN. The ROC curve takes TPR as
the Y-axis and FPR as the X-axis, so the value of AUC-ROC
is the area under the ROC curve. Similarly, the PR curve
summarises the relation between Precision and Recall. The
value of AUC ranges from 0 to 1, where a larger AUC result
indicates better performance. The result of AUC has been
extensively adopted in many anomaly detection works and
has become an essential measure of accuracy in correlational
research. Furthermore, the execution time is used as the effi-
ciency evaluation criterion. All experiments are run 15 times
and averaged results are reported.
```
## D Appendix D

### D.1 Execution Time

```
We use execution time as the efficiency metric to compare
OptIForest with other methods on 20 real world datasets ,
and the results are reported in Table 3. It can be seen in Ta-
ble 3 that our method OptIForest has much shorter execution
time than the deep anomaly detetion methods of REPEN and
RDP for most datasets. It is worth noting that the more our
method learns, the more time it will spend. As previously
discussed in the AUC comparison, achieving optimal perfor-
mance on small datasets necessitates more learning, resulting
in longer execution time compared to REPEN on some small
datasets. As expected, OptIForest reasonably has longer ex-
ecution time than other isolation forests where no learning is
conducted. So, it can be concluded that our approach strikes a
good balance between execution efficiency and detection per-
formance. It is appealing that OptIForest can achieve better
performance than deep learning methods but takes much less
execution cost.
```

