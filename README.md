# 【关于 NLP】 那些你不知道的事——信息抽取篇

> 作者：杨夕
> 
> 介绍：研读顶会论文，复现论文相关代码
> 
> NLP 百面百搭 地址：https://github.com/km1994/NLP-Interview-Notes
> 
> **[手机版NLP百面百搭](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=3&sn=5d8e62993e5ecd4582703684c0d12e44&chksm=1bbff26d2cc87b7bf2504a8a4cafc60919d722b6e9acbcee81a626924d80f53a49301df9bd97&scene=18#wechat_redirect)**
> 
> 推荐系统 百面百搭 地址：https://github.com/km1994/RES-Interview-Notes
> 
> **[手机版推荐系统百面百搭](https://mp.weixin.qq.com/s/b_KBT6rUw09cLGRHV_EUtw)**
> 
> 搜索引擎 百面百搭 地址：https://github.com/km1994/search-engine-Interview-Notes 【编写ing】
> 
> NLP论文学习笔记：https://github.com/km1994/nlp_paper_study
> 
> 推荐系统论文学习笔记：https://github.com/km1994/RS_paper_study
> 
> GCN 论文学习笔记：https://github.com/km1994/GCN_study
> 
> **推广搜 军火库**：https://github.com/km1994/recommendation_advertisement_search
![](other_study/resource/pic/微信截图_20210301212242.png)

> 手机版笔记，可以关注公众号 **【关于NLP那些你不知道的事】** 获取，并加入 【NLP && 推荐学习群】一起学习！！！

> 注：github 网页版 看起来不舒服，可以看 **[手机版NLP论文学习笔记](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=1&sn=14d34d70a7e7cbf9700f804cca5be2d0&chksm=1bbff26d2cc87b7b9d2ed12c8d280cd737e270cd82c8850f7ca2ee44ec8883873ff5e9904e7e&scene=18#wechat_redirect)**

- [【关于 NLP】 那些你不知道的事——信息抽取篇](#关于-nlp-那些你不知道的事信息抽取篇)
  - [介绍](#介绍)
    - [NLP 学习篇](#nlp-学习篇)
      - [理论学习篇](#理论学习篇)
        - [【关于 信息抽取】 那些的你不知道的事](#关于-信息抽取-那些的你不知道的事)
          - [【关于 实体关系联合抽取】 那些的你不知道的事](#关于-实体关系联合抽取-那些的你不知道的事)
          - [【关于 命名实体识别】那些你不知道的事](#关于-命名实体识别那些你不知道的事)
          - [【关于 关系抽取】那些你不知道的事](#关于-关系抽取那些你不知道的事)
          - [【关于 文档级别关系抽取】那些你不知道的事](#关于-文档级别关系抽取那些你不知道的事)
          - [【关于 事件抽取】那些你不知道的事](#关于-事件抽取那些你不知道的事)
          - [【关于 关键词提取】 那些你不知道的事](#关于-关键词提取-那些你不知道的事)
          - [【关于 新词发现】 那些你不知道的事](#关于-新词发现-那些你不知道的事)
  - [参考资料](#参考资料)

## 介绍

### NLP 学习篇

#### 理论学习篇

##### [【关于 信息抽取】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/)

###### [【关于 实体关系联合抽取】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/ERE_study/)

- [【关于 PURE】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/ERE_study/PURE/) 【强烈推荐】
  - 论文：A Frustratingly Easy Approach for Joint Entity and Relation Extraction
  - 阅读理由：反直觉！陈丹琦用pipeline方式刷新关系抽取SOTA 
  - 方法：建立两个 encoders，并独立训练:
    - encoder 1：entity model
      - 方法：建立在 span-level representations 上
    - encoder 2：relation model：只依赖于实体模型作为输入特征
      - 方法：builds on contextual representations specific to a given pair of span
  - 优点：
    - 很简单，但我们发现这种流水线方法非常简单有效；
    - 使用同样的预先训练的编码器，我们的模型在三个标准基准（ACE04，ACE05，SciERC）上优于所有以前的联合模型；
  - 问题讨论：
    - Q1、关系抽取最care什么？
      - 解答：引入实体类别信息会让你的关系模型有提升
    - Q2、共享编码 VS 独立编码 哪家强？
      -  解答：由于两个任务各自是不同的输入形式，并且需要不同的特征去进行实体和关系预测，也就是说：使用单独的编码器确实可以学习更好的特定任务特征。
    - Q3：误差传播不可避免？还是不存在？
      - 解答：并不认为误差传播问题不存在或无法解决，而需要探索更好的解决方案来解决此问题
    - Q4：Effect of Cross-sentence Context
      - 解答：使用跨句上下文可以明显改善实体和关系
- [【关于 PRGC】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/ERE_study/PRGC)
  - 论文：PRGC: Potential Relation and Global Correspondence Based JointRelational Triple Extraction
  - 来源：ACL 2021
  - 论文地址：https://arxiv.org/pdf/2106.09895
  - 开源代码：https://github.com/hy-struggle/PRGC
  - 动机：从非结构化文本中联合提取实体和关系是信息提取中的一项关键任务。最近的方法取得了可观的性能，但仍然存在一些固有的局限性：
    - 关系预测的冗余：TPLinker 为了避免曝光偏差，它利用了相当复杂的解码器，导致了稀疏的标签，关系冗余；
    - span-based 的提取泛化性差和效率低下;
  - 论文方法：
    - 从新颖的角度将该任务分解为三个子任务：
      - Relation  Judgement；
      - Entity  Extraction；
      - Subject-object Alignment；
    - 然后提出了一个基于 Potential Relation and Global Correspondence (PRGC) 的联合关系三重提取框架：
      - **Potential Relation Prediction**：给定一个句子，模型先预测一个可能存在关系的子集，以及得到一个全局矩阵；
      - **Relation-Specific Sequence Tagging**：然后执行序列标注，标注存在的主体客体，以处理 subjects  and  object 之间的重叠问题；
      - **Global Correspondence**：枚举所有实体对，由全局矩阵裁剪；
    - 实验结果：PRGC 以更高的效率在公共基准测试中实现了最先进的性能，并在重叠三元组的复杂场景中提供了一致的性能增益
- [【关于 实体关系联合抽取】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/ERE_study/实体关系联合抽取总结.md)
  1. pipeline  方法
     1. 思路：先命名实体识别（ NER） , 在 关系抽取（RE）
     2. 问题：
        1. 忽略两任务间的相关性
        2. 误差传递。NER 的误差会影响 RE 的性能
  2. end2end 方法
     1. 解决问题：实体识别、关系分类
     2. 思路：
        1. 实体识别
           1. BIOES 方法：提升召回？和文中出现的关系相关的实体召回
           2. 嵌套实体识别方法：解决实体之间有嵌套关系问题
           3. 头尾指针方法：和关系分类强相关？和关系相关的实体召回
           4. copyre方法
        2. 关系分类：
           1. 思路：判断 【实体识别】步骤所抽取出的实体对在句子中的关系
           2. 方法：
              1. 方法1：1. 先预测头实体，2. 再预测关系、尾实体
              2. 方法2：1. 根据预测的头、尾实体预测关系
              3. 方法3：1. 先找关系，再找实体 copyre
           3. 需要解决的问题：
              1. 关系重叠
              2. 关系间的交互
  3. 论文介绍
     1. 【paper 1】Joint entity recognition and relation extraction as a multi-head selection problem
     2. 【paper 2】Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy[ACL2017]
     3. 【paper 3】GraphRel:Modeling Text as Relational Graphs for Joint Entity and Relation Extraction [ACL2019]
     4. 【paper 4】CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning [AAAI2020]
     5. 【paper 5】Span-based Joint Entity and Relation Extraction with Transformer Pre-training [ECAI 2020]
     6. 【paper 6】A Novel Cascade Binary Tagging Framework for Relational Triple Extraction[ACL2020]
     7. 【paper 7】END-TO-END NAMED ENTITY RECOGNITION AND RELATION EXTRACTION USING PRE-TRAINED LANGUAGE MODELS

- [Incremental Joint Extraction of Entity Mentions and Relations](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/ERE_study/T2014_joint_extraction/)
- [【关于 Joint NER】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/ERE_study/JointER/)
  - 论文名称：Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy
- [【关于 GraphRel】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/ERE_study/ACL2019_GraphRel/)
  - 论文名称：论文名称：GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction
  - 动机
    - 想要自动提取特征的联合模型
      - 通过堆叠Bi-LSTM语句编码器和GCN (Kipf和Welling, 2017)依赖树编码器来自动学习特征
      - 用以考虑线性和依赖结构
        - 类似于Miwa和Bansal(2016)（一样是堆叠的）
          - 方法
            - 每个句子使用Bi-LSTM进行自动特征学习
            - 提取的隐藏特征由连续实体标记器和最短依赖路径关系分类器共享
          - 问题
            - 然而，在为联合实体识别和关系提取引入共享参数时：
              - 它们仍然必须将标记者预测的实体提及通过管道连接起来
              - 形成关系分类器的提及对
      - 考虑重叠关系
      - 如何考虑关系之间的相互作用
        - 2nd-phase relation-weighted GCN
        - 重叠关系(常见）
          - 情况
            - 两个三元组的实体对重合
            - 两个三元组都有某个实体mention
          - 推断
            - 困难（对联合模型尤其困难，因为连实体都还不知道）
    - 方法：
      - 学习特征
        - 通过堆叠Bi-LSTM语句编码器和GCN (Kipf和Welling, 2017)依赖树编码器来自动学习特征
      - 第一阶段的预测
        - GraphRel标记实体提及词，预测连接提及词的关系三元组
        - 用关系权重的边建立一个新的全连接图（中间图）
        - 指导：关系损失和实体损失
      - 第二阶段的GCN
        - 通过对这个中间图的操作
        - 考虑实体之间的交互作用和可能重叠的关系
        - 对每条边进行最终分类
        - 在第二阶段，基于第一阶段预测的关系，我们为每个关系构建完整的关系图，并在每个图上应用GCN来整合每个关系的信息，进一步考虑实体与关系之间的相互作用。
- [【关于 关系抽取 之 HBT】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/ERE_study/T20ACL_HBT_su/)
  - 论文名称：A Novel Hierarchical Binary Tagging Framework for Relational Triple Extraction
  - 论文会议：ACL 2020
  - keras4bert 版本：https://github.com/bojone/lic2020_baselines 【苏神 Lic2020 baseline】
  - pytorch 版本：https://github.com/powerycy/Lic2020- 【逸神 pytorch 复现版本】
  - 动机：
    - pipeline approach
      - 思路
        - 实体抽取：利用一个命名实体识别模型 识别句子中的所有实体；
        - 关系分类：利用 一个关系分类模型 对每个实体对执行关系分类。 【这一步其实可以理解为文本分类任务，但是和文本分类任务的区别在于，关系分类不仅需要学习句子信息，还要知道 实体对在 句子中 位置信息】 
      - 问题
        - 误差传递问题：由于 该方法将 实体-关系联合抽取任务 分成 实体抽取+关系分类 两个任务处理，所以 实体抽取任务的错误无法在后期阶段进行纠正，因此这种方法容易遭受错误传播问题；
    - feature-based models and neural network-based models 
      - 思路
        - 通过用学习表示替换人工构建的特征，基于神经网络的模型在三重提取任务中取得了相当大的成功。
      - 问题
        - 大多数现有方法无法正确处理句子包含多个相互重叠的关系三元组的情况。
    - 基于Seq2Seq模型  and GCN
      - 思路：
        - 提出了具有复制机制以提取三元组的序列到序列（Seq2Seq）模型。 他们基于Seq2Seq模型，进一步研究了提取顺序的影响，并通过强化学习获得了很大的改进。 
      - 问题：
        - 过多 negative examples：在所有提取的实体对中，很多都不形成有效关系，从而产生了太多的negative examples；
        - EPO(Entity Pair Overlap) 问题：当同一实体参与多个关系时，分类器可能会感到困惑。 没有足够的训练样例的情况下，分类器就很难准确指出实体参与的关系；
  - 方式：实现了一个不受重叠三元组问题困扰的HBT标注框架(Hierarchical Binary Tagging Framework)来解决RTE任务；论文并不是学习关系分类器f（s，o）→r，而是学习关系特定的标记器fr（s）→o；每个标记器都可以识别特定关系下给定 subject 的可能 object(s)。 或不返回任何 object，表示给定的主题和关系没有 triple。
  - 核心思想：把关系(Relation)建模为将头实体(Subject)映射到尾实体(Object)的函数，而不是将其视为实体对上的标签。
  - 思路：
    - 首先，我们确定句子中所有可能的 subjects； 
    - 然后针对每个subjects，我们应用特定于关系的标记器来同时识别所有可能的 relations 和相应的 objects。
  - 结构：
    - BERT Encoder层：使用 Bert 做 Encoder，其实就是 用 Bert 做 Embedding 层使用。
    - Hierarchical Decoder层
      - Subject tagger 层：用于 提取 Subject;
      - Relation-Specific Object Taggers 层：由一系列relation-specific object taggers（之所以这里是多个taggers是因为有多个可能的relation）；

###### [【关于 命名实体识别】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/)

- [【关于 命名实体识别 之 W2NER 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/AAAI2022_W2NER)
  - 论文：Unified Named Entity Recognition as Word-Word Relation Classification
  - 会议：AAAI 2022
  - 论文地址：https://arxiv.org/pdf/2112.10070.pdf
  - 代码：https://github.com/ljynlp/w2ner
  - 动机：
    - 如何 构建解决非嵌套，嵌套，不连续实体的统一框架？
      - span-based 只关注边界识别
      - Seq2Seq 可能会受到暴露偏差的影响
    - 论文方法：
      - 通过将统一的 NER 建模为 word-word relation classification（W2NER）
      - 该架构通过使用 Next-Neighboring-Word (NNW) 和 Tail-Head-Word-* (THW-*) 关系有效地建模实体词之间的相邻关系，解决了统一 NER 的内核瓶颈。
      - 基于 W2NER 方案，我们开发了一个神经框架，其中统一的 NER 被建模为单词对的 2D 网格。
      - 然后，我们提出了多粒度 2D 卷积，以更好地细化网格表示。
      - 最后，使用一个共同预测器来充分推理词-词关系。
    - 实验：在 14 个广泛使用的基准数据集上进行了广泛的实验，用于非嵌套，嵌套，不连续实体的 NER（8 个英文和 6 个中文数据集），其中我们的模型击败了所有当前表现最好的基线，推动了最先进的性能- 统一NER的mances
  
- [【关于 Few-Shot Named Entity Recognition】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/FewShotNER)
  - 论文名称：Few-Shot Named Entity Recognition: A Comprehensive Study
  - 将 few shot learning 应用于 NER 领域中需要面临的三个核心问题
    1. How to adapt meta-learning such as prototype-based methods for few-shot NER? （如何将元学习方法作为 prototype-based 的方法应用到 few-shot NER 领域中？）
    2. How to leverage freely-available web data as noisy supervised pre-training data?（如何利用大量免费可用的网页数据构造出 noisy supervised 方法中的预训练数据？）
    3. How to leverage unlabeled in-domain sentences in a semi-supervised manner?（如何在半监督的范式中利用好 in-domain 的无标注数据？）
- [【关于 AutoNER】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/EMNLP2018_AutoNER)
  - 论文名称：Learning Named Entity Tagger using Domain-Specific Dictionary
  - 会议： EMNLP2018
  - 论文地址：https://arxiv.org/abs/1809.03599
  - 项目地址：https://github.com/shangjingbo1226/AutoNER
  - 论文动机：
    - 基于机器学习的命名实体识别方法：需要 手工标注特征；
    - 基于深度学习的命名实体识别方法：需要大量标准数据；
    - 远程监督（结合外部词典）标注数据：生成的嘈杂标签对学习
  - 论文方法：提出了两种神经模型，以适应字典中嘈杂的远程监督：
    - 首先，在传统的序列标记框架下，我们提出了一个修改后的模糊 CRF 层来处理具有多个可能标签的标记。
    - 在确定远程监督中嘈杂标签的性质后，我们超越了传统框架，提出了一种新颖、更有效的神经模型 AutoNER，该模型具有新的 Tie or Break 方案。
    - 讨论了如何改进远程监督以获得更好的 NER 性能。
  - 实验结果：在三个基准数据集上进行的大量实验表明，仅使用字典而不需要额外的人力时，AutoNER 实现了最佳性能，并通过最先进的监督基准提供了具有竞争力的结果。
- [【关于 Continual Learning for NER】那些你不知道的事](#关于-continual-learning-for-ner那些你不知道的事)
  - 会议：AAAI2021
  - 论文：Continual Learning for Named Entity Recognition
  - 论文下载地址：https://assets.amazon.science/65/61/ecffa8df45ad818c3f69fb1cf72b/continual-learning-for-named-entity-recognition.pdf
  - 动机：业务扩展，需要新增 实体类型（eg:像 Sirior Alexa 这样的语音助手不断地为其功能引入新的意图，因此**新的实体类型经常被添加到他们的插槽填充模型**中）
  - 方法：研究 将 知识蒸馏（KD） 应用于 NER 的 CL 问题，通过 将 “teacher”模型的预测合并到“student”模型的目标函数中，该模型正在接受训练以执行类似但略有修改的任务。 通过学习输出概率分布，而不仅仅是标签，使得学生表现得与教师相似。
  - 论文贡献：
    - (i) 我们展示了如何使 CL 技术适应 NLU 域，以逐步学习 NER 的新实体类型； 
    - (ii) 我们在两个 EnglishNER 数据集上的结果表明，我们的 CL 方法使模型能够不断学习新的实体类型，而不会失去识别先前获得的类型的能力； 
    - (iii) 我们表明我们的半监督策​​略实现了与全监督设置相当的结果。
- [【关于 NER数据存在漏标问题】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/UnlabeledEntityProblem/)
  - 一、摘要
  - 二、为什么 数据会存在漏标？
  - 三、什么是 带噪学习
  - 四、NER 数据漏标问题所带来后果？
  - 五、NER 性能下降 **原因**是什么？
  - 六、论文所提出的方法是什么？
  - 七、数据漏标，会导致NER指标下降有多严重？
  - 八、对「未标注实体问题」的解决方案有哪些？
  - 九、如何降噪：改变标注框架+负采样？
    - 9.1 第一步：改变标注框架
    - 9.2 第二步：负采样
  - 十、负样本采样，效果如何？
- [【关于 LEX-BERT】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/ICLR2021_LEX_BERT/)【强烈推荐】
  - 推荐理由：在 query 中 引入 标签信息的方法，秒杀 Flat NER，登上 2021 年 Chinese NER SOTA。
  - 论文名称：《Lex-BERT: Enhancing BERT based NER with lexicons》
  - 动机：尽管它在NER任务中的表现令人印象深刻，但最近已经证明，添加词汇信息可以显著提高下游性能。然而，没有任何工作在不引入额外结构的情况下将单词信息纳入BERT。在我们的工作中，我们提出了词法BERT（lex-bert），这是一种在基于BERT的NER模型中更方便的词汇借用方法
  - 方法：
    - LEX-BERT V1：Lex BERT的第一个版本通过在单词的左右两侧插入特殊标记来识别句子中单词的 span。特殊标记不仅可以标记单词的起始位置和结束位置，还可以为句子提供实体类型信息
    - LEX-BERT V2：对于在句子中加宽的单词，我们没有在句子中单词的周围插入起始和结束标记，而是在句子的末尾附加一个标记[x]。请注意，我们将标记的位置嵌入与单词的起始标记绑定
- [【关于 嵌套命名实体识别（Nested NER）】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/NestedNER/)
  - [【关于 Biaffine Ner 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/ACL2020_NERasDependencyParsing/)
    - 动机：NER 研究 关注于 扁平化NER，而忽略了 实体嵌套问题；
    - 方法： 在本文中，我们使用基于图的依存关系解析中的思想，以通过 biaffine model 为模型提供全局的输入视图。 biaffine model 对句子中的开始标记和结束标记对进行评分，我们使用该标记来探索所有跨度，以便该模型能够准确地预测命名实体。
    - 工作介绍：在这项工作中，我们将NER重新确定为开始和结束索引的任务，并为这些对定义的范围分配类别。我们的系统在多层BiLSTM之上使用biaffine模型，将分数分配给句子中所有可能的跨度。此后，我们不用构建依赖关系树，而是根据候选树的分数对它们进行排序，然后返回符合 Flat 或  Nested NER约束的排名最高的树 span；
    - 实验结果：我们根据三个嵌套的NER基准（ACE 2004，ACE 2005，GENIA）和五个扁平的NER语料库（CONLL 2002（荷兰语，西班牙语），CONLL 2003（英语，德语）和ONTONOTES）对系统进行了评估。结果表明，我们的系统在所有三个嵌套的NER语料库和所有五个平坦的NER语料库上均取得了SoTA结果，与以前的SoTA相比，实际收益高达2.2％的绝对百分比。
  - [【关于 Biaffine 代码解析】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/ACL2020_NERasDependencyParsing/code_pytorch.md)
    - 摘要
    - 一、数据处理模块
      - 1.1 原始数据格式
      - 1.2 数据预处理模块 data_pre()
        - 1.2.1 数据预处理 主 函数
        - 1.2.2  训练数据加载 load_data(file_path)
        - 1.2.3 数据编码 encoder(sentence, argument)
      - 1.3 数据转化为 MyDataset 对象
      - 1.4 构建 数据 迭代器
      - 1.5 最后数据构建格式
    - 二、模型构建 模块
      - 2.1 主题框架介绍
      - 2.2 embedding layer
      - 2.2 BiLSTM
      - 2.3 FFNN
      - 2.4 biaffine model
      - 2.5 冲突解决
      - 2.6 损失函数
    - 三、学习率衰减 模块
    - 四、loss 损失函数定义
      - 4.1 span_loss 损失函数定义
      - 4.2 focal_loss 损失函数定义
    - 四、模型训练
  - [【关于 命名实体识别 之 GlobalPointer 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/GlobalPointer)
    - 博客：【[GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)】
    - 代码：https://github.com/bojone/GlobalPointer
    - 动机：
      - 在做实体识别或者阅读理解时，一般是用两个模块分别识别实体的首和尾；存在问题：出现 训练和预测时的不一致问题
    - 论文方法：
      - **GlobalPointer是基于内积的token-pair识别模块，它可以用于NER场景，因为对于NER来说我们只需要把每一类实体的“(首, 尾)”这样的token-pair识别出来就行了。**
    - 结论：
      - 利用**全局归一化**的思路来进行命名实体识别（NER），可以无差别地识别嵌套实体和非嵌套实体，在非嵌套（Flat NER）的情形下它能取得媲美CRF的效果，而在嵌套（Nested NER）情形它也有不错的效果。还有，在理论上，GlobalPointer的设计思想就比CRF更合理；而在实践上，它训练的时候不需要像CRF那样递归计算分母，预测的时候也不需要动态规划，是完全并行的，理想情况下时间复杂度是 O(1)。
  - [【关于 命名实体识别 之 Efficient GlobalPointer 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/EfficientGlobalPointer)
    - 博客：【[Efficient GlobalPointer：少点参数，多点效果](https://kexue.fm/archives/8877)】
    - 代码：https://github.com/bojone/GlobalPointer
    - 动机：原GlobalPointer参数利用率不高
    - 解决方法：**分解为“抽取”和“分类”**两个步骤，**“抽取”就是抽取出为实体的片段，“分类”则是确定每个实体的类型**。

- [【关于 NER trick】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/NER_study/NERtrick.md)
- [【关于TENER】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/ACL2019/ACL2019_TENER/)
  - 论文名称：TENER: Adapting Transformer Encoder for Name Entity Recognition
  - 动机：
    - 1. Transformer 能够解决长距离依赖问题；
    - 2. Transformer 能够并行化；
    - 3. 然而，Transformer 在 NER 任务上面效果不好。
  - 方法：
    -  第一是经验发现。 引入：相对位置编码
    -  第二是经验发现。 香草变压器的注意力分布是缩放且平滑的。 但是对于NER，因为并非所有单词都需要参加，所以很少注意是合适的。 给定一个当前单词，只需几个上下文单词就足以判断其标签。 平稳的注意力可能包括一些嘈杂的信息。 因此，我们放弃了点生产注意力的比例因子，而使用了无比例且敏锐的注意力。
- [【关于DynamicArchitecture】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/DynamicArchitecture/)
  - 介绍：Dynamic Architecture范式通常需要设计相应结构以融入词汇信息。
  - 论文：
    - [【关于 LatticeLSTM 】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/DynamicArchitecture/1_ACL2018_LatticeLSTM/)
      - 想法：在 char-based 的 LSTM 中引入词汇信息
      - 做法：
        - 根据大量语料生成词典；
        - 若当前字符与前面的字符无法组成词典中词汇，则按 LSTM 的方法更新记忆状态；
        - 若当前字符与前面的字符组成词典中词汇，从最新词汇中提取信息，联合更新记忆状态；
      - 存在问题：
        - 计算性能低下，导致其**不能充分利用GPU进行并行化**。究其原因主要是每个字符之间的增加word cell（看作节点）数目不一致；
        - 信息损失：
          - 1）每个字符只能获取以它为结尾的词汇信息，对于其之前的词汇信息也没有持续记忆。如对于「大」，并无法获得‘inside’的「长江大桥」信息。
          - 2）由于RNN特性，采取BiLSTM时其前向和后向的词汇信息不能共享，导致 Lattice LSTM **无法有效处理词汇信息冲突问题**
        - 可迁移性差：只适配于LSTM，不具备向其他网络迁移的特性。
    - [【关于 LR-CNN 】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/DynamicArchitecture/2_IJCAI2019_LR_CNN/)
      - 动机
        - 词信息引入问题；
         - lattice LSTM 问题：
           - 基于 RNN 结构方法不能充分利用 GPU 并行计算资源；
             - 针对句子中字符计算；
             - 针对匹配词典中潜在词
           - 很难处理被合并到词典中的潜在单词之间的冲突：
             - 一个字符可能对应词典中多个潜在词，误导模型
       - 方法：
        - Lexicon-Based CNNs：采取CNN对字符特征进行编码，感受野大小为2提取bi-gram特征，堆叠多层获得multi-gram信息；同时采取注意力机制融入词汇信息（word embed）；
        - Refining Networks with Lexicon Rethinking：由于上述提到的词汇信息冲突问题，LR-CNN采取rethinking机制增加feedback layer来调整词汇信息的权值：具体地，将高层特征作为输入通过注意力模块调节每一层词汇特征分布，利用这种方式来利用高级语义来完善嵌入单词的权重并解决潜在单词之间的冲突。
    - [【关于 CGN 】那些你不知道的事 ](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/DynamicArchitecture/3_EMNLP2019_CGN/)
      - 动机
        - 中文命名实体识别中，词边界 问题；
        - 如何 引入 词边界信息：
          - pipeline：CWS -> NER 
            - 问题：误差传递
          - CWS 和 NER 联合学习
            - 问题：标注 CWS 数据
          - 利用 词典 自动构建
            - 优点：比 CWS 标注数据 更容易获得
            - 问题：
              - 第一个挑战是整合自我匹配的词汇词；
                - 举例：“北京机场” (Beijing Airport) and “机场” (Airport) are the self-matched words of the character “机” (airplane)
              - 第二个挑战是直接整合最接近的上下文词汇词；
                - 举例：by directly using the semantic knowledge of the nearest contextual words “离开” (leave), an “I-PER” tag can be predicted instead of an “I-ORG” tag, since “希尔顿” (Hilton Hotels) cannot be taken as the subject of the verb “离开” 
        - 论文思路：
          - character-based Collaborative Graph：
            - encoding layer：
              - 句子信息：
                - s1：将 char 表示为 embedding;
                - s2：利用 biLSTM 捕获 上下文信息
              - lexical words 信息：
                - s1：将 lexical word 表示为 embedding;
              - 合并 contextual representation 和 word embeddings
            - a graph layer：
              - Containing graph (C-graph):
                - 思路：字与字之间无连接，词与其inside的字之间有连接；
                - 目的：帮助 字符 捕获 self-matched lexical words 的边界和语义信息
              - Transition graph (T-graph):
                - 思路：相邻字符相连接，词与其前后字符连接；
                - 目的：帮助 字符 捕获 相邻 上下文 lexical 词 的 语义信息
              - Lattice graph (L-graph):
                - 思路：通相邻字符相连接，词与其开始结束字符相连；
                - 目的：融合 lexical knowledge
              - GAT:
                - 操作：针对三种图，使用Graph Attention Network(GAN)来进行编码。最终每个图的输出
                  - > 其中 $G_k$ 为第k个图的GAN表示，因为是基于字符级的序列标注，所以解码时只关注字符，因此从矩阵中取出前n行作为最终的图编码层的输出。
            - a fusion layer：
              - 目的：融合 三种 graphs 中不同 的 lexical 知识 
            - a decoding layer:
              - 操作：利用 CRF 解码
    - [【关于 LGN 】那些你不知道的事 ](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/DynamicArchitecture/4_EMNLP2019_LGN/)
      - 动机：
        - 在 char-base Chinese NER 中，同一个字符可能属于多个 lexicon word，存在 overlapping ambiguity 问题
          - 举例(如下图)
            - 字符[流] 可以 匹配词汇 [河流] 和 [流经] 两个词汇信息，但是 Lattice LSTM 只能利用 [河流]；
        - Lattice LSTM这种RNN结构仅仅依靠前一步的信息输入，而不是利用全局信息
          - 举例
            - 字符 [度]只能看到前序信息，不能充分利用 [印度河] 信息，从而造成标注冲突问题
        - Ma等人于2014年提出，想解决overlapping across strings的问题，需要引入「整个句子中的上下文」以及「来自高层的信息」；然而，现有的基于RNN的序列模型，不能让字符收到序列方向上 remain characters 的信息；
      - 方法：
        - Graph Construction and Aggregation
        - Graph Construction 
        - Local Aggregation
        - Global Aggregation
        - Recurrent-based Update Module
    - [【关于 FLAT】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/DynamicArchitecture/5_ACL2020_FLAT/)
      - 动机
        - 方法一：设计一个动态框架，能够兼容词汇输入；
          - 代表模型： 
            - Lattice LSTM：利用额外的单词单元编码可能的单词，并利用注意力机制融合每个位置的变量节点
            - LR-CNN：采用不同窗口大小的卷积核来编码 潜在词
          - 问题：
            - RNN 和 CNN 难以解决长距离依赖问题，它对于 NER 是有用的，例如： coreference（共指）
            - 无法充分利用 GPU 的并行计算能力
        - 方法二：将 Lattice 转化到图中并使用 GNN 进行编码：
          - 代表模型
            - Lexicon-based GN(LGN)
            - Collaborative GN(CGN)
          - 问题
            - 虽然顺序结构对于NER仍然很重要，并且 Graph 是一般的对应物，但它们之间的差距不可忽略;
            - 需要使用 LSTM 作为底层编码器，带有顺序感性偏置，使模型变得复杂。
      - 方法：将Lattice结构展平，将其从一个有向无环图展平为一个平面的Flat-Lattice Transformer结构，由多个span构成：每个字符的head和tail是相同的，每个词汇的head和tail是skipped的。
- [【关于 ACL 2019 中的NER】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/ACL2019/)
  - [named entity recognition using positive-unlabeled learning](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/NER_study/ACL2019/JointER/)
  - [【关于 GraphRel】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/ACL2019/ACL2019_NERusingPositive-unlabeledLearning/)
    - 论文名称：GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction
  - [Fine-Grained Entity Typing in Hyperbolic Space（在双曲空间中打字的细粒度实体）](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/ACL2019/Fine-GrainedEntityTypinginHyperbolicSpace/)
  - [【关于 TENER】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/ACL2019/ACL2019_TENER/)
    - 论文名称：TENER: Adapting Transformer Encoder for Name Entity Recognition
- [【关于 EMNLP 2019 中的NER】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/EMNLP2019/)
  - [CrossWeigh从不完善的注释中训练命名实体标注器](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/EMNLP2019/CrossWeigh从不完善的注释中训练命名实体标注器/)
  - [利用词汇知识通过协同图网络进行中文命名实体识别](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/EMNLP2019/利用词汇知识通过协同图网络进行中文命名实体识别/)
  - [一点注释对引导低资源命名实体识别器有很多好处](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NER_study/EMNLP2019/一点注释对引导低资源命名实体识别器有很多好处/)

###### [【关于 关系抽取】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/)

- [End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures【2016】](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/T2016_LSTM_Tree/)
- [【关于 ERNIE】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/NRE_paper_study/ERNIE/)
- [【关于 GraphRel】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/GraphRel/)
- [【关于 R_BERT】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/R_BERT)
- [【关于 Task 1：全监督学习】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/T1_FullySupervisedLearning/)
  - [Relation Classification via Convolutional Deep Neural Network](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/T1_FullySupervisedLearning/T1_Relation_Classification_via_CDNN/)
  - [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/T1_FullySupervisedLearning/T2_Attention-Based_BiLSTM_for_RC/)
  - [Relation Classification via Attention Model](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/information_extraction/NRE_paper_study/T1_FullySupervisedLearning/T3_RC_via_attention_model_new/)
- [【关于 Task 2：远程监督学习】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/T2_DistantSupervisedLearning/)
  - [Relation Classification via Convolutional Deep Neural Network](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/T2_DistantSupervisedLearning/T1_Piecewise_Convolutional_Neural_Networks/)
  - [NRE_with_Selective_Attention_over_Instances](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/NRE_paper_study/T2_DistantSupervisedLearning/T2_NRE_with_Selective_Attention_over_Instances/)

###### [【关于 文档级别关系抽取】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/Doc-level_Relation_Extraction/)

- [【关于 Double Graph Based Reasoning for Document-level Relation Extraction】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/Doc-level_Relation_Extraction/DoubleGraphBasedReasoningforDocumentlevelRelationExtraction/)
- [【关于 ATLOP】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/Doc-level_Relation_Extraction/ATLOP/)
  - 论文：Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling
  - 发表会议：AAAI
  - 论文地址：https://arxiv.org/abs/2010.11304
  - github：https://github.com/wzhouad/ATLOP
  - 论文动机：
    - 对于文档级RE，一个文档包含多个实体对，需要同时对它们之间的关系进行分类 【语句级RE只包含一对实体对】
    - 对于文档级RE，一个实体对可以在与不同关系关联的文档中多次出现【对于句子级RE，每个实体对只能出现一个关系】 -> 多标签问题
    - 目前对于文档关系抽取主流的做法是采用基于graph的方法来做，但是很多基于BERT的工作也能够得到很好的结果，并且在基于graph的模型的实验部分，也都证明了BERT以及BERT-like预训练模型的巨大提升，以至于让人怀疑是否有必要引入GNN？作者发现如果只用BERT的话，那么对于不同的entity pair，entity的rep都是一样的，这是一个很大的问题，那是否能够不引入graph的方式来解决这个问题呢？
  - 论文方法：
    - localized context pooling
      - 解决问题：解决了 using the same entity embedding for allentity pairs 问题
      - 方法：使用与当前实体对相关的额外上下文来增强 entity embedding。不是从头开始训练一个new context attention layer ，而是直接将预先训练好的语言模型中的注意头转移到实体级的注意上
    - adaptive thresholding
      - 解决问题：问题 1 的 多实体对问题 和 问题 2 实体对存在多种关系问题
      - 方法：替换为先前学习中用于多标签分类的全局阈值，该阈值为**可学习的依赖实体的阈值**。
- [【关于 自适应Focal Loss和知识蒸馏的文档级关系抽取】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/Doc-level_Relation_Extraction/AdaptiveFocalLossAndKnowledgeDistillation/)
  - 论文：Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation
  - 发表会议：ACL 2022
  - 论文地址：https://arxiv.org/abs/2203.10900
  - github：https://github.com/tonytan48/KD-DocRE
  - 论文动机：
    - **大部分文档级别的实体关系横跨多个句子**，关系抽取模型需要捕捉更长的上下文信息；
    - **同一文档中包含大量实体，文档级别关系抽取需要同时抽取所有实体间的关系**，其复杂度与文档中的实体数成平方关系，分类过程中存在大量的负样本；
    - **文档级别关系抽取的样本类别属于长尾分布**。以清华大学发布的 DocRED 数据集为例，频率前十的关系占到了所有关系的 60%，而剩下的 86 种关系只占全部关系三元组的 40%；
    - 由于文档级别的数据标注任务较难，现有的数据集中人工标注的训练数据十分有限。大量的训练数据为远程监督[2]的训练数据，而**远程监督的数据中存在大量的噪音，限制模型的训练**。
  - 论文方法：
    - 提出了一个包含三个新组件的 DocRE 半监督框架。
      - 首先，我们使用轴向注意力模块来学习实体对之间的相互依赖关系，从而提高了两跳关系的性能。
      - 其次，我们提出了一种自适应焦点损失来解决 DocRE 的类不平衡问题。
      - 最后，我们使用知识蒸馏来克服人工注释数据和远程监督数据之间的差异。
  - 实验结果：对两个 DocRE 数据集进行了实验。我们的模型始终优于强大的基线，其性能在 DocRED 排行榜上超过了之前的 SOTA 1.36 F1 和 1.46 Ign_F1 分数。
- [【关于 RelationPrompt】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/Doc-level_Relation_Extraction/RelationPrompt/)
  - 论文：RelationPrompt: Leveraging Prompts to Generate Synthetic Data for Zero-Shot Relation Triplet Extraction
  - 发表会议：ACL 2022
  - 论文地址：https://arxiv.org/abs/2203.09101
  - github：https://github.com/declare-lab/RelationPrompt 【未更新】
  - 动机：尽管关系提取在构建和表示知识方面很重要，但**很少有研究集中在推广到看不见的关系类型**。
  - 论文方法：
    - 介绍了零样本关系三元组提取（ZeroRTE）的任务设置，以鼓励对低资源关系提取方法的进一步研究。给定一个输入句子，每个提取的三元组由头部实体、关系标签和尾部实体组成，其中在训练阶段看不到关系标签。
    - 为了解决 ZeroRTE，建议通过提示语言模型生成结构化文本来合成关系示例。具体来说，我们**统一语言模型提示和结构化文本方法来设计结构化提示模板，用于在以关系标签提示（RelationPrompt）为条件时生成合成关系样本**。
    - 为了克服在句子中提取多个关系三元组的局限性，设计了一种新颖的三元组搜索解码方法。
  - 实验结果：在 FewRel 和 Wiki-ZSL 数据集上的实验显示了 RelationPrompt 对 ZeroRTE 任务和零样本关系分类的有效性。


###### [【关于 事件抽取】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/event_extraction/)

- [【关于 MLBiNet】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/event_extraction/MLBiNet/)
  - 论文：MLBiNet: A Cross-Sentence Collective Event Detection Network
  - 会议： ACL2021
  - 论文下载地址：https://arxiv.org/pdf/2105.09458.pdf
  - 论文代码：https://github.com/zjunlp/DocED
  - 动机：跨句事件抽取旨在研究如何同时识别篇章内多个事件
  - 论文方法：论文将其重新表述为 **Seq2Seq 任务**，并提出了一个多层双向网络 (Multi-Layer Bidirectional Network，MLBiNet) 来 **融合跨句语义和关联事件信息，从而增强内各事件提及的判别**
  - 论文思路： 在解码事件标签向量序列时
    - 首先，为建模句子内部事件关系，我们提出双向解码器用于同时捕捉前向和后向事件依赖；
    - 然后，利用信息聚合器汇总句子语义和事件提及信息；
    - 最后，通过迭代多个由双向解码器和信息聚合器构造的单元，并在每一层传递邻近句子的汇总信息，最终感知到整个文档的语义和事件提及信息。

###### [【关于 关键词提取】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/keyword_ex_study/)

- [【关于 关键词提取】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/keyword_ex_study/)
  - 一、TF-IDF关键词提取算法
    - 1.1 理论基础
    - 1.2 计算公式
      - 1.2.1 词频 （Term Frequency，TF）
      - 1.2.2 逆文本频率(Inverse Document Frequency，IDF)
      - 1.2.3 TF-IDF
    - 1.3 应用
    - 1.4 实战篇
      - 1.4.1 TF-IDF算法 手撸版
      - 1.4.2 TF-IDF算法 Sklearn 版
      - 1.4.3 TF-IDF算法 jieba 版
  - 二、PageRank算法【1】
    - 2.1 理论学习
  - 三、TextRank算法【2】
    - 3.1 理论学习
    - 3.2 实战篇
      - 3.2.1 基于Textrank4zh的TextRank算法版
      - 3.2.2 基于jieba的TextRank算法实现
      - 3.2.3 基于SnowNLP的TextRank算法实现
- [【关于 KeyBERT 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/keyword_ex_study/KeyBert/)
  - 论文：Sharma, P., & Li, Y. (2019). Self-Supervised Contextual Keyword and Keyphrase Retrieval with Self-Labelling.
  - 论文地址：https://www.preprints.org/manuscript/201908.0073/download/final_file
  - 论文代码：https://github.com/MaartenGr/KeyBERT
  - 一、摘要
  - 二、动机
  - 三、论文方法
  - 四、实践
    - 4.1 安装
    - 4.2 KeyBERT 调用
    - 4.3 语料预处理
    - 4.4 利用 KeyBert 进行关键词提取
- [【关于 One2Set 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/keyword_ex_study/kg_one2set/)
  - 论文名称：One2Set: Generating Diverse Keyphrases as a Set
  - 论文：https://aclanthology.org/2021.acl-long.354/
  - 代码：https://github.com/jiacheng-ye/kg_one2set
  - 会议：ACL2021

###### [【关于 新词发现】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/word_discovery/)

- [【关于 新词发现】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/word_discovery/)
- [【关于 AutoPhrase】那些你不知道的事](https://github.com/km1994/nlp_paper_study_information_extraction/tree/master/information_extraction/word_discovery/AutoPhrase/)
  - 论文：AutoPhrase: Automated Phrase Mining from Massive Text Corpora
  - 会议：IEEE
  - 论文地址：https://arxiv.org/abs/1702.04457
  - 源码 Python 版本：https://github.com/luozhouyang/AutoPhraseX
  - 什么是 Phrase Mining？
    - 答：Phrase Mining 作为文本分析的基本任务之一，旨在从文本语料库中提取高质量的短语。
  - hrase Mining 有何用途？
    - 短语挖掘在各种任务中都很重要，例如信息提取/检索、分类法构建和主题建模。
  - Phrase Mining 现状？
    - 大多数现有方法依赖于复杂的、训练有素的语言分析器，因此在没有额外但昂贵的适应的情况下，可能在新领域和流派的文本语料库上表现不佳。虽然也有一些数据驱动的方法来从大量特定领域的文本中提取短语。
  - Phrase Mining 存在问题？
    1. 非 自动化
    2. 需要人类专家来设计规则或标记短语
    3. 依赖于 语言分析器
    4. 应用到新的领域效果不好
  - 论文方法 ？
    1. Robust Positive-Only Distant Training：使用wiki和freebase作为显眼数据，根据知识库中的相关数据构建Positive Phrases,根据领域内的文本生成Negative Phrases，构建分类器后根据预测的结果减少负标签带来的噪音问题。
    2. POS-Guided Phrasal Segmentation：使用POS词性标注的结果，引导短语分词，利用POS的浅层句法分析的结果优化Phrase boundaries。
  - 论文效果 ？
    - AutoPhrase可以支持任何语言，只要该语言中有通用知识库。与当下最先进的方法比较，新方法在跨不同领域和语言的5个实际数据集上的有效性有了显著提高。


## 参考资料

1. [【ACL2020放榜!】事件抽取、关系抽取、NER、Few-Shot 相关论文整理](https://www.pianshen.com/article/14251297031/)
2. [第58届国际计算语言学协会会议（ACL 2020）有哪些值得关注的论文？](https://www.zhihu.com/question/385259014)
