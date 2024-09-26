# AmsTextembeddingModel
Tencent AMS General Text Embeddings Model Factory, Brilliant Ads Retrieval Model
Abstract
对于不同类型的内容理解任务，此前的业界方案普遍分为在线召回和在线分类两种：
1）召回方案面向没有明显层级结构的大规模标签分配问题，通用结构就是相似度检索倒排后进行判定留存。
2）分类方案面向标签具有层级结构的标签分类问题，业界方案普遍基于预训练好的bert进行逐级的分类问题，具体方案包括模糊递进，模型交叉共享等方案。
大模型时代，内容理解任务从理解框架，理解逻辑到理解效果都有了翻天覆地的变化。
轻链路的实时流程通过在线的llm理解+bert召回做内容标签实时召回。
重链路的离线流程则通过多次大模型环节精准召回内容标签后缓存到线上的查询库，通过扩大缓存+相似映射的方法得打内容标签的实时召回。

For different types of content understanding tasks, previous industry solutions are generally divided into two categories: online recall and online classification:
Recall solutions are aimed at large-scale label assignment problems without obvious hierarchical structures. The general structure is to perform similarity retrieval and then determine retention based on the inverted index.
Classification solutions are aimed at label classification problems with hierarchical structures. Industry solutions generally use pre-trained BERT for step-by-step classification problems. Specific solutions include fuzzy progression, model cross-sharing, and other methods.
In the era of large language models, content understanding tasks have undergone earth-shaking changes in terms of understanding frameworks, understanding logic, and understanding effects.
The real-time process of light chains performs real-time recall of content labels through online LLM understanding + BERT recall.
The offline process of heavy chains caches the accurately recalled content labels through multiple large model stages into the online query library, and achieves real-time recall of content labels through the method of expanding cache and similarity mapping.
https://doc.weixin.qq.com/flowchart-addon
无论哪一种方案，一个优质的bert都可以帮助理解结果和理解效率事半功倍，马到成功。

No matter which plan, a high-quality BERT can help understand the results and double the efficiency of understanding, leading to immediate success.
Introduction
基于大模型的内容理解流程在ams广告场景取得了显著效果，各场景都在llm理解文本信息生成summary后，基于通用文本表示向量召回这一流程下，显著提高了对应场景内容理解标签的关联率和准确率。

The content understanding process based on large models has achieved significant results in the AMS advertising scenario. In all scenarios, after the LLM understands the text information and generates a summary, the process of recalling based on a common text representation vector significantly improves the association rate and accuracy of the corresponding scenario's content understanding labels.
如果说大模型的理解能力决定了内容理解任务的天花板，召回模型的准确率则决定了内容理解的下限，想输出标准结果给下游任务使用，embedding的质量至关重要。
为了提高视频号场景内容理解任务的稳定性，帮助内容理解标签在下有任务中起到更大的效果，起到更好的作用，我们从数据，模型结构，训练策略三个角度进行优化和实验验证：
1.数据上我们通过规则+策略召回大量搜索query-点击item数据，llm多轮投票的方法获取可靠的正/富样本对，扩充海量的视频号场景数据，学习专有生态下的文章风格和文本知识，大幅度增强训练数据。
2.模型结果上，我们对基于正样本的infoNCE训练框架进行了结构优化，扩充了负样本的学习方式，通过在负采样batch重拼接dummy pair并改进了loss中的temperature规则，显著扩充了模型对生产样本的学习能力。
3.训练策略上，我们通过对不同任务添加instrcut前缀，区分不同任务的召回逻辑，让模型自适应的学到不同任务的差距，从而帮助模型在多任务训练互相兼容，彼此解耦。
4.此外，我们基于LLM Cocktail探索了在时间紧急，资源不足的情况下，对历史模型基于融合方式快速得到效果提升的方法。
最终在人工标注数据集中的各个字段都有了全面的效果提升，其中类目准确率从52%提升到了77%。

If the understanding ability of large models determines the ceiling of content understanding tasks, then the accuracy of the recall model determines the lower limit of content understanding. To output standard results for downstream tasks, the quality of embeddings is crucial.
To improve the stability of the content understanding task in the video number scenario and help the content understanding labels play a greater role and function better in downstream tasks, we optimize and experimentally verify from three perspectives: data, model structure, and training strategy:
In terms of data, we recall a large amount of search query-click item data through rules and strategies, and obtain reliable positive/rich sample pairs through the multi-round voting method of LLM, expanding the massive video number scenario data, learning the article style and text knowledge under the proprietary ecosystem, and greatly enhancing the training data.
In terms of model structure, we have optimized the infoNCE training framework based on positive samples, expanded the learning method of negative samples, significantly expanded the model's learning ability for production samples by re-splicing dummy pairs in the negative sampling batch and improving the temperature rule in the loss.
In terms of training strategy, we add the instrcut prefix to different tasks to distinguish the recall logic of different tasks, allowing the model to adaptively learn the gap between different tasks, thereby helping the model to be compatible with each other in multi-task training and decouple from each other.
In addition, based on LLM Cocktail, we explored the method of quickly improving the effect based on the fusion method for historical models in the case of urgent time and insufficient resources.
Ultimately, comprehensive improvements have been made in various fields in the manually annotated dataset, among which the category accuracy has increased from 52% to 77%.
Pre Work
内容理解任务的第一个通用文本向量模型基于m3e-large在文章-概念，搜索-点击，小程序-类目等通用文本对上基于batch负采样作增量训练迭代而得。
搜索场景的通用文本向量模型则放弃了各场景的通用文本对，通过商品中台积累的商品描述文本和对应的类目标签做增量生成，通过在loss中引入angular margin差异化训练样本提高了预训练的收敛速度和embedding质量。通过reshape采样batch扩大正样本的感受视野，提高了小样本下场景迁移的学习效率。

The first universal text vector model for content understanding tasks is based on m3e-large and is obtained through incremental training iterations on general text pairs such as article-concept, search-click, mini-program-category, etc., using batch negative sampling.
The universal text vector model for search scenarios gives up the general text pairs of each scenario and generates incrementally through the product description text accumulated by the product middle platform and the corresponding category tags. By introducing angular margin into the loss to differentiate training samples, the convergence speed of pre-training and the quality of embeddings are improved. By reshaping the sampling batch to expand the receptive field of positive samples, the learning efficiency of scene transfer under small samples is improved.
Our Work
negatives dummy inhenced
视频号场景的数据基于召回样本基于大模型进行相关性判定，从而得到独立的正样本对和负样本对。
而传统对比学习主要是基于正样本训练或者将负样本对扩充拼接到原本的softmax的正方形loss矩阵右侧。
这两种方法负样本都不会学到自己的负相关性，最多只能在分母处传导部分梯度信息。为了解决这个问题，我们将负样本对的的第一列文本作为dummy pos加入训练，人为制造负样本对的正样本关系，从而学习负样本自己负相关性。

In the shipinhao scenario, data is based on recall samples and undergoes relevance determination through a large model, thereby obtaining independent positive sample pairs and negative sample pairs.
Traditional contrastive learning is mainly based on positive sample training or expanding and splicing negative sample pairs to the right side of the original softmax square loss matrix.
These two methods do not allow negative samples to learn their own negative correlation; at most, they can only transmit some gradient information at the denominator. To solve this problem, we add the first column of text of the negative sample pair as a dummy pos to the training, artificially creating a positive sample relationship for the negative sample pair, thereby learning the negative correlation of the negative sample itself.

reshape batch
MoCo3提到基于info-NCE的batch负采样训练效果会随着batch大小而显著变好，由于视频号场景的标题文本较长，attention cache占据内存较大，模型的训练内存被显著压缩，batch的大小原小于搜索场景。
受siglip启发，我们将batch负采样的矩阵进行分块儿梯度求导，通过减少相似度矩阵的大小，减少内存消耗的情况下扩大batch内正样本的感受野，从而近似达到更大的batch训练效果。

MoCo3 mentions that the training effect of batch negative sampling based on info-NCE will significantly improve with the increase of batch size. Due to the longer title text in the video number scenario and the larger memory occupied by the attention cache, the training memory of the model is significantly compressed, and the batch size is originally smaller than that of the search scenario.
Inspired by siglip, we perform block gradient derivation on the batch negative sampling matrix. By reducing the size of the similarity matrix and reducing memory consumption, we expand the receptive field of positive samples within the batch, thereby approximately achieving the training effect of a larger batch.

task identity instruct
对不同类型的样本加instruct前缀处理，让模型在学习的时候可以区分任务类型，针对性学习匹配模式。
the formulation of temperature for batch size and negatives train startup
由于负样本的加入，为了最大程度利用负样本信息，我们在原本的batch把负样本拼接到了尾部，避免负样本对自身的关系被彻底消融。
伴随而来的问题则是dummy pair本身的相似度太高，在weclip默认的最优温度0.05下，负样本的相似度<0.8即可以得到较低的loss。
为了解决这个问题，我们通过约束正样本和负样本的期望相似度，基于batch推导了对应的温度关系公式，从而大大增加负样本对对训练的帮助。

