# Data Collection and Quality Challenges in Deep Learning: A Data-Centric AI Perspective
Steven Euijong Whang, Yuji Roh, Hwanjun Song, Jae-Gil Lee

现实世界中的许多数据集都很小、很脏、有偏见，甚至是有毒的。 

数据质量：
- 数据验证
- 清洗
- 整合

使用鲁棒模型训练技术来处理不完美的数据

对偏差和公平性的研究：

可在模型训练之前、期间或之后应用的公平性措施和不公平性缓解技术

https://arxiv.org/pdf/2112.06409

### 概述  Summary
第 2 节: 数据收集技术
- 数据获取
- 数据标注
- 改进现有数据和模型。 

在第 3 节: 数据验证、数据清理、数据消毒和数据整合的主要方法。 
- 数据验证——可视化，模式信息。 
- 数据清理——最近的技术更多是为了提高模型的准确性。 
- 数据清除——防御中毒攻击的不同功能。 
- 数据整合：challenging 由于多模态数据

在第 4 节：有噪声或缺失的标签会 $\rightarrow$ 测试数据的泛化效果不佳。 
- 针对噪声标签的现有研究:
    - （i）累积噪声
    - （ii）对训练数据的部分探索。 
- 混合技术（如 SELFIE）和半监督技术（如 DivideMix）可以在训练数据有噪声的情况下达到很高的准确率。 
- 半监督（如 Mix Match）和自监督（如 JigsawNet）技术正在积极开发，以利用丰富的无标记数据。 

在第 5 节: 公平性措施、不公平性缓解技术以及收敛与鲁棒性技术。
- 缓解可以在模型训练之前、期间或之后进行。
- 当训练数据可以修改时，预处理非常有用。
- 当可以修改训练算法时，可以使用内部处理。
- 当我们无法修改数据和模型训练时，可以使用后处理。
- 收敛性技术可分为公平-稳健技术、稳健-公平技术和等价合并技术。

设想展望：
- 数据清理和稳健训练： 目前，数据清理越来越受到机器学习的重视，但被认为不如鲁棒训练有效。 我们认为，这两种技术应继续融合，以获得最佳效果。 
- 数据验证和模型公平性： 最近的数据验证工作指出，人工智能伦理是验证工作的挑战之一。 我们认为，模型公平性最终将被并入数据验证过程。 
- 数据收集： 迄今为止，大多数机器学习文献都假定输入数据已经给出。 与此同时，为实现准确的机器学习而进行数据收集目前已成为数据管理领域一个活跃的研究方向。 我们相信这一趋势将继续扩大，因为数据收集还需要考虑公平性和鲁棒性。 
- 模型训练和测试： 改进模型训练和测试协议正在成为处理数据质量问题的另一种解决方案。 模型在数据样本上的结果为评估数据提供了有用的知识，有助于开发准确、稳健的推理管道。 我们认为，模型的学习动态为解释稳健性和公平性提供了新的视角。 
- 模型的公平性和稳健性： 在机器学习领域，值得信赖的人工智能正变得越来越重要，我们相信，包括公平性和稳健性在内的各方面问题必须一并解决，而不是逐一解决。 值得信赖的人工智能还有其他要素，包括隐私和可解释性，这些最终也应成为以数据为中心的人工智能的一部分。

### 笔记 Notes
- Hwanjun Song, Minseok Kim, and Jae-Gil Lee. SELFIE: refurbishing unclean samples for robust deep learning. In ICML, pages 59075915, 2019. 处理 bad data 的稳健方法。

- mainly supervised learning, 需要寻找非监督学习的例子加入综述。

- https://arxiv.org/pdf/1811.03402 A Survey on Data Collection for Machine Learning: a Big Data -- AI Integration Perspective 数据收集方面参考



# Data quality dimensions for fair AI （非重要文章）
Camilla Quaresmini, Giuseppe Primiero

AI systems are not intrinsically neutral and biases trickle in any type of technological tool. In particular when dealing with people, AI algorithms reflect technical errors originating with mislabeled data. As they feed wrong and discriminatory classifications, perpetuating structural racism and marginalization, these systems are not systematically guarded against bias. In this article we consider the problem of bias in AI systems from the point of view of Information Quality dimensions. We illustrate potential improvements of a bias mitigation tool in gender classification errors, referring to two typically difficult contexts: the classification of non-binary individuals and the classification of transgender individuals. The identification of data quality dimensions to implement in bias mitigation tool may help achieve more fairness. Hence, we propose to consider this issue in terms of completeness, consistency, timeliness and reliability, and offer some theoretical results.

人工智能系统中的偏见问题。 

非二元个体的分类和变性个体的分类。

https://arxiv.org/pdf/2305.06967

### 结论
重点：及时性 -> 开发更公平、更具包容性的分类工具的基础维度。 

```
尽管如 Buolamwini 和 Gebru [2018] 以及 Angwin 等人 [2016] 等重要著作所示，准确度指标至关重要，但人工智能系统中的不公平问题更为广泛，更具基础性。 这可以用数据质量来表示： 人工智能系统的局限性在于最大限度地提高准确性，即使系统在统计上变得准确，有些问题仍然无法解决。 技术问题已经变成了文化问题。 而且，由于分类工作总是文化的反映，标签集的完整性和标签的（受限）一致性具有认识论价值：构建人工智能需要我们了解社会，而社会反映了个人的本体。 错误标识首先是一种本体论错误[Keyes, 2018]。 我们提出，及时性对于性别认同的定义至关重要。 如果我们准备将性别视为一种随时间而变化的属性[Ruberg and Ruelos, 2020, p.6]，而且它也可能不是单一的，因为一个人可能会认同不止一个并非相互排斥的标签，那么就需要改变范式。 事实上，当前的分类方法深深植根于对性别的三个假设： 1. 二元论，性别只能分为两个相互排斥的类别；2.静态性，性别身份类别一旦被指定就不会改变；3.从身体特征推导，性别可以从视觉线索中正确推断。 如果要设计出更公平的性别分类和更具包容性的性别模型，就必须解决这些设计上的局限性。 在这一方向上的进一步工作包括：对所提出的模型进行实施和经验验证；提出进一步的理论结果，例如与个人和群体公平性的不可能性结果有关的理论结果；设计一个扩展功能，以计算错误标签随着时间的推移而变为正确标签的概率（即，在一个时间段内，错误标签变为正确标签的概率）。 例如，D'Asaro 和 Primiero [2021]、Primiero 和 D'Asaro [2022]、Termine 等人[2021]、D'Asaro 等人[2023]。
```

# From Data Quality to Model Quality: an Exploratory Study on Deep Learning （很好的实践方法论文章，可以引入新的因素进行分析）
Tianxing He, Shengcheng Yu, Ziyuan Wang, Jieqiong Li, Zhenyu Chen

本文主要考虑数据质量的四个方面：
- 数据集平衡
- 数据集大小
- 标签质量
- 数据集污染。 

我们在 MNIST 和 Cifar-10 上进行了实验，试图找出这四个方面对模型质量的影响。 实验结果表明，四个方面都对模型质量有决定性影响。这意味着这些方面数据质量的下降会降低模型的准确性。

https://arxiv.org/pdf/1906.11882

# Improving Data Quality through Deep Learning and Statistical Models
Wei Dai, Kenji Yoshigoe, William Parsley

传统的数据质量控制方法是基于用户经验或以前建立的业务规则，这不仅限制了性能，而且非常耗时，准确性也低于理想水平。 

利用深度学习，我们可以利用计算资源和先进技术来克服这些挑战，为用户提供更大的价值。

我们提供了一个基于深度学习和统计模型算法的创造性数据质量框架，用于识别数据质量：
- 如何识别离群数据
- 如何通过深度学习提高数据质量

https://arxiv.org/pdf/1810.07132

# A Theoretical Framework for AI-driven data quality monitoring in high-volume data environments（展望相关）
Nikhil Bangad, Vivekananda Jayaram, Manjunatha Sughaturu Krishnappa, Amey Ram Banarse, Darshan Mohan Bidkar, Akshay Nagpal, Vidyasagar Parlapalli

将异常检测、分类和预测分析结合在一起

用于实时、可扩展的数据质量管理。 关键组件包括智能数据摄取层、自适应预处理机制、上下文感知特征提取和基于人工智能的质量评估模块。 持续学习模式是我们框架的核心，可确保适应不断变化的数据模式和质量要求。 我们还探讨了可扩展性、隐私和集成到现有数据生态系统中的影响。 虽然没有提供实际结果，但它为未来的研究和实施奠定了坚实的理论基础，推动了数据质量管理，并鼓励在动态环境中探索人工智能驱动的解决方案。

https://arxiv.org/pdf/2410.08576


# book:
## Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Unveiling AI's Potential Through Tools, Techniques, and Applications
Pohsun Feng, Ziqian Bi, Yizhu Wen, Xuanhe Pan, Benji Peng, Ming Liu, Jiawei Xu, Keyu Chen, Junyu Liu, Caitlyn Heqi Yin, Sen Zhang, Jinlang Wang, Qian Niu, Ming Li, Tianyang Wang

This book serves as an introduction to deep learning and machine learning, focusing on their applications in big data analytics. It covers essential concepts, tools like ChatGPT and Claude, hardware recommendations, and practical guidance on setting up development environments using libraries like PyTorch and TensorFlow. Designed for beginners and advanced users alike, it provides step-by-step instructions, hands-on projects, and insights into AI's future, including AutoML and edge computing.

https://arxiv.org/pdf/2410.01268    

# In 2024:
1. Data Quality: From Backroom to Boardroom
For too long, data quality has been an afterthought, relegated to the technical IT department. This is changing fast. Businesses are recognizing that high-quality data is a strategic asset, as vital to success as financial resources or skilled personnel.  Here’s how this shift is manifesting:

C-suite and executive sponsorship: CEOs, CFOs, and other executives are actively driving data quality initiatives. They understand that inaccurate reports, misleading forecasts, and compliance issues caused by poor data can seriously hinder business growth and reputation.
Business KPIs tied to data quality: Companies are incorporating data quality metrics into their performance dashboards. This goes beyond just counting errors – it could involve tracking improvements in decision-making accuracy, the reduction of customer complaints related to data issues, or faster time to insights.
Data quality as a competitive differentiator:  In a world where many organizations have access to similar technology and information, those with the cleanest, most reliable data gain an operational and strategic edge. The ability to make quick, confident decisions based on high-quality data can be a major competitive advantage.
Cross-departmental collaboration: No single team owns data quality. Sales, marketing, finance, product development, and operations all have a stake in the problem (and its solution). This breakdown of silos leads to better communication and shared data quality goals.
This shift elevates data quality to a core business function, ensuring that data-driven insights are reliable, actionable, and aligned with the organization’s strategic goals.

In today’s data-driven world, it’s not just about the quantity of data your organization collects but also about its quality. Bad data leads to bad decisions, missed opportunities, regulatory headaches, and a whole lot of frustration. Fortunately, the field of data quality is evolving rapidly, and these trends are shaping the future:

2. Data Observability: Your Data's Early Warning System
Borrowing concepts from observability practices in software development (DevOps), data observability provides a comprehensive and proactive approach to monitoring the health of your data and data pipelines.  Here’s how it goes beyond traditional monitoring:

Beyond “Is it working?” Traditional monitoring typically tells you if something has broken. Data observability digs deeper, focusing on metrics like:
Freshness: How up-to-date is your data, and are updates arriving on schedule?
Distribution: Does the data’s distribution match expectations (e.g., no unexpected spikes or dips in key metrics)?
Volume: Is the expected amount of data flowing in and out of the system?
Completeness: Is the data complete, all key columns contain values?
Schema: Are there unexpected changes to data structure or field definitions?
Real-time alerts Data observability tools can trigger alerts when these metrics deviate from the norm, allowing teams to address anomalies before they snowball into major problems. Learn more about incidents and alerts in the DQOps platform. 
Root cause analysis: It’s not just about identifying that something is wrong, but why. Data observability allows you to track issues back through the data pipeline, pinpoint the source, and determine whether the problem lies in upstream systems, code errors, or configuration issues. Read more about root cause analysis in our blog. 
Preventing Downstream Impacts: Catching errors early means you can potentially prevent them from contaminating analytics dashboards, messing up customer records, or causing incorrect AI predictions.
Data Observability vs. Data Monitoring Think of data monitoring as checking the oil level in your car’s engine. Data observability is like having a full array of sensors and gauges monitoring your car’s performance in real time, alerting you of potential problems before they leave you stranded on the side of the road. You can read more about the differences between data observability and data monitoring in our blog post. 
Data observability is crucial for complex data environments, as it enables data teams to maintain trust in their data and react quickly to evolving data needs.

3. Everyone's in the Data Quality Game
Traditionally, data quality was seen as the exclusive responsibility of IT teams or dedicated data engineers.  This is changing rapidly towards a more democratized approach where business users play a proactive role. Here’s why this is happening:

Business users know their data best: The people who work with data daily (customer service reps, sales teams, analysts) often have the deepest understanding of how it’s used, the common errors they encounter, and the impact those errors have on their work.
Self-service tools lower the technical barrier: Modern data quality platforms often have intuitive, user-friendly interfaces. This means business users can participate in basic profiling, spot inconsistencies, suggest corrections, and collaborate on data improvement without needing extensive technical expertise.
Empowerment leads to ownership: When business users are directly involved in improving data quality, they develop a sense of ownership and accountability. They are more likely to maintain high standards and less likely to tolerate shortcuts that lead to inaccurate data.
Data literacy boost: Participation in data quality initiatives helps business users become more data-savvy. They improve their understanding of data structures, the importance of proper formatting, and common sources of error.
Faster feedback loops: Instead of waiting on data teams to identify and fix issues, business users can directly flag anomalies or potential problems. This leads to shorter error correction cycles and less downstream impact.
 Note: This doesn’t eliminate the need for specialized data quality teams. Rather, it creates a collaborative model where business users act as “eyes on the ground” while data professionals focus on complex tasks, tool development, and system-wide improvements.

4. AI and ML: Your Data Quality Superstars
Artificial intelligence and machine learning are transforming how we approach data quality. Far from replacing human expertise, AI and ML offer a powerful toolkit that enhances and scales traditional data quality processes. Here’s how:

Pattern recognition at scale: AI algorithms can effortlessly analyze vast amounts of data, identifying patterns and anomalies that would be difficult or tedious for humans to spot. This includes detecting duplicates, format inconsistencies, missing values, and unusual outliers.
Smarter anomaly detection: AI models can learn to distinguish between “normal” variations and genuine data quality issues. This reduces false positives and lets data teams focus on the most critical problems.
Automated data cleansing suggestions:  Beyond identifying errors, AI can recommend corrections or cleansing fixes. For example, suggesting standardized address formats or filling in missing values based on previous entries.
Root-cause analysis assistance: ML models can be trained to understand relationships between different data fields and events. This helps trace data quality issues back to their source, supporting the drive to fix the underlying problem, not just the symptom.
Adaptive learning: AI-powered data quality systems continuously learn and improve over time. As new data flows in and users provide feedback, the models become more accurate and efficient at identifying and resolving issues.
Important Note: The success of AI in data quality depends on:

High-quality training data: AI models learn from examples, so it’s important to feed them reliable and well-labeled data.
Human oversight and guidance: AI should be seen as a powerful tool, not a magic bullet. Data experts are still crucial for interpreting AI’s output, making final decisions, and refining models.
AI and ML in data quality aren’t about eliminating human involvement, but rather empowering teams to work smarter, faster, and with a greater focus on strategic tasks.

5. The Cloud Democratizes Data Quality
Cloud-based data quality platforms are breaking down barriers and making sophisticated tools and resources accessible to businesses of all sizes. Here’s how the cloud is changing the data quality landscape:

No upfront infrastructure investment: Traditionally, implementing enterprise-grade data quality tools required significant hardware purchases, software installation, and ongoing maintenance. Cloud solutions eliminate those hurdles, operating on a subscription model.
Scalability on demand:  Cloud platforms let you easily adjust your data quality resources as your needs evolve. If you have a sudden influx of data, need to launch a new data quality initiative, or onboard additional users, the cloud provides flexibility.
Faster innovation: Cloud providers constantly update and improve their data quality offerings. This means organizations benefit from the latest features and techniques without lengthy in-house development cycles.
Focus on core competencies: By outsourcing a degree of data quality infrastructure and tool management to the cloud, internal IT teams can focus on strategic projects and support, rather than low-level maintenance.
Lower barriers to entry for small and medium-sized businesses: Advanced data quality capabilities were once reserved for large enterprises with hefty budgets. Cloud solutions level the playing field, letting smaller companies access powerful tools that can improve their competitiveness.
Easier collaboration: Cloud-based platforms make it easier for distributed teams or partners to collaborate on data quality initiatives, sharing insights and working towards common goals.
 

Important Considerations:

Security and Compliance: When evaluating cloud-based data quality solutions, it’s critical to carefully assess the provider’s security measures and their ability to meet regulatory compliance standards relevant to your industry.
Integration: Consider how the cloud data quality platform will integrate with your existing data systems (data warehouses, CRM systems, etc.). Seamless integration is important to maintain efficient workflows.
 

The cloud is democratizing data quality, making it a practical reality for organizations that may have previously struggled to implement robust data quality practices.

6. Getting to the Root of the Problem
Data quality isn’t just about fixing errors as they pop up. For lasting improvements, it’s vital to understand why those errors occur in the first place. Here’s why a root-cause analysis approach is crucial:

Preventing recurrence: If you only focus on the surface-level error,  the same problem (or a similar one) is likely to keep happening. Root-cause analysis helps you identify and address the underlying issue so it doesn’t keep causing data quality headaches.
Systemic vs. human errors:  Root-cause analysis helps distinguish between errors caused by human mistakes and those stemming from systemic problems.  This guides appropriate solutions – is it additional training that’s needed, or a process change, or perhaps a faulty data entry system?
Examples of root causes:
Poorly designed input forms: If a data entry form doesn’t have clear field definitions or validation rules, it will inevitably lead to inconsistent, messy data.
Manual processes prone to typos: Reliance on spreadsheets or manual data copying introduces risks of human error.
Lack of synchronization between systems: When data exists in multiple systems without proper updates and reconciliation, data quality inevitably suffers.
Outdated or missing data standards: If the organization lacks clear guidelines on things like date formats or address structures, you’ll end up with incompatible data.
Methods for root-cause analysis: Techniques like the “5 Whys” (asking ‘why’ repeatedly to drill down to the cause) and fishbone diagrams (visually mapping potential causes) are useful tools.
Benefits of Root-Cause Analysis in Data Quality:

Long-term data quality improvement: By addressing root causes, you establish systems and processes that produce more reliable data in the future.
Reduced firefighting: Teams spend less time constantly fixing similar errors and more time on proactive data quality strategies.
Cost savings: Preventing errors upstream often yields significant cost savings compared to continuously cleaning up messy data.
Improved decision-making: When you can trust the data, you’re less likely to make choices based on faulty assumptions, mitigating business risks.
 

Root-cause analysis turns data quality into a proactive effort, driving lasting improvements rather than short-term patches.

7. Data Governance: The Foundation of It All
Data governance is the framework of policies, standards, processes, roles, and responsibilities that define how an organization manages, uses, and protects its data assets. Think of it as the rulebook that ensures everyone is playing by the same data quality rules. Here’s why it’s so crucial:

Consistency across the organization: Data governance defines clear standards for data formats, definitions, quality metrics, and how data flows through the organization. This prevents data silos and ensures that everyone is working with the same understanding of what the data means and how reliable it is.
Accountability and ownership: Data governance outlines who is responsible for different data sets, who is authorized to access and modify them, and who should be involved in resolving data quality issues. This sense of ownership promotes accountability and helps ensure data is treated as a valuable asset.
Regulatory compliance: Data governance helps organizations meet complex data privacy and security regulations like GDPR, CCPA, and industry-specific standards. Consistent policies and processes make it easier to demonstrate compliance.
Improved trust:  When data is well-governed, users throughout the organization have more confidence in its accuracy and reliability. This leads to better adoption of analytics tools, more informed decision-making, and increased trust in data-driven insights.
Enabling data quality initiatives: Data governance isn’t just about rules. It also encompasses the processes and technologies that support data quality. This includes tools for data profiling, cleansing, monitoring, and lineage tracking.
Key Components of Data Governance:

Data policies: Formal, high-level statements about how data should be managed, protected, and used.
Data standards: Detailed definitions of data elements, formats, allowable values, and quality thresholds.
Data ownership and stewardship: Assigning clear roles for data owners (often business-side decision-makers) and data stewards (technical experts responsible for day-to-day management).
Processes for data lifecycle management: Defining processes for data creation, collection, storage, usage, and archival/deletion.
Metrics and reporting: Establishing ways to measure data quality, track progress, and communicate performance to stakeholders.
Data governance is an ongoing effort, not a one-time project. It requires strong leadership, cross-functional collaboration, and continuous adaptation as an organization’s data needs and the regulatory environment evolve.

## Others
arXiv:2410.18558  [pdf, other]  cs.CL
Infinity-MM: Scaling Multimodal Performance with Large-Scale and High-Quality Instruction Data

Authors: Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, Zhenchong Hu, Bo-Wen Zhang, Jijie Li, Dong Liang, Yingli Zhao, Yulong Ao, Yaoqi Liu, Fangxiang Feng, Guang Liu

Abstract: Vision-Language Models (VLMs) have recently made significant progress, but the limited scale and quality of open-source instruction data hinder their performance compared to closed-source models. In this work, we address this limitation by introducing Infinity-MM, a large-scale multimodal instruction dataset with 40 million samples, enhanced through rigorous quality filtering and deduplication. We also propose a synthetic instruction generation method based on open-source VLMs, using detailed image annotations and diverse question generation. Using this data, we trained a 2-billion-parameter VLM, Aquila-VL-2B, achieving state-of-the-art (SOTA) performance for models of similar scale. This demonstrates that expanding instruction data and generating synthetic data can significantly improve the performance of open-source models. 

### 大规模多模态
arXiv:2410.17337  [pdf, other]  cs.CL cs.AI cs.IR
Captions Speak Louder than Images (CASLIE): Generalizing Foundation Models for E-commerce from High-quality Multimodal Instruction Data

Authors: Xinyi Ling, Bo Peng, Hanwen Du, Zhihui Zhu, Xia Ning

Abstract: Leveraging multimodal data to drive breakthroughs in e-commerce applications through Multimodal Foundation Models (MFMs) is gaining increasing attention from the research community. However, there are significant challenges that hinder the optimal use of multimodal e-commerce data by foundation models: (1) the scarcity of large-scale, high-quality multimodal benchmark datasets; and (2) the lack of effective multimodal information integration methods. To address these challenges, in this paper, we introduce MMECInstruct, the first-ever, large-scale, and high-quality multimodal instruction dataset for e-commerce. We also develop CASLIE, a simple, lightweight, yet effective framework for integrating multimodal information for e-commerce. Leveraging MMECInstruct, we fine-tune a series of e-commerce MFMs within CASLIE, denoted as CASLIE models. Our comprehensive evaluation demonstrates that CASLIE models substantially outperform 5 categories of advanced baseline models in the in-domain evaluation. Moreover, CASLIE models show strong generalizability to out-of-domain settings. MMECInstruct and CASLIE models are publicly accessible through https://ninglab.github.io/CASLIE/.

### 数据过多需要数据蒸馏
arXiv:2410.09982  [pdf, other]  cs.LG cs.CL
Self-Data Distillation for Recovering Quality in Pruned Large Language Models

Authors: Vithursan Thangarasa, Ganesh Venkatesh, Nish Sinnadurai, Sean Lie

Abstract: Large language models have driven significant progress in natural language processing, but their deployment requires substantial compute and memory resources. As models scale, compression techniques become essential for balancing model quality with computational efficiency. Structured pruning, which removes less critical components of the model, is a promising strategy for reducing complexity. However, one-shot pruning often results in significant quality degradation, particularly in tasks requiring multi-step reasoning. To recover lost quality, supervised fine-tuning (SFT) is commonly applied, but it can lead to catastrophic forgetting by shifting the model's learned data distribution. Therefore, addressing the degradation from both pruning and SFT is essential to preserve the original model's quality. In this work, we propose self-data distilled fine-tuning to address these challenges. Our approach leverages the original, unpruned model to generate a distilled dataset that preserves semantic richness and mitigates catastrophic forgetting by maintaining alignment with the base model's knowledge. Empirically, we demonstrate that self-data distillation consistently outperforms standard SFT, improving average accuracy by up to 8% on the HuggingFace OpenLLM Leaderboard v1. Specifically, when pruning 6 decoder blocks on Llama3.1-8B Instruct (i.e., 32 to 26 layers, reducing the model size from 8.03B to 6.72B parameters), our method retains 91.2% of the original model's accuracy compared to 81.7% with SFT, while reducing real-world FLOPs by 16.30%. Furthermore, our approach scales effectively across datasets, with the quality improving as the dataset size increases. 

### LLM 里的数据质量：给出反馈函数
arXiv:2410.11540  [pdf, other]  cs.LG
Data Quality Control in Federated Instruction-tuning of Large Language Models

Authors: Yaxin Du, Rui Ye, Fengting Yuchi, Wanru Zhao, Jingjing Qu, Yanfeng Wang, Siheng Chen

Abstract: By leveraging massively distributed data, federated learning (FL) enables collaborative instruction tuning of large language models (LLMs) in a privacy-preserving way. While FL effectively expands the data quantity, the issue of data quality remains under-explored in the current literature on FL for LLMs. To address this gap, we propose a new framework of federated instruction tuning of LLMs with data quality control (FedDQC), which measures data quality to facilitate the subsequent filtering and hierarchical training processes. Our approach introduces an efficient metric to assess each client's instruction-response alignment (IRA), identifying potentially noisy data through single-shot inference. Low-IRA samples are potentially noisy and filtered to mitigate their negative impacts. To further utilize this IRA value, we propose a quality-aware hierarchical training paradigm, where LLM is progressively fine-tuned from high-IRA to low-IRA data, mirroring the easy-to-hard learning process. We conduct extensive experiments on 4 synthetic and a real-world dataset, and compare our method with baselines adapted from centralized setting. Results show that our method consistently and significantly improves the performance of LLMs trained on mix-quality data in FL. 

### 人工机器混合
arXiv:2410.11056  [pdf, other]  cs.CL
Beyond Human-Only: Evaluating Human-Machine Collaboration for Collecting High-Quality Translation Data

Authors: Zhongtao Liu, Parker Riley, Daniel Deutsch, Alison Lui, Mengmeng Niu, Apu Shah, Markus Freitag

Abstract: Collecting high-quality translations is crucial for the development and evaluation of machine translation systems. However, traditional human-only approaches are costly and slow. This study presents a comprehensive investigation of 11 approaches for acquiring translation data, including human-only, machineonly, and hybrid approaches. Our findings demonstrate that human-machine collaboration can match or even exceed the quality of human-only translations, while being more cost-efficient. Error analysis reveals the complementary strengths between human and machine contributions, highlighting the effectiveness of collaborative methods. Cost analysis further demonstrates the economic benefits of human-machine collaboration methods, with some approaches achieving top-tier quality at around 60% of the cost of traditional methods. We release a publicly available dataset containing nearly 18,000 segments of varying translation quality with corresponding human ratings to facilitate future research. 

### FYI：漏洞检测，影响数据准确性的
arXiv:2410.06030  [pdf, other]  cs.CR cs.AI 
doi
10.1109/EuroSPW59978.2023.00008
Data Quality Issues in Vulnerability Detection Datasets

Authors: Yuejun Guo, Seifeddine Bettaieb

Abstract: Vulnerability detection is a crucial yet challenging task to identify potential weaknesses in software for cyber security. Recently, deep learning (DL) has made great progress in automating the detection process. Due to the complex multi-layer structure and a large number of parameters, a DL model requires massive labeled (vulnerable or secure) source code to gain knowledge to effectively distinguish between vulnerable and secure code. In the literature, many datasets have been created to train DL models for this purpose. However, these datasets suffer from several issues that will lead to low detection accuracy of DL models. In this paper, we define three critical issues (i.e., data imbalance, low vulnerability coverage, biased vulnerability distribution) that can significantly affect the model performance and three secondary issues (i.e., errors in source code, mislabeling, noisy historical data) that also affect the performance but can be addressed through a dedicated pre-processing procedure. In addition, we conduct a study of 14 papers along with 54 datasets for vulnerability detection to confirm these defined issues. Furthermore, we discuss good practices to use existing datasets and to create new ones.





___
# New idea
```
在深度神经架构，损失函数，输入训练数据，正则化方面的robustness训练技术
```
- 深度神经架构：robust structure
- 损失函数：robust loss function, loss adjustment technique(loss correction 118, loss reweighting 41, label refurbishment 129)
- 输入训练数据：various sample selection technique
- 正则化

在深度神经网络的训练中，提高模型的鲁棒性是一个重要的研究方向。以下是在损失函数、输入训练数据和正则化方面的一些鲁棒性训练技术：

### 1. 损失函数方面
- **对抗性损失**：采用对抗样本生成技术，通过优化模型使其在对抗样本上表现良好。例如，使用对抗训练方法，将对抗样本与原始样本一起用于训练。
- **平滑损失**：使用标签平滑（Label Smoothing）技术，将真实标签稍微平滑化，减少模型对训练样本的过拟合。
- **鲁棒损失**：如Huber损失，它对离群点的敏感性较低，可以提高模型在数据噪声存在时的鲁棒性。

### 2. 输入训练数据方面
- **数据增强**：通过旋转、平移、缩放等方式扩展训练数据集，使模型更能适应不同的输入变化。
- **对抗训练**：在训练过程中引入对抗样本，增强模型对输入扰动的抵抗力。
- **样本选择**：选择具有代表性和挑战性的样本进行训练，以提高模型在边界样本上的表现。

### 3. 正则化方面
- **L2正则化**：通过增加权重的惩罚项，限制模型复杂度，防止过拟合。
- **Dropout**：随机丢弃一定比例的神经元，减少模型对特定神经元的依赖，从而提高鲁棒性。
- **数据扰动**：在训练过程中对输入数据进行小幅扰动，增加模型对输入变化的适应能力。

### 4. 其他技术
- **集成学习**：结合多个模型的预测结果，减少单一模型的偏差，提高整体的鲁棒性。
- **迁移学习**：在不同但相关的任务上进行训练，可以提高模型在目标任务上的表现和鲁棒性。

通过以上技术的结合使用，可以有效提高深度神经网络在各种不确定性条件下的鲁棒性。

___



___
在深度学习中，针对鲁棒性（robustness）的训练技术不断发展，以下是在深度神经架构、损失函数、输入训练数据和正则化方面的一些新模型和技术：

### 1. 深度神经架构
- **对抗性神经网络（Adversarial Neural Networks）**：如MAD（Min-Max Adversarial Defense）模型，结合对抗样本生成和模型训练，提升对抗鲁棒性。
- **鲁棒性强化的卷积神经网络（Robust CNNs）**：如P4CNN，通过设计特定的结构和卷积操作，增强对输入扰动的抵抗能力。
- **ResNet和其变体**：改进的ResNet模型，如ResNeXt，针对对抗样本的防御进行了优化。

### 2. 损失函数
- **对抗性损失函数**：如Margin-based对抗损失，旨在增加对抗样本和真实样本之间的边界，提高模型鲁棒性。
- **平滑损失函数**：如Spherical Loss，用于提升模型对噪声和输入扰动的鲁棒性。

### 3. 输入训练数据
- **MixUp和CutMix**：这些数据增强方法不仅混合样本，还结合了标签信息，提高模型在各种输入变化下的鲁棒性。
- **AutoAugment**：利用强化学习自动搜索数据增强策略，提升模型的泛化能力和鲁棒性。

### 4. 正则化
- **信息瓶颈方法（Information Bottleneck）**：通过控制信息流来提高模型鲁棒性，减少对特定输入特征的依赖。
- **多模型集成**：通过集成多个具有不同初始化和训练策略的模型，提高整体鲁棒性。

### 5. 新兴技术
- **鲁棒性优化（Robust Optimization）**：研究基于不确定性的方法，通过优化算法增强模型对输入扰动的鲁棒性。
- **动态权重调整（Dynamic Weight Adjustment）**：根据训练过程中的性能动态调整损失函数中的权重，使模型更加关注鲁棒性。

### 6. 迁移学习与自监督学习
- **自监督学习模型**：如SimCLR和BYOL，能够从无标签数据中学习表示，提高模型在下游任务中的鲁棒性。

这些新模型和技术的应用，能够显著提升深度神经网络在面对各种输入变化和对抗样本时的表现。随着研究的深入，未来可能会有更多创新方法涌现。

___
# Notes 2.0
### 深度神经架构
1. Adversarial Training Can Provably Improve Robustness: Theoretical Analysis of Feature Learning Process Under Structured Data

    Authors: Binghui Li, Yuanzhi Li

    首先，我们证明，通过使用标准训练（经验风险梯度下降），网络学习器主要学习非稳健特征，而不是稳健特征，从而导致对抗性示例由与非稳健特征负方向一致的扰动产生。 然后，我们考虑基于梯度的对抗训练算法，该算法运行梯度上升来寻找对抗示例，并在对抗示例的经验风险上运行梯度下降来更新模型。 我们证明，对抗训练法可以加强鲁棒特征学习，抑制非鲁棒特征学习，从而提高网络鲁棒性。 最后，我们还通过在真实图像数据集（包括 MNIST、CIFAR10 和 SVHN）上的实验验证了我们的理论发现。

    https://arxiv.org/pdf/2410.08503

2. 鲁棒性强化的卷积神经网络（CNN）在处理输入扰动和对抗攻击方面进行了多种改进。以下是一些 notable 的模型和方法：

    2.1. **P4CNN（Pyramid Pooling and Perceptual Loss for CNNs）**
   - 该模型通过金字塔池化和感知损失函数来提高鲁棒性，适用于多尺度特征学习。

    2.2. **DeepResilience**
   - 结合对抗训练与网络结构的改进，提升模型在对抗样本上的表现。

    2.3. **Truncated Networks**
   - 使用截断的神经网络，减少模型的复杂性，从而提高其对扰动的鲁棒性。

    2.4. **Adversarially Robust Training (ART)**
   - 采用对抗性样本进行训练，改进常规CNN结构以增强鲁棒性。

    2.5. **Defensive Distillation**
   - 通过知识蒸馏方法，将对抗训练与蒸馏结合，提高模型在对抗攻击下的性能。

    2.6. **STN（Spatial Transformer Networks）**
   - 引入空间变换模块，使网络对输入的几何变换（如旋转、缩放等）具有更强的鲁棒性。

    2.7. **Deep Residual Networks (ResNet) with Adversarial Training**
   - 将对抗训练应用于ResNet架构，增强对抗鲁棒性。

    2.8. **CutMix和MixUp** 放到第三点
   - 虽然不是特定的CNN架构，但这类数据增强技术通过混合训练样本提高模型对输入扰动的鲁棒性，可以与任何CNN架构结合使用。

    2.9. **R3D（Robust 3D CNNs）**
   - 针对视频数据的鲁棒性强化网络，利用三维卷积提高对视频干扰的抵抗力。

    2.10. **Noisy Student Training**
   - 在训练过程中引入噪声样本，增强模型对噪声和扰动的鲁棒性。

### Loss function 损失函数
1. Label smoothing

    Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It

    Authors: Guoxuan Xia, Olivier Laurent, Gianni Franchi, Christos-Savvas Bouganis

    https://arxiv.org/pdf/2403.14715

2. 将 GAN loss 当作优化函数对象：

    GANetic Loss for Generative Adversarial Networks with a Focus on Medical Applications

    Authors: Shakhnaz Akhmedova, Nils Körber

    https://arxiv.org/pdf/2406.05023

3. GAN loss 优化：蒙特卡罗 GAN（MCGAN）的算法。

    MCGAN: Enhancing GAN Training with Regression-Based Generator Loss

    Authors: Baoren Xiao, Hao Ni, Weixin Yang

    https://arxiv.org/pdf/2405.17191

4. margin-based loss

    4.1. **"Adversarial Training for Free!"**  
   - 论文提出了一种无额外计算代价的对抗训练方法，强调了边界的重要性，并探讨了如何通过边界提升鲁棒性。

    4.2. **"Towards Robust Neural Networks via Regularized Adversarial Training"**  
   - 该论文讨论了通过正则化和对抗训练结合的方法，提出了Margin-based对抗损失的变体，并验证了其在多种任务上的有效性。

    4.3. **"Theoretical Analysis of the Adversarial Vulnerability of Deep Neural Networks"**  
   - 这篇论文提供了对深度神经网络对抗脆弱性的理论分析，间接涉及了Margin-based损失的相关性。

    4.4. **"Generative Adversarial Training for Robustness and Generalization"**  
   - 研究了生成对抗网络在提升模型鲁棒性和泛化能力方面的应用，提到Margin-based方法在生成对抗样本时的优势。

### 输入训练数据
1. 数据增强

    通过对训练数据进行各种变换（如旋转、平移、缩放、裁剪、颜色变换等）来增加数据的多样性，从而提高模型对不同输入的适应能力。

2. 噪声注入
    
    在训练数据中添加随机噪声（如高斯噪声或盐和胡椒噪声），使模型更能适应现实世界中的数据不确定性。

3. 数据规范化
    
    对输入数据进行标准化（如零均值单位方差）或归一化（将数据缩放到特定范围），以减少特征之间的差异，使模型训练更加稳定。

4. 类别平衡

    确保训练数据中各个类别的样本数量相对均衡，避免模型对某些类别的偏向，从而提升在不平衡数据集上的性能。

5. 去噪处理 / self training
    5.1. **"Denoising Autoencoders"**  
   - Vincent, P., et al. (2008).  
   该论文介绍了去噪自编码器的概念，展示了如何通过对输入数据添加噪声来训练自编码器，使其能够学习数据的有用特征。

    5.2. **"Deep Learning for Image Denoising: A Review"**  
   - Zhang, K., et al. (2017).  
   这篇综述文章回顾了深度学习在图像去噪中的应用，分析了不同模型的效果和性能。

    5.3. **"BM3D: A New Approach to Image Denoising"**  
   - Dabbech, H., et al. (2012).  
   该论文提出了一种基于块匹配和3D变换的方法（BM3D）来进行图像去噪，取得了显著的效果。

    5.4. **"Non-local Means Denoising"**  
   - Buades, A., et al. (2005).  
   论文提出了非局部均值去噪方法，通过考虑图像中像素之间的相似性来有效去除噪声。

    5.5. **"Generative Adversarial Networks for Image Denoising"**  
   - Yang, Y., et al. (2018).  
   研究了生成对抗网络（GAN）在图像去噪中的应用，展示了其潜力和优势。

    5.6. **"A Survey on Denoising Techniques for Image Processing"**  
   - Kaur, R., et al. (2019).  
   该论文对各种图像去噪技术进行了综述，包括传统方法和基于深度学习的方法。

6. Cutmix 两项（前第一点里）

### 正则化 Normalization
1. DropBlock

描述：改进的Dropout方法，随机丢弃连续区域的特征图，从而增强模型对特定输入模式的鲁棒性。

2. Entropy Regularization

描述：通过增加模型输出分布的熵，使得模型在做决策时更加谨慎，从而提高对输入变化的鲁棒性。

3. Meta-Learning Regularization

    描述：通过在元学习框架中引入正则化，帮助模型更好地适应不同任务，提高泛化能力。

4. Progressive Growing
    
    描述：逐渐增加网络的复杂性，在训练早期使用较简单的模型，逐步增加层数和节点，从而提高模型的稳定性和鲁棒性。

5. Information Bottleneck（信息瓶颈）

    描述：一种用于理解和优化机器学习模型的信息论框架。其核心思想是通过最小化输入数据与模型预测之间的信息量，来提取出与预测任务最相关的信息。

6. 变分推断 (Variational Inference)

    概念：通过引入变分分布来近似复杂后验分布，从而进行高效的推断和学习。

    应用：常用于贝叶斯模型和生成模型中，如变分自编码器（VAE）。
7. 互信息最大化 (Mutual Information Maximization)

    概念：通过最大化输入和输出之间的互信息，提取最相关的特征。
    
    应用：在无监督学习和对比学习中，帮助提升特征的表达能力。
8. 稀疏编码 (Sparse Coding)
    
    概念：将数据表示为稀疏线性组合，以捕捉输入数据的结构信息，同时减少冗余。
    
    应用：广泛用于信号处理和图像处理，帮助提取有效特征。













