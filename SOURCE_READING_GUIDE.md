# HyperAMA EEG 源码学习方案

学习顺序是：**先建立整体大纲，理解每个部分的作用；再沿调用关系逐函数阅读；最后结合张量、公式和梯度逐行精读。**第一轮完成后再进入第二轮，避免同时处理模型思想和代码细节。

时间有限时，可先阅读 [五段核心代码讲解](CORE_CODE_GUIDE.md)：只看 EEG 编码、动态融合、双曲对齐、情绪辅助项和最终分类。源码中的 CORE-1 至 CORE-5 标记与五个阶段对应。

第一条主线固定为 FACED 九分类、动态融合 `pretrain_mode=2`、`hyper_distance`。读通后再加入 `hyper_emotion`，最后扩展到二分类和 SEED。本文中的行号对应 2026 年 10 月 9 日的本地源码；源码调整后以类名、函数名定位为准。文件链接相对于项目根目录。

## 项目目录树与逐文件说明

先阅读 [完整目录树与逐文件说明](PROJECT_FILE_MAP.md)，建立文件位置与模块职责的对应关系。索引展开所有现有文件，逐项说明作用和阅读位置，也区分历史配置、辅助脚本、数据资产及配置引用但当前副本未包含的文件。

主线目录按以下顺序认识；完整子目录与每个文件的解释见上述索引。

```text
HyperAMA-EEG-main/
├── cfgs/                   配置数据、模型、训练协议和实验开关
├── train_ext.py            组装预训练流程
├── data/                   读取样本、组织 batch、处理特征
├── model/                  定义 EEG 网络、训练封装和双曲损失
├── extract_features.py     加载检查点并保存 EEG 特征
├── train_mlp.py             训练分类器并计算指标
├── scripts/                串联三阶段实验
├── tests/                  验证数据协议和损失行为
├── tools/                  对照原始源码的计算
├── utils/                  日志、格式转换、重排与可视化
├── FACED/                  离线语义特征与已有特征切窗
├── multimodel_fusion/      文本图像对齐训练和分析图
├── data_preprocess/        MATLAB 参考预处理
└── After_remarks/          受试者行为记录与视频播放顺序
```

## 学习安排与完成标准

| 轮次 | 学习重点 | 暂时阅读的深度 | 完成时应能做到 |
| --- | --- | --- | --- |
| 第一轮 A | 大纲、模块职责、数据流 | 理解概念和模块关系，源码只作定位 | 用自己的话讲清输入、预训练、特征提取和分类 |
| 第二轮 B | 类与函数、调用关系、输入输出 | 以函数为单位，暂不展开每条计算语句 | 从入口追踪一个 batch，指出每个函数的上游和下游 |
| 第三轮 C | 逐行实现、数学原理、梯度 | 每次精读一小段连续代码 | 解释关键行做了什么、为什么这样写、改动会影响什么 |

每次学习按“回顾上次结论 → 一个小单元 → 复述 → 自检 → 记下疑问”推进。若自检不能独立完成，就回到当前单元补齐；不以阅读页数作为完成依据。

第一轮所需的基础概念是样本、标签、特征向量、训练与验证。张量广播、交叉熵、反向传播、双曲映射等内容，放到第三轮对应代码出现时再学习。

## 第一轮 A 先掌握大纲与各部分作用

这一轮先阅读下面的概念说明。源码位置用于建立模块与文件的对应关系，不要求逐行展开。

### A1 整体任务与两条工作流程

项目要完成跨受试者 EEG 情绪识别：使用一组受试者的数据学习表示，再对留出的受试者进行分类评估。不同人的 EEG 存在差异，文本和图像提供与刺激视频有关的语义信息，帮助训练 EEG 表示。

流程分成预训练、特征提取、分类器训练三个阶段。这里的“预训练”包含使用情绪标签的探针损失，启用 emotion 后还包含情绪监督，不能直接理解为完全不使用标签的自监督训练。

```text
预训练
EEG 窗口 → EEG 编码器 → EEG 投影头 ────────────────────┐
                                                       ├→ 对比损失 → 更新参数
文本与图像的离线特征 → 冻结对齐层 → 可训练投影层 → 融合目标 ─┘
          └→ 两个情绪探针 → 计算融合权重与探针损失

特征提取与下游分类
EEG 窗口 → 训练好的 EEG 主干 → 保存特征 → MLP → 情绪类别
```

**核心认识：**双曲几何参与预训练对齐；当前默认分类流程使用主干输出的普通 1024 维特征和 MLP。最终提取 EEG 特征时，不需要为每个输入同时提供文本与图像特征。

源码定位：[实验调度](scripts/run_ablation.py)、[预训练入口](train_ext.py)、[提取入口](extract_features.py)、[分类入口](train_mlp.py)。

自检：不打开源码，画出三个阶段的输入、输出，并标出文本与图像参与的阶段。

### A2 数据准备与 batch 组织

模型一次处理的是 EEG 时间窗口。FACED 主线中，每个窗口包含 32 个通道、5 秒信号，采样率为 125 Hz，因此每个通道有 625 个时间点。文本、图像特征已经离线提取，并按视频及窗口位置与 EEG 对应。

数据集负责“按索引取出一个完整样本”；采样器负责“决定哪些样本一起组成一个 batch”；DataModule 将数据准备、数据集和 DataLoader 接起来。

这里的预训练采样器会选择一对受试者，为两人取相同视频、相同窗口位置的数据，再拼接起来。默认九分类使用 28 个视频，一个 batch 通常由两名受试者各 28 个窗口组成，共 56 个样本。这种排列还服务于分组归一化和跨受试者负样本屏蔽。

源码定位：[FACEDDataModule](data/pl_datamodule.py)、[FACED_Dataset_new 与 PretrainSampler](data/dataset.py)。

自检：分别解释情绪标签、视频编号、受试者编号的作用；解释“样本是什么”和“batch 怎样组成”为什么是两个问题。

### A3 EEG 编码器与投影头

EEG 编码器把原始信号转换成便于学习和分类的向量。它先进行时间卷积，再使用不同时间感受野的时空卷积分支，随后结合局部、全局注意力，最后经过卷积和池化得到主干特征。

主干后面还有一个投影头，为对比学习转换表示。预训练使用投影后的向量；默认特征提取使用投影前的主干向量。二者虽然都是 1024 维，但来自不同位置，承担不同职责。

源码定位：[Conv_att_simple_new](model/models.py)。

自检：解释“主干”和“投影头”的区别，并在整体图中标出下游分类读取哪一处输出。

### A4 文本图像对齐与动态融合

文本和图像可能提供不同质量的情绪信息。动态融合会先用两个线性分类探针分别预测情绪，再根据它们对真实标签的分类损失确定权重：某种模态损失较小，就分配更大的权重。

语义目标的生成涉及三种不同模块：已有权重的冻结对齐层负责变换离线语义特征；可训练投影层适配当前 EEG 对齐任务；探针读取原始语义特征，帮助确定融合权重。FACED 训练时还会对 batch 内同一视频的权重取平均。

融合目标会随可训练投影层和探针的变化而变化。它不是一个始终不变的标签向量，也不是直接把文本类别和图像类别投票合并。

源码定位：[ExtractorModel 的初始化及 training_step](model/pl_models.py)。

自检：用“文本判断更准、图像判断较差”的例子说明权重应如何变化；指出被冻结和需要训练的模块。

### A5 对比学习与双曲几何

对比学习让 EEG 表示靠近与它配对的语义目标，同时与有效负样本区分。一个 batch 有 B 个 EEG 和 B 个语义目标，就可以得到 B × B 的匹配分数矩阵。

`hyper_distance` 将归一化表示映射到庞加莱球，以负的双曲距离计算匹配分数，然后送入对比损失。距离越小，匹配分数越大。主干中的卷积和 MLP 本身仍是普通神经网络运算。

主对齐损失的正样本是对应配对。对于“同一视频、不同受试者”的非对应配对，代码将其屏蔽，减少假负样本干扰；被屏蔽不等于被加入正样本集合。

`hyper_emotion` 进一步添加情绪级损失，把同情绪的多个融合语义目标作为一个 EEG 样本的正样本。主对齐项和情绪项分别使用自己的样本关系。

源码定位：[MultiModalInfoNCELoss](model/loss/con_loss.py)、[ContrastiveGeometry 与情绪损失](model/loss/hyperbolic.py)。

自检：解释双曲距离位于流程的哪一步，以及“配对对齐”和“同情绪聚合”各自鼓励什么。

### A6 训练管理与验证

训练入口组装配置、受试者划分、数据和模型。Lightning 负责重复调用训练与验证函数、执行自动反向传播和参数更新。训练函数返回损失；验证函数计算指标；检查点与早停根据指定的验证指标选模型。

需要分清三个角色：编码器负责计算表示，Lightning 模型负责定义一个 batch 的训练行为，Trainer 负责组织整个训练循环。外层模型的 `forward` 与训练时调用的内部编码器 `forward` 也需要分开追踪。

源码定位：[train_ext](train_ext.py)、[ExtractorModel](model/pl_models.py)。

自检：讲清一次 batch 从数据加载到参数更新经过哪些角色，以及保存检查点的指标从哪里产生。

### A7 特征提取与 MLP 分类

预训练后，程序加载检查点，用 EEG 主干提取特征。默认还会对输入做基于训练受试者统计量的归一化，并在每个 trial 内对提取特征做 LDS 平滑。随后保存特征，供 MLP 分类器使用。

MLP 阶段读取这些特征，使用训练集拟合标准化变换，再训练普通多层感知机预测情绪。预训练检索指标与最终 MLP 分类指标来自不同过程，应分别理解。

源码定位：[extract_features](extract_features.py)、[feature_processing](data/feature_processing.py)、[train_mlp](train_mlp.py)。

自检：解释为何预训练损失下降、检索准确率上升和 MLP 准确率上升并不是同一件事。

### 第一轮的项目目录地图

| 位置 | 职责 | 阅读时机 |
| --- | --- | --- |
| [cfgs](cfgs) | 数据、模型、训练协议、实验开关 | 第一轮认识，第二轮核对生效配置 |
| [scripts/run_ablation.py](scripts/run_ablation.py) | 串起三阶段实验并记录状态 | 第二轮入口 |
| [data](data) | 数据读取、索引、采样、特征处理 | 第二轮数据链 |
| [model/models.py](model/models.py) | EEG 网络、投影模块、MLP 网络 | 第二轮网络链 |
| [model/pl_models.py](model/pl_models.py) | FACED 的训练、验证、预测逻辑 | 第二轮核心 |
| [model/loss](model/loss) | 主对齐损失、双曲几何、辅助情绪损失 | 第二轮损失链 |
| [FACED](FACED)、[data_preprocess](data_preprocess) | 数据与离线语义特征的准备脚本 | 主线读通后按需补读 |
| [multimodel_fusion](multimodel_fusion) | 文本图像对齐相关脚本与权重 | 先知道权重用途，再追踪生成过程 |
| [tests](tests) | 用具体输入与断言描述预期行为 | 第三轮对应单元配套阅读 |
| [HYPERAMA_EXPERIMENTS.md](HYPERAMA_EXPERIMENTS.md)、[REPRODUCIBILITY.md](REPRODUCIBILITY.md) | 实验选项与协议说明 | 第一轮完成后对照阅读 |

**第一轮完成产物：**一张整体流程图、一张模块职责表，以及对 A1 至 A7 自检问题的口头或书面回答。此时先不要求推导双曲公式。

## 第二轮 B 沿调用关系逐函数阅读

每个函数先填写六项：它解决什么问题、谁调用它、输入是什么、输出是什么、调用了谁、是否改变状态或参数。能填写这六项后再进入函数内部细节。

### B1 配置与入口

依次阅读 [公共配置](cfgs/config.yaml)、[FACED 配置](cfgs/data/FACED.yaml)、[模型配置](cfgs/model/cnn_att_dual.yaml)、[FACED 协议](cfgs/protocol/FACED.yaml)、[实验覆盖](cfgs/experiment/hyper_distance.yaml)。

记录实际主线：`pretrain_mode=2`、`geometry=hyperbolic`、固定曲率参数 `c=0.1`、对比温度 `0.07`、融合温度 `0.1`、情绪项关闭。`fusion_alpha=0.7` 用于静态融合分支，不是动态融合的固定权重。

然后阅读 [run_ablation.py](scripts/run_ablation.py) 的 `main`（41 行）与 `verify_features`（24 行），以及 [train_ext.py](train_ext.py) 的 `train_ext`（95 行）。重点找受试者划分、模型创建、回调创建和 `trainer.fit`。

最后检查 [get_extractor_class](model/__init__.py)（5 行）及入口中的配置改写。FACED 入口会设置 `proj_type` 和 `use_ln_backbone`；判断实际网络时，要把配置合成和 Python 内的赋值一起考虑。

产物：一张“配置项 → 使用位置 → 控制行为”的表。

### B2 数据函数

| 顺序 | 文件与函数 | 先理解的接口 |
| --- | --- | --- |
| 1 | [pl_datamodule.py](data/pl_datamodule.py) `FACEDDataModule.__init__`，110 行 | 保存数据路径、受试者划分和采样设置 |
| 2 | 同文件 `prepare_data` / `setup`，135 / 141 行 | 准备缓存，并创建训练与验证数据集 |
| 3 | 同文件 `train_dataloader` / `val_dataloader`，164 / 177 行 | 将数据集和采样器接成数据流 |
| 4 | [dataset.py](data/dataset.py) `FACED_Dataset_new.__init__`，21 行 | 加载窗口元信息及文本图像特征 |
| 5 | 同类 `__len__` / `__getitem__`，118 / 126 行 | 从局部索引定位全局样本，并返回配对数据 |
| 6 | 同文件 `PretrainSampler.__init__` / `__len__` / `__iter__`，533 / 559 / 562 行 | 构造受试者配对、报告 batch 数、生成索引批次 |

FACED 多模态路径的返回值是 `eeg, label, txt_feat, img_feat, vid_id, sub_id`。`seg_id` 在数据集内部用于选窗口，没有作为第七项返回。

补读数据准备时，再进入 [io_utils.py](data/io_utils.py) 的 `load_processed_FACED_NEW_data`（33 行）与 `save_sliced_data`（449 行）。

产物：一个 batch 的字段表，注明形状、含义和用于哪项计算。

### B3 编码器函数

阅读 [models.py](model/models.py)：先看 `Conv_att_simple_new.forward`（172 行），再回到 `__init__`（54 行）核对层的定义，随后看 `stratified_layerNorm`（21 行）、`ResidualAdd.forward`（17 行）和 `set_saveFea`（280 行）。

第一次读 `forward` 时只标出六段：输入归一化、卷积、注意力、后续卷积、池化、输出分支。先写清各段输入输出，再逐层计算形状。

产物：主干与投影头结构图，并标注 `saveFea` 两种取值对应的出口。

### B4 训练封装与融合函数

阅读 [pl_models.py](model/pl_models.py) 的 `ExtractorModel.__init__`（33 行）、`configure_optimizers`（213 行）和 `training_step`（220 行）。初始化中的大段历史注释可以先略过，重点找到实际创建的模块。

在 `training_step` 中先跟随 mode 2：解包数据 → 类型处理 → 切换输出模式 → 冻结对齐 → 可训练投影 → 探针 → 融合权重 → 语义融合 → EEG 前向 → 损失汇总。

产物：模块更新表，标出冻结对齐层、语义投影层、探针、EEG 主干和 EEG 投影头分别由什么损失更新。

### B5 几何与损失函数

| 顺序 | 文件与函数 | 核心职责 |
| --- | --- | --- |
| 1 | [hyperbolic.py](model/loss/hyperbolic.py) `ContrastiveExperimentMixin._init_contrastive_experiment`，122 行 | 根据配置创建几何对象并接入主损失 |
| 2 | [con_loss.py](model/loss/con_loss.py) `MultiModalInfoNCELoss.__init__` / `forward`，202 / 237 行 | 建立分数矩阵、屏蔽假负样本、计算双向交叉熵 |
| 3 | [hyperbolic.py](model/loss/hyperbolic.py) `ContrastiveGeometry.__init__` / `_normalize` / `project` / `forward`，15 / 27 / 30 / 38 行 | 设置曲率、归一化、映射、计算两两距离 |
| 4 | 同文件 `emotion_positive_mask`，50 行 | 根据情绪标签建立多正样本关系 |
| 5 | 同文件 `multi_positive_loss` 及内部 `directional`，63 / 65 行 | 对正样本的对数概率求平均，并处理双向形式 |
| 6 | 同文件 `MultiScaleContrastiveLoss.__init__` / `forward`，82 / 100 行 | 管理可选辅助项及其权重 |
| 7 | 同文件 `ContrastiveExperimentMixin._scale_losses`，137 行 | 调用辅助项并记录日志 |

先完成前三步，确认 `hyper_distance` 的完整路径，再打开 [hyper_emotion.yaml](cfgs/experiment/hyper_emotion.yaml) 跟随后四步。当前辅助尺度注册表只有 emotion；类名含有 MultiScale 不代表已经启用了多种辅助损失。

产物：损失调用图，以及主对齐掩码和情绪正样本掩码的区别说明。

### B6 验证函数

阅读 [pl_models.py](model/pl_models.py) 的 `validation_step`（382 行），按训练函数的同样顺序追踪。逐项比较特征来源、融合权重、损失、检索指标和其他诊断指标。

重点定位 443 行附近逐样本计算的融合权重，以及 502 行附近的检索分数。随后回到 [train_ext.py](train_ext.py) 186 行，确认检查点和早停监听的指标。

产物：一张训练与验证差异表，避免仅因函数结构相似就认为计算完全相同。

### B7 特征提取函数

阅读 [extract_features.py](extract_features.py) 的 `load_extraction_data`（21 行）、`extraction_labels`（42 行）和 `extract_features`（77 行）；再看 [pl_models.py](model/pl_models.py) 的 `predict_step`（614 行）如何调用外层 `forward`（199 行）。

继续追踪 [feature_processing.py](data/feature_processing.py) 的 `normalize_extraction_input`（20 行）及 `smooth_trial_features`（46 行）。注意提取阶段关闭分组归一化的设置，以及默认从主干输出特征的设置。平滑的数学细节最后再看 [LDS](data/data_process.py)（3 行）。

产物：从检查点到特征文件的数据流图，标注每项统计量在哪个数据子集上计算。

### B8 MLP 函数与完整调用图

阅读 [train_mlp.py](train_mlp.py) 的 `train_mlp`（68 行），再读 [models.py](model/models.py) 的 `simpleNN3.__init__` / `forward`（296 / 319 行），以及 [pl_models.py](model/pl_models.py) 的 `MLPModel`（626 行）。

MLP 封装中依次读 `configure_optimizers`、`training_step`、`validation_step`、`predict_step`。最后看分类入口如何重新加载检查点、计算和汇总指标。

完整主线应能整理为：

```text
run_ablation.main
  ├─ train_ext
  │    ├─ FACEDDataModule → Dataset + PretrainSampler → batch
  │    ├─ Conv_att_simple_new + ExtractorModel
  │    └─ Trainer.fit
  │         ├─ training_step → 动态融合 + EEG forward → 损失
  │         │                   └─ InfoNCE → ContrastiveGeometry
  │         │                   └─ 可选 emotion 损失
  │         ├─ Lightning 自动反向传播与优化器更新
  │         └─ validation_step → 指标 → 检查点与早停
  ├─ extract_features
  │    └─ predict_step → ExtractorModel.forward → 主干特征 → 平滑与保存
  └─ train_mlp
       └─ 标准化 → simpleNN3 + MLPModel → 选模与指标
```

**第二轮完成产物：**能够从一个 batch 的来源追踪到损失，再从检查点追踪到最终分类结果；每个关键函数都能填写输入输出和调用关系。

## 第三轮 C 逐函数逐行弄懂实现原理

每个单元先回顾对应模块的作用，再精读。一个单元可能涉及多个小段，每次讲解以一个自然代码块为边界，通常控制在 10 至 30 行；较长函数分多次完成。

### 每段代码的固定讲解顺序

1. **上下文：**当前位于哪一阶段、哪个函数，上一段产生了什么。
2. **代码原文：**给出当前需要解释的小段及其源码位置。
3. **逐行语义：**解释语法、变量意义、执行顺序和状态变化。
4. **张量变化：**写出形状、数据类型、广播方向和每个轴的含义。
5. **数学对应：**将运算对应到公式；简单代码只需说明运算关系。
6. **梯度与训练作用：**标出可训练参数、`detach`、`no_grad` 与冻结边界。
7. **具体例子：**使用小 batch 或少量数值手算，检查理解。
8. **自检：**预测改变一项配置或输入后会发生什么，并回到代码解释原因。

学习笔记按以下模板填写。无需梯度或公式的工具函数，可将相应栏标为“不涉及”。

```text
单元与源码位置：
这段代码解决的问题：
上游输入与下游使用者：
逐行解释：
输入形状 → 中间形状 → 输出形状：
对应公式或计算关系：
哪些参数收到梯度：
依赖的配置与隐含前提：
小例子的预期结果：
仍需确认的问题：
```

### 逐行精读单元与顺序

| 单元 | 源码位置与阅读范围 | 必须弄懂的原理 | 自检产物 |
| --- | --- | --- | --- |
| C01 配置与实例化 | `cfgs` 主线配置；`train_ext` 中模型创建；`get_extractor_class` | 配置插值、覆盖、对象创建；默认参数与实际传入值的区别 | 记录模型、两个温度和曲率的最终来源 |
| C02 样本索引 | `dataset.py` 的 FACED 初始化、118 行 `__len__`、126 行 `__getitem__` | 整除、取余、局部到全局索引、视频与窗口对齐 | 用一个索引算出受试者、视频和窗口 |
| C03 采样与归一化 | `PretrainSampler` 533 至 594 行，分段阅读；`models.py` 21 行 | 受试者配对、拼接顺序、分组统计、reshape 与 permute | 画出前后两半 batch，解释打乱排列的影响 |
| C04 主干卷积 | `models.py` 54 行初始化；172 至 184 行前向 | 卷积核、padding、dilation、groups、通道拼接 | 逐层列出时间长度与通道数 |
| C05 注意力与输出 | `models.py` 186 至 277 行，按注意力、后续卷积、出口拆开 | 广播、逐元素加权、池化、残差投影、saveFea | 写出主干向量与投影向量的生成位置 |
| C06 模块初始化与冻结 | `pl_models.py` 108 至 197 行及 213 行优化器 | 加载权重、随机数状态、eval、requires_grad、no_grad 的区别 | 完成各模块的更新关系表 |
| C07 训练准备与探针 | `pl_models.py` 220 至 293 行，先沿九分类 mode 2 | 解包、dtype、detach、线性分类、交叉熵 | 标注语义投影和探针各读哪一份特征 |
| C08 动态权重与融合 | `pl_models.py` 295 至 318 行；368 至 380 行 | 逐样本损失、sigmoid、按视频聚合、广播、总损失 | 手算一个融合权重并解释梯度去向 |
| C09 主对齐损失 | `con_loss.py` 237 至 320 行，按分数、掩码、交叉熵拆开 | 两两矩阵、布尔掩码、对角配对、双向损失 | 手画 4 × 4 主损失掩码 |
| C10 双曲几何 | `hyperbolic.py` 14 至 47 行及 122 至 133 行 | 单位归一化、指数映射、负曲率、负距离、FP32、对象接入 | 写出从特征到分数的完整公式 |
| C11 情绪多正样本 | `hyperbolic.py` 50 至 117 行；137 至 144 行 | 同标签关系、log_softmax、正样本平均、权重与对称计算 | 将情绪掩码与 C09 掩码并排比较 |
| C12 验证与模型选择 | `pl_models.py` 382 行起按分支阅读，重点 443、502 行；`train_ext.py` 186 行 | 训练验证差异、检索标签、指标聚合、早停 | 画出验证标签参与计算的位置 |
| C13 提取与预处理 | `extract_features`；`predict_step`；`normalize_extraction_input`；`smooth_trial_features` | 检查点恢复、输出模式、训练统计量、trial 边界 | 从单个 EEG 窗口追踪到保存的特征行 |
| C14 分类与评估 | `train_mlp` 中划分、标准化、训练、检查点恢复；`simpleNN3`；`MLPModel` | 训练集拟合变换、分类 CE、准确率、macro F1、kappa | 区分预训练检索分数与最终分类分数 |

C04 中按实际参数计算卷积长度，不要假设 padding 总能保持输入长度。C06 中，`eval()` 改变 Dropout 等层的行为，参数冻结阻止参数求梯度，`no_grad()` 控制计算图记录，三者作用不同。

C10 学习双曲原理时，先理解代码里的单位向量、映射和距离，再补充庞加莱球的数学定义。当前固定曲率并先做单位归一化，非零向量映射后的半径基本相同；不能仅凭使用双曲距离就认定模型已经学习到径向层级。

### C08 的逐行讲解示例

以下是 [pl_models.py](model/pl_models.py) 297 至 300 行的实际计算，位于 `torch.no_grad()` 块内：

```python
loss_t_sample = loss_func_none(t_logits, labels)
loss_i_sample = loss_func_none(i_logits, labels)
diff = (loss_i_sample - loss_t_sample) / self.conf_temp
inst_alpha = torch.sigmoid(diff).unsqueeze(1)
```

第一行计算每个文本样本的分类损失。因为交叉熵采用 `reduction='none'`，输入类别分数 `[B, 9]` 和标签 `[B]` 后，返回 `[B]`，而不是一个 batch 平均标量。

第二行计算图像探针的逐样本损失，形状同样是 `[B]`。两种模态使用同一组真实情绪标签，因而可以比较它们在当前样本上的分类损失。

第三行先用图像损失减去文本损失，再除以融合温度。如果文本预测更好，文本损失更小，差值为正。较小的融合温度会放大同样的损失差异。

第四行用 sigmoid 将差值变成 0 到 1 之间的文本权重，再从 `[B]` 变为 `[B, 1]`。后续与 `[B, 1024]` 的文本特征相乘时，权重沿特征维广播，每个样本的 1024 个维度共享一个权重。

数学对应为：

\[
\alpha_i=\sigma\left(\frac{\ell_{\mathrm{image},i}-\ell_{\mathrm{text},i}}{T_{\mathrm{conf}}}\right).
\]

例如文本损失为 0.2、图像损失为 0.4、融合温度为 0.1，则差值为 2，文本权重约为 0.881。两者损失相等时，权重为 0.5。这里的数值是讲解例子。

此处计算不建立梯度图。探针由前面单独计算的探针损失训练；语义投影层和 EEG 网络由后续对比损失训练。继续阅读 302 至 313 行，才能看到 FACED 训练时同视频权重平均、加权融合及再次归一化的完整过程。

自检：如果互换两种模态的损失，文本权重应如何变化？如果省略 `unsqueeze(1)`，为什么不能按预期给每个特征向量加权？

### C09 与 C11 共用的小 batch 练习

使用以下四个样本，行表示 EEG，列表示融合语义目标：

```text
样本编号：  0  1  2  3
视频编号：  0  1  0  1
受试者号：  0  0  1  1
情绪标签：  0  1  0  1
```

主对齐损失的对角线为正样本；`(0,2)`、`(2,0)`、`(1,3)`、`(3,1)` 因同视频且不同受试者被屏蔽。其他非对角项保留为负样本。

情绪损失中，第 0、2 行的正目标列为 0、2，第 1、3 行的正目标列为 1、3。它使用独立的情绪关系，不继承主损失的屏蔽矩阵。这里选用了简化标签；真实 batch 中，不同视频也可能共享同一情绪标签。

情绪项对每个正样本的对数概率求平均，其形式是 `-mean(log p_positive)`，需要与 `-log(sum(p_positive))` 区分。先手画两个矩阵，再对照实现。

## 用测试核对学习结论

阅读测试时按“输入与配置 → 调用对象 → 断言 → 断言背后的原因”展开。以下是配套阅读位置，不要求为学习启动完整训练。

| 对应单元 | 测试位置 | 用来核对的理解 |
| --- | --- | --- |
| C09 | [test_hyperbolic.py](tests/test_hyperbolic.py) 55 行 | 换成双曲距离后，原有主损失掩码仍如何工作 |
| C10 | 同文件 41、148 行 | 映射与闭式公式、几何计算精度与有限梯度 |
| C11 | 同文件 74、90、99 行 | 多正样本公式、关闭辅助项、两组实验配置差异 |
| C06 至 C12 | 同文件 114 行 | 训练、验证、检查点保存与恢复的连接 |
| C08 与 C12 | [test_protocols.py](tests/test_protocols.py) 111、114 行 | 训练分组融合与验证逐样本融合 |
| C13 | 同文件 179、256 行 | 归一化统计量来源与平滑的 trial 边界 |

## 阅读中需要持续核对的区别

| 容易混淆的内容 | 当前 FACED 主线应如何理解 |
| --- | --- |
| 文件里存在的类与实际使用的类 | 沿实例化和调用路径确认；导入或定义不等于参与当前训练 |
| 代码注释与实际计算 | 优先核对语句和生效配置，例如 BatchNorm 并不保证数值位于 0 到 1 |
| 两种温度 | `loss_temp` 调整对比分数，`conf_temp` 调整融合权重 |
| 多种投影层 | 冻结语义对齐层、可训练语义投影层、EEG 投影头有不同职责 |
| 训练输出与提取输出 | 默认分别读取 EEG 投影头之后与之前的表示 |
| 主损失屏蔽与情绪正样本 | 屏蔽是假负样本处理；同情绪正样本由独立关系定义 |
| 训练与验证融合 | FACED 训练按视频平均权重，验证逐样本计算权重 |
| `ZeroShot_Acc` 与无标签推理 | 当前验证融合使用真实标签计算权重，指标含义必须结合这条路径理解 |
| 留出折与独立测试集 | 当前留出折参与选模并用于报告结果，不应当作完全隔离的独立测试集 |

## 主线完成后的扩展顺序

1. **FACED 二分类。**阅读训练、验证和 MLP 中去除中性类及重映射标签的代码，检查 EEG、语义特征、标签、视频编号和受试者编号是否同步筛选。
2. **其他融合模式。**阅读 `pretrain_mode` 0、1、3，比较纯文本、纯图像和静态融合。每次只变更阅读分支，先写预期行为再查代码。
3. **SEED。**从 [get_extractor_class](model/__init__.py)、[SEED 训练封装](model/seed_pl_models.py)、[SEED 数据模块](data/pl_datamodule.py) 和对应配置进入，重新核对 session、采样、融合分组与归一化协议。
4. **离线准备与数学细节。**追踪语义特征生成、对齐权重生成、LDS 实现；需要研究推导时再深入双曲距离及梯度公式。

## 学习进度与后续讲解方式

- [ ] 第一轮 A1 至 A7：能复述整体流程和各模块职责。
- [ ] 第二轮 B1 至 B3：能追踪配置、数据与编码器。
- [ ] 第二轮 B4 至 B6：能追踪融合、损失与验证。
- [ ] 第二轮 B7 至 B8：能追踪特征提取与分类。
- [ ] 第三轮 C01 至 C05：能逐行解释配置、索引、采样和主干。
- [ ] 第三轮 C06 至 C08：能逐行解释冻结、探针和动态融合。
- [ ] 第三轮 C09 至 C11：能逐行解释主损失、双曲几何和情绪项。
- [ ] 第三轮 C12 至 C14：能逐行解释验证、提取和分类评估。
- [ ] 完成小 batch 练习，并用配套测试核对结论。

后续讲解直接使用单元编号，例如“从 A1 开始，只讲整体任务和各部分作用”“继续 B2，先解释函数之间的调用关系”“开始 C08，每次解释一个代码块，配形状和手算例子”。先按第一轮顺序完成大纲，再逐步增加代码深度。
