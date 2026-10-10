# HyperAMA EEG 目录树与逐文件说明

本索引用于先认识文件分工，再按 [源码学习方案](SOURCE_READING_GUIDE.md) 进入函数与逐行阅读。文件用途按当前源码中的定义、调用及输入输出说明。

目录快照日期为 2026 年 10 月 10 日，共 **208 个文件**：85 个源码、配置、文档、脚本及图片文件，另有 123 个受试者 MATLAB 行为元数据文件。下方完整列出实际文件；After_remarks 中只有一个文件的子目录合并显示为 subXXX/After_remarks.mat，每个文件均附用途。

阅读标记中，“核心”对应当前 FACED 主线；“扩展”用于后续二分类、SEED 或结构比较；“历史”表示保留文件及其当前使用限制；其余为环境、说明、数据资产或辅助功能。学习单元编号与源码学习方案一致。

## 完整目录树

```text
HyperAMA-EEG-main/
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── CORE_CODE_GUIDE.md
├── DATA.md
├── extract_features.py
├── HYPERAMA_EXPERIMENTS.md
├── LICENSE
├── PROJECT_FILE_MAP.md
├── project_seed_features.py
├── README.md
├── REPRODUCIBILITY.md
├── requirements-preprocess.txt
├── requirements.txt
├── SECURITY.md
├── SOURCE_READING_GUIDE.md
├── train_ext.py
├── train_mlp.py
├── cfgs/
│   ├── config.yaml
│   ├── data/
│   │   ├── FACED_def_c2.yaml
│   │   ├── FACED.yaml
│   │   └── SEED.yaml
│   ├── experiment/
│   │   ├── hyper_distance.yaml
│   │   └── hyper_emotion.yaml
│   ├── model/
│   │   ├── cnn_att_dual.yaml
│   │   ├── cnn_att.yaml
│   │   ├── cnn_clisa.yaml
│   │   └── cnn_trans.yaml
│   ├── protocol/
│   │   ├── FACED_def_c2.yaml
│   │   ├── FACED.yaml
│   │   └── SEED.yaml
│   └── readme_cfg.md
├── data/
│   ├── __init__.py
│   ├── data_process.py
│   ├── dataset.py
│   ├── feature_processing.py
│   ├── io_utils.py
│   └── pl_datamodule.py
├── model/
│   ├── __init__.py
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── con_loss.py
│   │   ├── hyperbolic.py
│   │   └── seed_con_loss.py
│   ├── metric/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── models.py
│   ├── pl_models.py
│   └── seed_pl_models.py
├── scripts/
│   ├── launch_ablation.sh
│   └── run_ablation.py
├── tests/
│   ├── test_hyperbolic.py
│   ├── test_protocols.py
│   └── test_repository.py
├── tools/
│   └── check_source_parity.py
├── utils/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── experiment_metrics.py
│   ├── inspect_image_features_1024.py
│   ├── move_checkpoints.py
│   ├── preprocess.py
│   ├── reorder_vids.py
│   ├── utils.py
│   ├── vis_feature_space.py
│   └── vis_text_compare.py
├── FACED/
│   ├── FACED_get_image_feature_clip_vit.py
│   ├── FACED_Qwen_nolabel.py
│   ├── move_window_timelen5_tiemstep2_photo.py
│   ├── move_window_timelen5_timestep2_EEG.py
│   ├── move_window_timelen5_timestep2_text.py
│   └── README.md
├── data_preprocess/
│   ├── AutoICA_SEED.m
│   ├── nt_find_bad_channels_custom.m
│   └── README.md
├── multimodel_fusion/
│   ├── fusion10/
│   │   ├── figures/
│   │   │   ├── 01_before_by_modality.png
│   │   │   ├── 02_after_by_modality.png
│   │   │   ├── 03_text_shift.png
│   │   │   ├── 04_before_by_emotion.png
│   │   │   └── 05_after_by_emotion.png
│   │   ├── README.md
│   │   └── train_text_image_alignment.py
│   ├── seed_fusion1/
│   │   ├── README.md
│   │   └── train_text_image_alignment.py
│   └── visualize_alignment.py
├── .github/
│   └── workflows/
│       └── ci.yml
└── After_remarks/
    ├── sub000/After_remarks.mat  # sub000 的行为记录与播放顺序
    ├── sub001/After_remarks.mat  # sub001 的行为记录与播放顺序
    ├── sub002/After_remarks.mat  # sub002 的行为记录与播放顺序
    ├── sub003/After_remarks.mat  # sub003 的行为记录与播放顺序
    ├── sub004/After_remarks.mat  # sub004 的行为记录与播放顺序
    ├── sub005/After_remarks.mat  # sub005 的行为记录与播放顺序
    ├── sub006/After_remarks.mat  # sub006 的行为记录与播放顺序
    ├── sub007/After_remarks.mat  # sub007 的行为记录与播放顺序
    ├── sub008/After_remarks.mat  # sub008 的行为记录与播放顺序
    ├── sub009/After_remarks.mat  # sub009 的行为记录与播放顺序
    ├── sub010/After_remarks.mat  # sub010 的行为记录与播放顺序
    ├── sub011/After_remarks.mat  # sub011 的行为记录与播放顺序
    ├── sub012/After_remarks.mat  # sub012 的行为记录与播放顺序
    ├── sub013/After_remarks.mat  # sub013 的行为记录与播放顺序
    ├── sub014/After_remarks.mat  # sub014 的行为记录与播放顺序
    ├── sub015/After_remarks.mat  # sub015 的行为记录与播放顺序
    ├── sub016/After_remarks.mat  # sub016 的行为记录与播放顺序
    ├── sub017/After_remarks.mat  # sub017 的行为记录与播放顺序
    ├── sub018/After_remarks.mat  # sub018 的行为记录与播放顺序
    ├── sub019/After_remarks.mat  # sub019 的行为记录与播放顺序
    ├── sub020/After_remarks.mat  # sub020 的行为记录与播放顺序
    ├── sub021/After_remarks.mat  # sub021 的行为记录与播放顺序
    ├── sub022/After_remarks.mat  # sub022 的行为记录与播放顺序
    ├── sub023/After_remarks.mat  # sub023 的行为记录与播放顺序
    ├── sub024/After_remarks.mat  # sub024 的行为记录与播放顺序
    ├── sub025/After_remarks.mat  # sub025 的行为记录与播放顺序
    ├── sub026/After_remarks.mat  # sub026 的行为记录与播放顺序
    ├── sub027/After_remarks.mat  # sub027 的行为记录与播放顺序
    ├── sub028/After_remarks.mat  # sub028 的行为记录与播放顺序
    ├── sub029/After_remarks.mat  # sub029 的行为记录与播放顺序
    ├── sub030/After_remarks.mat  # sub030 的行为记录与播放顺序
    ├── sub031/After_remarks.mat  # sub031 的行为记录与播放顺序
    ├── sub032/After_remarks.mat  # sub032 的行为记录与播放顺序
    ├── sub033/After_remarks.mat  # sub033 的行为记录与播放顺序
    ├── sub034/After_remarks.mat  # sub034 的行为记录与播放顺序
    ├── sub035/After_remarks.mat  # sub035 的行为记录与播放顺序
    ├── sub036/After_remarks.mat  # sub036 的行为记录与播放顺序
    ├── sub037/After_remarks.mat  # sub037 的行为记录与播放顺序
    ├── sub038/After_remarks.mat  # sub038 的行为记录与播放顺序
    ├── sub039/After_remarks.mat  # sub039 的行为记录与播放顺序
    ├── sub040/After_remarks.mat  # sub040 的行为记录与播放顺序
    ├── sub041/After_remarks.mat  # sub041 的行为记录与播放顺序
    ├── sub042/After_remarks.mat  # sub042 的行为记录与播放顺序
    ├── sub043/After_remarks.mat  # sub043 的行为记录与播放顺序
    ├── sub044/After_remarks.mat  # sub044 的行为记录与播放顺序
    ├── sub045/After_remarks.mat  # sub045 的行为记录与播放顺序
    ├── sub046/After_remarks.mat  # sub046 的行为记录与播放顺序
    ├── sub047/After_remarks.mat  # sub047 的行为记录与播放顺序
    ├── sub048/After_remarks.mat  # sub048 的行为记录与播放顺序
    ├── sub049/After_remarks.mat  # sub049 的行为记录与播放顺序
    ├── sub050/After_remarks.mat  # sub050 的行为记录与播放顺序
    ├── sub051/After_remarks.mat  # sub051 的行为记录与播放顺序
    ├── sub052/After_remarks.mat  # sub052 的行为记录与播放顺序
    ├── sub053/After_remarks.mat  # sub053 的行为记录与播放顺序
    ├── sub054/After_remarks.mat  # sub054 的行为记录与播放顺序
    ├── sub055/After_remarks.mat  # sub055 的行为记录与播放顺序
    ├── sub056/After_remarks.mat  # sub056 的行为记录与播放顺序
    ├── sub057/After_remarks.mat  # sub057 的行为记录与播放顺序
    ├── sub058/After_remarks.mat  # sub058 的行为记录与播放顺序
    ├── sub059/After_remarks.mat  # sub059 的行为记录与播放顺序
    ├── sub060/After_remarks.mat  # sub060 的行为记录与播放顺序
    ├── sub061/After_remarks.mat  # sub061 的行为记录与播放顺序
    ├── sub062/After_remarks.mat  # sub062 的行为记录与播放顺序
    ├── sub063/After_remarks.mat  # sub063 的行为记录与播放顺序
    ├── sub064/After_remarks.mat  # sub064 的行为记录与播放顺序
    ├── sub065/After_remarks.mat  # sub065 的行为记录与播放顺序
    ├── sub066/After_remarks.mat  # sub066 的行为记录与播放顺序
    ├── sub067/After_remarks.mat  # sub067 的行为记录与播放顺序
    ├── sub068/After_remarks.mat  # sub068 的行为记录与播放顺序
    ├── sub069/After_remarks.mat  # sub069 的行为记录与播放顺序
    ├── sub070/After_remarks.mat  # sub070 的行为记录与播放顺序
    ├── sub071/After_remarks.mat  # sub071 的行为记录与播放顺序
    ├── sub072/After_remarks.mat  # sub072 的行为记录与播放顺序
    ├── sub073/After_remarks.mat  # sub073 的行为记录与播放顺序
    ├── sub074/After_remarks.mat  # sub074 的行为记录与播放顺序
    ├── sub075/After_remarks.mat  # sub075 的行为记录与播放顺序
    ├── sub076/After_remarks.mat  # sub076 的行为记录与播放顺序
    ├── sub077/After_remarks.mat  # sub077 的行为记录与播放顺序
    ├── sub078/After_remarks.mat  # sub078 的行为记录与播放顺序
    ├── sub079/After_remarks.mat  # sub079 的行为记录与播放顺序
    ├── sub080/After_remarks.mat  # sub080 的行为记录与播放顺序
    ├── sub081/After_remarks.mat  # sub081 的行为记录与播放顺序
    ├── sub082/After_remarks.mat  # sub082 的行为记录与播放顺序
    ├── sub083/After_remarks.mat  # sub083 的行为记录与播放顺序
    ├── sub084/After_remarks.mat  # sub084 的行为记录与播放顺序
    ├── sub085/After_remarks.mat  # sub085 的行为记录与播放顺序
    ├── sub086/After_remarks.mat  # sub086 的行为记录与播放顺序
    ├── sub087/After_remarks.mat  # sub087 的行为记录与播放顺序
    ├── sub088/After_remarks.mat  # sub088 的行为记录与播放顺序
    ├── sub089/After_remarks.mat  # sub089 的行为记录与播放顺序
    ├── sub090/After_remarks.mat  # sub090 的行为记录与播放顺序
    ├── sub091/After_remarks.mat  # sub091 的行为记录与播放顺序
    ├── sub092/After_remarks.mat  # sub092 的行为记录与播放顺序
    ├── sub093/After_remarks.mat  # sub093 的行为记录与播放顺序
    ├── sub094/After_remarks.mat  # sub094 的行为记录与播放顺序
    ├── sub095/After_remarks.mat  # sub095 的行为记录与播放顺序
    ├── sub096/After_remarks.mat  # sub096 的行为记录与播放顺序
    ├── sub097/After_remarks.mat  # sub097 的行为记录与播放顺序
    ├── sub098/After_remarks.mat  # sub098 的行为记录与播放顺序
    ├── sub099/After_remarks.mat  # sub099 的行为记录与播放顺序
    ├── sub100/After_remarks.mat  # sub100 的行为记录与播放顺序
    ├── sub101/After_remarks.mat  # sub101 的行为记录与播放顺序
    ├── sub102/After_remarks.mat  # sub102 的行为记录与播放顺序
    ├── sub103/After_remarks.mat  # sub103 的行为记录与播放顺序
    ├── sub104/After_remarks.mat  # sub104 的行为记录与播放顺序
    ├── sub105/After_remarks.mat  # sub105 的行为记录与播放顺序
    ├── sub106/After_remarks.mat  # sub106 的行为记录与播放顺序
    ├── sub107/After_remarks.mat  # sub107 的行为记录与播放顺序
    ├── sub108/After_remarks.mat  # sub108 的行为记录与播放顺序
    ├── sub109/After_remarks.mat  # sub109 的行为记录与播放顺序
    ├── sub110/After_remarks.mat  # sub110 的行为记录与播放顺序
    ├── sub111/After_remarks.mat  # sub111 的行为记录与播放顺序
    ├── sub112/After_remarks.mat  # sub112 的行为记录与播放顺序
    ├── sub113/After_remarks.mat  # sub113 的行为记录与播放顺序
    ├── sub114/After_remarks.mat  # sub114 的行为记录与播放顺序
    ├── sub115/After_remarks.mat  # sub115 的行为记录与播放顺序
    ├── sub116/After_remarks.mat  # sub116 的行为记录与播放顺序
    ├── sub117/After_remarks.mat  # sub117 的行为记录与播放顺序
    ├── sub118/After_remarks.mat  # sub118 的行为记录与播放顺序
    ├── sub119/After_remarks.mat  # sub119 的行为记录与播放顺序
    ├── sub120/After_remarks.mat  # sub120 的行为记录与播放顺序
    ├── sub121/After_remarks.mat  # sub121 的行为记录与播放顺序
    └── sub122/After_remarks.mat  # sub122 的行为记录与播放顺序
```

## 各文件的作用与阅读位置

### 根目录文件

程序入口、项目说明和学习文档集中在这里。预训练、特征提取和 MLP 分类是三个独立入口。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [CITATION.cff](CITATION.cff) | 机器可读的项目与论文引用信息，供代码托管平台和引用工具读取。 | 说明 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 说明贡献代码时的环境、检查项目和结果报告方式。 | 维护 |
| [CORE_CODE_GUIDE.md](CORE_CODE_GUIDE.md) | 按五个阶段摘录并讲解核心计算，与源码中的 CORE-1 至 CORE-5 阅读标记对应，适合先掌握模型主线。 | 快速入门 |
| [DATA.md](DATA.md) | 说明 EEG、离线语义特征、投影权重和行为元数据的来源及使用边界，并列出发布资产校验信息。 | 说明 |
| [extract_features.py](extract_features.py) | 特征提取入口：逐折读取 EEG 和检查点，归一化输入，调用预测路径取得主干特征，按 trial 平滑并保存；另保留不使用预训练的特征分支。 | 核心 B7 |
| [HYPERAMA_EXPERIMENTS.md](HYPERAMA_EXPERIMENTS.md) | 记录 hyper_distance 与 hyper_emotion 的设计、配置、命令及验证约定；其中服务器路径属于既有环境说明，不能直接视为当前可用路径。 | 说明 |
| [LICENSE](LICENSE) | 代码的 MIT 许可证；数据与其他资产的适用范围另见 DATA.md。 | 说明 |
| [PROJECT_FILE_MAP.md](PROJECT_FILE_MAP.md) | 当前文件目录树及逐文件用途索引，与源码学习计划配套使用。 | 先读 |
| [project_seed_features.py](project_seed_features.py) | SEED 离线投影入口：读取已有文本和图像投影权重，逐个一秒向量进行变换，保存保留窗口内时间维的特征，供 SEED 数据集读取。 | SEED 扩展 |
| [README.md](README.md) | 项目总入口，介绍任务、环境、数据准备和三阶段运行方式，并链接学习文档。 | 说明 |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | 记录原 AMA 的 FACED、SEED 协议、默认设置、源码对照与评估限制，帮助判断实现与实验条件。 | 说明 |
| [requirements-preprocess.txt](requirements-preprocess.txt) | 在核心依赖上补充 Qwen 图像描述等离线预处理所需的可选依赖。 | 环境 |
| [requirements.txt](requirements.txt) | 核心 Python 依赖及固定版本，覆盖 PyTorch、Lightning、Hydra、数据处理、绘图和 geoopt。 | 环境 |
| [SECURITY.md](SECURITY.md) | 说明敏感数据或安全问题的报告方式。 | 维护 |
| [SOURCE_READING_GUIDE.md](SOURCE_READING_GUIDE.md) | 源码学习计划，按大纲、函数调用、逐行原理三轮组织，包含自检题和学习进度。 | 先读 |
| [train_ext.py](train_ext.py) | 预训练入口：合成配置，划分受试者，创建 DataModule、EEG 网络和训练封装，设置日志、检查点、早停并调用 Trainer.fit。 | 核心 B1/B4 |
| [train_mlp.py](train_mlp.py) | 下游分类入口：读取各折特征、划分受试者、拟合标准化、可选 PCA，训练 MLP，恢复最佳检查点并汇总分类指标。 | 核心 B8 |

### cfgs 配置目录

配置按数据、模型、协议和实验覆盖分层组织。阅读时还要查看 Python 入口是否改写配置。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [cfgs/config.yaml](cfgs/config.yaml) | Hydra 主配置：组合数据、模型和训练协议，定义预训练模式、损失、优化器、MLP、特征提取、日志与折范围。 | 核心 B1 |
| [cfgs/data/FACED_def_c2.yaml](cfgs/data/FACED_def_c2.yaml) | FACED 二分类配置；仍使用 28 视频缓存，训练和分类代码再移除中性类并映射为负性、正性标签。 | 二分类扩展 |
| [cfgs/data/FACED.yaml](cfgs/data/FACED.yaml) | FACED 九分类数据配置：EEG 与语义特征路径、32 通道、123 名受试者、28 个视频、窗口参数及冻结对齐权重路径。 | 核心 B1 |
| [cfgs/data/SEED.yaml](cfgs/data/SEED.yaml) | SEED 三分类配置：15 名受试者、3 个 session、62 通道、留一受试者协议，以及离线投影前后特征路径。 | SEED 扩展 |
| [cfgs/experiment/hyper_distance.yaml](cfgs/experiment/hyper_distance.yaml) | 双曲距离实验覆盖：设置实验标识与 hyperbolic 几何，并关闭 emotion 辅助项。 | 核心 B1/B5 |
| [cfgs/experiment/hyper_emotion.yaml](cfgs/experiment/hyper_emotion.yaml) | 情绪辅助实验覆盖：保留双曲几何，打开同情绪多正样本损失，并使用独立实验标识。 | 核心 B5 |
| [cfgs/model/cnn_att_dual.yaml](cfgs/model/cnn_att_dual.yaml) | 当前默认 EEG 模型配置，指向 Conv_att_simple_new，使用局部与全局双分支注意力及残差投影头。 | 核心 B3 |
| [cfgs/model/cnn_att.yaml](cfgs/model/cnn_att.yaml) | 同一 EEG 类的单分支注意力配置，global_att=False，seg_att=7；与默认双分支配置的差异也包含注意力窗口。 | 结构对照 |
| [cfgs/model/cnn_clisa.yaml](cfgs/model/cnn_clisa.yaml) | 保留的 CLISA 基线配置，目标为 ConvNet_complete_baseline_new；当前 model/models.py 未定义该类，不能直接作为可运行主线。 | 历史配置 |
| [cfgs/model/cnn_trans.yaml](cfgs/model/cnn_trans.yaml) | 保留的 Transformer 注意力配置，目标为 Conv_att_transformer；当前 model/models.py 未定义该类。 | 历史配置 |
| [cfgs/protocol/FACED_def_c2.yaml](cfgs/protocol/FACED_def_c2.yaml) | 二分类采用的 FACED 训练协议覆盖，与九分类保持相应训练参数。 | 二分类扩展 |
| [cfgs/protocol/FACED.yaml](cfgs/protocol/FACED.yaml) | 覆盖 FACED 的训练轮数、早停、采样次数、融合温度、探针权重和 MLP 设置。 | 核心 B1 |
| [cfgs/protocol/SEED.yaml](cfgs/protocol/SEED.yaml) | 覆盖 SEED 特有的采样方式、采样次数、融合和 MLP 参数及计算精度；不能直接套用 FACED 解释。 | SEED 扩展 |
| [cfgs/readme_cfg.md](cfgs/readme_cfg.md) | 简要说明各配置目录的用途，以及通过配置文件或命令行设置参数的方法。 | 说明 |

### data 数据目录

这里主要是数据处理源码。目录名 data 不代表其中已经包含 FACED 或 SEED 的原始 EEG 数据。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [data/__init__.py](data/__init__.py) | 数据包入口，目前只有注释，没有主动导出对象或运行数据处理。 | 包结构 |
| [data/data_process.py](data/data_process.py) | 实现 LDS 平滑及历史 running normalization 等数值处理；当前提取主线通过 smooth_trial_features 调用其中 LDS。 | 核心 B7/C13 |
| [data/dataset.py](data/dataset.py) | 定义 FACED、SEED 等 Dataset、样本索引映射、视频语义特征配对及多种采样器；主线重点读 FACED_Dataset_new 与 PretrainSampler。 | 核心 B2 |
| [data/feature_processing.py](data/feature_processing.py) | 提供提取阶段基于训练受试者的归一化、SEED 统计量与分片数据读取，以及不跨 trial 边界的特征平滑封装。 | 核心 B7 |
| [data/io_utils.py](data/io_utils.py) | 负责 EEG 文件读取、数据集格式适配、窗口切片和缓存保存；FACED 使用 pkl，SEED 路径处理逐 trial 的 mat 数据。 | 核心 B2 |
| [data/pl_datamodule.py](data/pl_datamodule.py) | 定义 Lightning DataModule，组织数据准备、训练验证数据集及 DataLoader，并选择批次采样方式。 | 核心 B2 |

### model 模型目录

网络结构、训练行为与损失分开定义。默认 FACED 路径使用 pl_models.py，SEED 使用独立封装。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [model/__init__.py](model/__init__.py) | 导出 EEG 网络与 FACED 训练类，并由 get_extractor_class 根据数据集选择 FACED 或 SEED 训练封装。 | 核心 B1 |
| [model/loss/__init__.py](model/loss/__init__.py) | 空的损失包初始化文件，使该目录作为 Python 包使用。 | 包结构 |
| [model/loss/con_loss.py](model/loss/con_loss.py) | FACED 主线使用其中 MultiModalInfoNCELoss，计算双向配对对齐并按视频、受试者屏蔽假负样本；文件也保留其他损失实验类，需由调用关系判断是否启用。 | 核心 B5 |
| [model/loss/hyperbolic.py](model/loss/hyperbolic.py) | 双曲扩展核心：归一化、庞加莱球映射与距离、情绪正样本关系、多正样本损失，以及将几何与辅助项接入训练类的 Mixin。 | 核心 B5/C10/C11 |
| [model/loss/seed_con_loss.py](model/loss/seed_con_loss.py) | SEED 专用 InfoNCE，使用 content_id 屏蔽相同视频与时间段的非对角配对，并支持共享的双曲分数计算。 | SEED 扩展 |
| [model/metric/__init__.py](model/metric/__init__.py) | 空的指标包初始化文件。 | 包结构 |
| [model/metric/metrics.py](model/metric/metrics.py) | 提供通用 top-k accuracy 计算；FACED 封装保留其导入，当前主要分类指标由 torchmetrics 等计算。 | 辅助 |
| [model/models.py](model/models.py) | 定义 EEG 编码器 Conv_att_simple_new、ResidualAdd、分组归一化与 MLP simpleNN3；决定网络结构和训练、提取两种输出。 | 核心 B3/B8 |
| [model/pl_models.py](model/pl_models.py) | FACED 的 Lightning 封装：初始化语义投影层与探针，执行动态融合、对比训练、验证和预测；还定义共享的下游 MLPModel。 | 核心 B4/B6/B8 |
| [model/seed_pl_models.py](model/seed_pl_models.py) | SEED 专用预训练封装，保留其模态特征、融合分组和内容编号处理，并接入共享双曲几何与辅助损失。 | SEED 扩展 |

### scripts 实验调度目录

负责启动和串联实验；实际训练行为仍由训练入口与模型封装定义。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [scripts/launch_ablation.sh](scripts/launch_ablation.sh) | Linux 启动包装脚本，设置线程数并向 run_ablation.py 传递实验名和 GPU；项目与 Python 路径写死，需要匹配实际环境。 | 运行辅助 |
| [scripts/run_ablation.py](scripts/run_ablation.py) | 消融实验调度器：设置实验与 GPU，顺序执行预训练、提取、MLP，检查特征，并保存命令、配置、源码哈希和运行状态。 | 核心 B1 |

### tests 和 tools 验证目录

测试用于核对具体行为，源码对照工具用于比较外部原始实现。存在测试文件不等于当前环境已经执行并通过全部测试。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [tests/test_hyperbolic.py](tests/test_hyperbolic.py) | 测试双曲映射、主损失掩码、多正样本损失、消融配置差异、梯度、精度和检查点恢复，适合配合损失精读。 | 配套测试 |
| [tests/test_protocols.py](tests/test_protocols.py) | 测试数据集协议、训练验证融合、SEED 索引、训练集统计量、特征格式、trial 平滑边界和视频顺序恢复。 | 配套测试 |
| [tests/test_repository.py](tests/test_repository.py) | 检查源码约定、配置合成、合成数据上的 EEG 特征切窗、必要文档与引用元数据。 | 配套测试 |
| [tools/check_source_parity.py](tools/check_source_parity.py) | 给定原始 FACED 与 SEED 源码目录后，对照保留的损失、梯度、采样、MLP、归一化和平滑计算；属于源码对照工具。 | 验证扩展 |

### utils 辅助工具目录

包括日志回调、旧格式转换、重排、检查与绘图。部分脚本在文件顶层执行工作，应通过阅读调用关系判断是否属于当前主线。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [utils/__init__.py](utils/__init__.py) | 空的工具包初始化文件。 | 包结构 |
| [utils/callbacks.py](utils/callbacks.py) | 定义 WarmupEarlyStopping，在指定预热轮数前跳过早停检查；当前训练入口使用自己的 EarlyStopping 配置，没有启用该类。 | 辅助 |
| [utils/experiment_metrics.py](utils/experiment_metrics.py) | ExperimentMetrics 回调，在验证结束后把折、epoch、global_step 与标量指标追加到 JSONL；跳过 sanity check，并处理非有限数值。 | 核心辅助 |
| [utils/inspect_image_features_1024.py](utils/inspect_image_features_1024.py) | 读取指定 npy，打印形状、类型、局部数值和范围，帮助检查语义特征格式；路径为待替换示例。 | 检查辅助 |
| [utils/move_checkpoints.py](utils/move_checkpoints.py) | 按写死的十个检查点文件名，从源目录移动到目标目录；顶层代码直接执行文件操作，属于历史整理脚本。 | 历史工具 |
| [utils/preprocess.py](utils/preprocess.py) | 将旧 FACED pkl 按通道拼接成 MATLAB 数据格式，命令行接收输入、输出路径；属于格式转换工具。 | 离线辅助 |
| [utils/reorder_vids.py](utils/reorder_vids.py) | 读取 After_remarks 中的视频播放顺序，提供重排及恢复函数；当前特征提取中的对应 running normalization 块已注释，测试仍验证重排逻辑。 | 辅助 |
| [utils/utils.py](utils/utils.py) | 通用日志设置、随机种子辅助函数与混淆矩阵计算；其中种子函数接收环境对象，不是当前 Lightning 训练入口的统一初始化函数。 | 辅助 |
| [utils/vis_feature_space.py](utils/vis_feature_space.py) | 对文本与图像特征做 t-SNE，分别按模态及情绪着色并保存图像；输入路径需与本地特征对应。 | 可视化 |
| [utils/vis_text_compare.py](utils/vis_text_compare.py) | 在 t-SNE 空间比较原始文本、对齐后文本和图像，并展示文本变化轨迹。 | 可视化 |

### FACED 离线语义与特征准备目录

这里的脚本准备输入特征。Qwen 生成文本描述、CLIP 图像编码以及已有特征切窗是不同步骤；不能仅凭文件名将其视为完整的文本编码流水线。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [FACED/FACED_get_image_feature_clip_vit.py](FACED/FACED_get_image_feature_clip_vit.py) | 使用 CLIP 视觉模型提取裁剪帧的隐藏特征，进行 PCA 降维并保存滑窗图像特征；输入是图像帧。 | 离线预处理 |
| [FACED/FACED_Qwen_nolabel.py](FACED/FACED_Qwen_nolabel.py) | 使用 Qwen2-VL 为视频帧生成客观文本描述，清理文本并保存 JSON；它生成描述，不直接输出最终 1024 维文本向量。 | 离线预处理 |
| [FACED/move_window_timelen5_tiemstep2_photo.py](FACED/move_window_timelen5_tiemstep2_photo.py) | 将已有逐秒 1024 维图像特征对齐到最后 30 秒，再切为 5 秒窗口、2 秒步长，保存默认形状为 13×5×1024 的数组；文件名保留原有 tiemstep 拼写。 | 离线辅助 |
| [FACED/move_window_timelen5_timestep2_EEG.py](FACED/move_window_timelen5_timestep2_EEG.py) | 对已提取的逐秒 EEG 特征做滑窗，默认特征维为 160，提供 process_eeg 和命令行入口；与主线原始 32 通道 EEG 的读取切窗不同。 | 离线辅助 |
| [FACED/move_window_timelen5_timestep2_text.py](FACED/move_window_timelen5_timestep2_text.py) | 对已有逐秒文本向量做末尾 30 秒对齐与滑窗，保存默认 13×5×1024 数组；此文件不执行 CLIP 文本编码。 | 离线辅助 |
| [FACED/README.md](FACED/README.md) | 说明 FACED 离线预处理脚本、可选依赖，以及 EEG 特征切窗工具的参数使用。 | 离线说明 |

### data_preprocess MATLAB 参考目录

当前 Python 主线不依赖 MATLAB。这里保留的独立预处理与当前 SEED 输入协议不同。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [data_preprocess/AutoICA_SEED.m](data_preprocess/AutoICA_SEED.m) | 独立 MATLAB 预处理函数，进行重采样、滤波、坏道处理、ICA 去伪迹、重参考和保存；当前 Python 主线不调用它，所需自定义插值助手未随当前副本提供。 | 历史预处理 |
| [data_preprocess/nt_find_bad_channels_custom.m](data_preprocess/nt_find_bad_channels_custom.m) | 根据信号超过相对或绝对阈值的时间比例检测坏通道，返回坏道索引及保留好通道的映射。 | 历史预处理 |
| [data_preprocess/README.md](data_preprocess/README.md) | 说明 MATLAB 工具的参考性质、58 通道输出与当前 SEED Python 62 通道输入的差异，以及单独使用时所需依赖。 | 历史说明 |

### multimodel_fusion 语义对齐目录

训练文本与图像的对齐投影头，以及保存分析图。该离线对齐训练与 EEG 对比预训练分属两个阶段。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [multimodel_fusion/fusion10/figures/01_before_by_modality.png](multimodel_fusion/fusion10/figures/01_before_by_modality.png) | 保存的对齐前文本、图像分布图，按模态区分，用于观察原始模态差异。 | 图像资产 |
| [multimodel_fusion/fusion10/figures/02_after_by_modality.png](multimodel_fusion/fusion10/figures/02_after_by_modality.png) | 保存的对齐后文本、图像分布图，按模态区分，用于与对齐前图比较。 | 图像资产 |
| [multimodel_fusion/fusion10/figures/03_text_shift.png](multimodel_fusion/fusion10/figures/03_text_shift.png) | 保存的文本向量在降维空间中的变化轨迹图，用于观察对齐前后位置变化。 | 图像资产 |
| [multimodel_fusion/fusion10/figures/04_before_by_emotion.png](multimodel_fusion/fusion10/figures/04_before_by_emotion.png) | 保存的对齐前语义特征分布图，按九种情绪着色。 | 图像资产 |
| [multimodel_fusion/fusion10/figures/05_after_by_emotion.png](multimodel_fusion/fusion10/figures/05_after_by_emotion.png) | 保存的对齐后语义特征分布图，按九种情绪着色。 | 图像资产 |
| [multimodel_fusion/fusion10/README.md](multimodel_fusion/fusion10/README.md) | 解释 FACED 对齐资产及文本、图像投影权重的校验信息；其中提及的 pt 权重当前本地副本未包含。 | 离线说明 |
| [multimodel_fusion/fusion10/train_text_image_alignment.py](multimodel_fusion/fusion10/train_text_image_alignment.py) | 离线训练 FACED 文本、图像残差投影头，使用双向 InfoNCE，保存 projector 权重及投影数组；它不训练 EEG 编码器。 | 离线训练 |
| [multimodel_fusion/seed_fusion1/README.md](multimodel_fusion/seed_fusion1/README.md) | 说明 SEED 的离线对齐权重和先投影再窗口平均的准备流程；当前副本仅含说明与训练脚本，未含所列 pt 权重。 | SEED 说明 |
| [multimodel_fusion/seed_fusion1/train_text_image_alignment.py](multimodel_fusion/seed_fusion1/train_text_image_alignment.py) | SEED 语义离线对齐训练，使用双向 InfoNCE 保存两个投影头及按情绪子目录组织的投影数组。 | SEED 离线训练 |
| [multimodel_fusion/visualize_alignment.py](multimodel_fusion/visualize_alignment.py) | 读取对齐前后的语义特征，计算 t-SNE、轮廓系数和跨模态相似度等，输出五类对比图；实际输入、输出目录由脚本顶部设置决定。 | 可视化 |

### 工程配置文件

这些文件服务于版本管理和自动检查，不定义模型的数学计算。

| 文件 | 作用与使用关系 | 阅读位置 |
| --- | --- | --- |
| [.gitattributes](.gitattributes) | 规定文本换行符及 MATLAB 文件的二进制属性。 | 维护 |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | GitHub Actions 检查流程：安装核心依赖、检查 Python 语法、运行无真实数据测试，并解析三种数据配置。 | 维护 |
| [.gitignore](.gitignore) | 排除本地数据、模型权重、日志、检查点、缓存和生成特征等文件，避免将运行产物混入源码。 | 维护 |

### After_remarks 行为元数据目录

目录树中 sub000 至 sub122 的每个 After_remarks.mat 分别属于对应编码受试者，共 123 个，作用相同但内容按受试者区分。每份文件记录 28 个 trial，字段包含 score、trial、vid、Accuracy、ResponseTime；字段说明见 [DATA.md](DATA.md)。

[utils/reorder_vids.py](utils/reorder_vids.py) 从 After_remark 结构读取视频编号，构造受试者实际观看顺序，再用于特征重排和顺序恢复。这些文件是行为记录，不是 EEG 波形、Python 代码或模型权重。

当前 [extract_features.py](extract_features.py) 中读取观看顺序并进行 running normalization 的代码块已注释；[tests/test_protocols.py](tests/test_protocols.py) 仍验证相关重排逻辑。学习时先理解用途及读取函数，无需逐个打开 123 份数值文件。

## 配置引用但当前副本未包含的内容

以下内容没有列入实际目录树。既有 README 中提及某项资产，不代表当前磁盘上已有对应文件。

| 内容 | 与代码的关系 |
| --- | --- |
| fusion10 下的 projector_text.pt、projector_image.pt | FACED 数据配置引用的两个冻结语义对齐权重；本地 fusion10 当前只有说明、训练脚本和五张图片。 |
| seed_fusion1 下的 projector_text.pt、projector_image.pt | project_seed_features.py 使用的 SEED 离线投影权重；本地当前未包含。 |
| datasets/FACED、data/SEED、features 等数据目录 | 对应 EEG 或语义特征输入；当前副本未包含，具体位置由配置决定。 |
| seed_fusion1/projected_text、projected_image | SEED 离线投影产生的输入数组目录；当前副本未包含。 |
| 检查点、提取特征、实验日志 | 由训练、提取、分类流程产生的运行文件，当前目录树中未包含。 |
| nt_interpolate_bad_channels_custom.m 及 MATLAB 通道辅助文件 | 仅独立 MATLAB 参考工具需要；不属于当前 Python 主线的必需步骤。 |

## 初读时最需要区分的文件关系

1. **models.py 与 pl_models.py：**前者定义网络怎样计算，后者定义 FACED 一个 batch 怎样训练、验证和预测；SEED 逻辑在 seed_pl_models.py。
2. **con_loss.py 与 hyperbolic.py：**前者处理主对齐配对、屏蔽和交叉熵，后者提供双曲分数和可选情绪项，并通过 Mixin 接入训练。
3. **离线对齐训练与 EEG 预训练：**multimodel_fusion 中的训练脚本学习文本图像投影；train_ext.py 学习 EEG 与融合语义的对齐。
4. **两类 EEG 切窗：**data/io_utils.py 处理主线 EEG 数据；FACED/move_window_timelen5_timestep2_EEG.py 处理已经提取好的逐秒特征。
5. **模型配置与可用实现：**cnn_clisa.yaml、cnn_trans.yaml 的目标类当前缺失；cnn_att.yaml 和 cnn_att_dual.yaml 指向实际存在的 Conv_att_simple_new。
6. **分析图片与分类结果：**fusion10/figures 下的五张图展示语义对齐空间，不是 HyperAMA 下游 EEG 分类准确率的证明。

阅读主线为：配置 → train_ext.py → data → model/models.py → model/pl_models.py → model/loss → extract_features.py → train_mlp.py。其余文件按上表的“阅读位置”补充。
