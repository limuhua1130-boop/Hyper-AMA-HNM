# HyperAMA 加入 HNM：最小改动版运行说明

日期：2026-10-11。规范号：`hyperama_hnm_constant_cone_v4`。

本版以 `HyperAMA-EEG-main` 的原始备份为基准，在双曲损失接口接入材料到 EEG 的蕴含锥和 HNM 原论文公式（8）—（9）的对比目标。径向间隔惩罚和距离过远惩罚已删除；本次进一步删除锥损失预热，从第一轮起使用恒定锥权重 `0.01`。半径头、半径范围和数值保护保持不变。这是 HyperAMA 与 HNM 的融合适配，不是完整复现 HNM 编码器及训练协议。新增部分及各项取舍见根目录的 `HNM融合说明.md`。此次完成的是代码接入和工程验证，没有运行真实数据训练或同步服务器。

## 1. 改动范围与备份

唯一改动的原文件是 `model/loss/hyperbolic.py`，只有两个接入位置：

1. 开启 `train.hnm.enabled` 时，使用 `HNMGeometry` 和新增的 `HNMPaperContrastiveLoss`；未开启时继续使用原 `ContrastiveGeometry`、InfoNCE 和原假阴性掩码。
2. 在原 `_scale_losses` 接口追加 HNM 辅助损失和诊断日志。

新增算法文件只有 `model/loss/hnm.py`；新增运行配置只有 `cfgs/experiment/hnm_h2.yaml`。其他新增文件是本说明、检查报告、测试及测试输出。

原 `train_ext.py`、`extract_features.py`、`model/__init__.py` 和 `README.md` 已从修改前备份恢复。原材料投影与融合、EEG 编码器、数据格式、采样器、训练及验证步骤、优化器、调度器、检查点选择、特征提取、LDS 和下游 MLP 均复用原文件。原 InfoNCE 和掩码文件也未修改，仍供原 Hyper 基线使用；只有启用 HNM 时替换对比损失对象。原有 YAML 和依赖列表没有修改。本次改动 HNM 算法、HNM YAML、对应测试及本说明，新增根目录说明，没有再修改任何原 Hyper 文件。

上一实现中的材料窗口去重、七字段数据封装、独立 HNM 模型类、自定义距离函数、重写的对比损失、训练与验证统一改写、跨视频情绪增强、H0/H1/H3 配置和新运行器均已撤回。旧的 `scripts/run_hnm.py` 命令不再适用。

保留以下完整备份，均有同目录 `manifest.json` 记录逐文件 SHA-256：

```text
原 Hyper，加入 HNM 前：
D:\pytorch_study\Guolab\XiongIdea\backups\HyperAMA-before-HNM-20261010-221614\HyperAMA-EEG-main.zip

本次收缩前的上一 HNM 实现：
D:\pytorch_study\Guolab\XiongIdea\backups\HyperAMA-before-minimal-HNM-20261010-224709\HyperAMA-EEG-main.zip

本次替换对比损失前（含极小量保护）：
D:\pytorch_study\Guolab\XiongIdea\backups\HyperAMA-before-paper-alignment-20261011-002211\HyperAMA-EEG-main.zip

本次删除距离惩罚前：
D:\pytorch_study\Guolab\XiongIdea\backups\HyperAMA-before-remove-distance-guards-20261011-004758\HyperAMA-EEG-main.zip

本次删除锥损失预热前：
D:\pytorch_study\Guolab\XiongIdea\backups\HyperAMA-before-remove-cone-warmup-20261011-005301\HyperAMA-EEG-main.zip
```

备份覆盖各时间点 Hyper 目录内的文件，不包含目录外 EEG 数据、特征缓存和 Python 环境。需要查看旧版时解压到单独目录。本版检查点带有 HNM 配置校验，应使用本版代码和训练时相同的配置提取特征；上一 HNM 实现的检查点不视为兼容。

## 2. 原超参数及新增参数报告

没有修改原 Hyper 基线配置值。FACED 默认仍为固定 `c=0.1`、对齐维度 `1024`、随机种子 `7`、Adam 学习率 `0.0007`、weight decay `0.00015`、最大/最小轮数 `80/5`、早停 patience `5`、置信度温度 `0.1`、探针权重 `2`、`w_clip=1`。SEED 继续继承自身的数据配置和原协议。HNM 对比温度沿用上一版：以 `0.07` 初始化，采用 `tau = exp(log_tau).clamp_min(0.01)`，依据论文第 3.2 节；原 Hyper 基线仍使用固定温度。**本次参数变更报告：删除 `cone.zero_epochs=3` 和 `cone.ramp_epochs=5`，`cone.weight=0.01` 数值不变，从第一轮起生效；其余超参数数值均未修改。两项距离惩罚的 `guard` 配置在上一版已删除。**

HNM 配置位于 `train.hnm`，运行标识仍为 `log.run=212`、`log.exp_name=hnm_h2`。检查点和提取特征目录随 `log.run` 变化；最终指标文件使用 `runs/${log.exp_name}/results.json`，因此不同实验必须同时区分运行编号和实验名称。下表为当前保留的参数，数值全部沿用上一版，没有调整或效果调参：

| 新参数 | 默认值 | 用途 |
| --- | --- | --- |
| `radius.cone_k` | 0.1 | 锥张角常数 K |
| `radius.aperture_max_deg` | 30 | 最大锥半张角，单位为度 |
| `radius.maximum` | 2.5 | 原点测地半径上界 |
| `radius.initial` | 2.0 | 两个分支的初始测地半径 |
| `radius.minimum_direction_norm` | 1e-8 | 方向非退化检查阈值 |
| `cone.weight` | 0.01 | 第一轮起生效的恒定锥权重 |
| `cone.tangent_epsilon` | 1e-10 | 锥角横向分量的平滑极小量，避免重合时数值退化 |
| `contrastive.minimum_temperature` | 0.01 | HNM 论文第 3.2 节的温度下限 |

半径下界由 `asinh(2K/sin(30°))/sqrt(c)` 推出，约为 `1.233400`，不是另设参数。旧设计提案的初始半径 `0.8` 在当前曲率和锥角下不可行，因此上一 H2 实现已改用 `2.0`，本版保持该值。原 Hyper 对单位方向执行 `expmap0` 的原点测地半径恰为 `2.0`，因此初始双曲位置与原 Hyper 一致。

**保留的项目设计及依据：**

- 半径头提供原 Hyper 固定半径映射缺少的径向自由度；选择两个 `Linear(1024,1)` 是复用原编码器的适配方式，不是 HNM 原网络结构。初始半径 `2.0` 用于保持原 Hyper 初始位置。
- 最大半角 `30°` 用于避免锥过宽并远离 `asin` 的导数奇异边界；其下界约 `1.2334` 比公式定义域所需的约 `0.6283` 更严格。最大半径 `2.5` 用于限制径向逃逸。30° 和 2.5 均不是论文规定，也未经效果验证。
- 极小量保护仍保留，属于本项目数值处理，不能标为已核实的 HNM 官方实现。锥损失预热已删除：论文公式（13）为 `L_con + lambda * L_ent`，第 3.2 节给出 `lambda=0.01`，全文未找到锥权重预热或渐增的说明。本版按这一固定权重从第一轮计算锥项。
- 编码器、固定曲率、维度、优化器和其他训练协议继续使用 HyperAMA。

原 Hyper 的 `contrastive.scales.emotion` 保留原实现且默认关闭，没有加入新的跨视频情绪损失。

## 3. 数据流与损失接入

FACED 仍接收原六字段 batch：`(EEG, label, text, image, video_id, subject_id)`。典型 EEG 形状为 `[B,1,32,625]`，预提取文本为 `[B,5,1024]`、图像为 `[B,1024]`。文本序列取均值、冻结 Align、在线投影、置信度加权融合等步骤均由原模型执行。

新增模块只接收原流程已输出的 `[B,1024]` EEG 表示和 `[B,1024]` 材料表示。每一行材料是对应 EEG 的父节点。保持原来的批内材料计算，不做窗口去重；同一视频在不同样本中不保证拥有完全相同的父节点坐标。本版实现的是配对的“材料父节点 → EEG 子节点”两层约束，不是额外构造全局共享材料树。

对每个分支，复用原单位归一化得到方向 `u`，新增一个 `Linear(1024,1)`：

```text
u = normalize(h)
rho = rho_min + (rho_max - rho_min) * sigmoid(W u + b)
z = 原 geoopt.PoincareBall.expmap0((rho / 2) * u)
```

两个半径头独立，共 2050 个可训练参数；本次对比温度另增加 1 个参数，合计比原 Hyper 增加 2051 个可训练参数，均由原 `Adam(self.parameters())` 自动管理。初始化不推进原模型随机数状态。半径头统一读取单位方向，使对比损失与辅助损失使用同一映射。输出仍为 `[B,1024]`，没有拼接额外坐标，也没有额外方向投影。

启用 HNM 时，对比目标严格按 [HNM 原论文](https://arxiv.org/pdf/2605.09392) 公式（8）—（9）的印刷形式实现，距离继续调用原 `geoopt` 的 `ball.dist`：

```text
S[i,k] = -d_H(EEG_i, material_k) / tau
L_E2M = sum_i(logsumexp_{k != i}(S[i,k]) - S[i,i])
L_M2E = sum_i(logsumexp_{k != i}(S[k,i]) - S[i,i])
L_con_paper = (L_E2M + L_M2E) / 2
```

正样本只用于分子（等价写法中的 `-S[i,i]`）；分母只排除对角正样本，其他批内样本全部保留。不使用原 `vid_sub`、文本阈值或 SEED `content_id` 掩码，仍接受这些元数据以复用原调用接口。同视频/同内容的其他样本因此也作为负样本，这是按论文目标实现后的实际行为。批内按求和计算，不能替换成原默认交叉熵的均值；采用 `logsumexp` 避免直接指数计算下溢。该损失允许为负，不做截断。

有效 batch 必须至少包含 2 对样本；否则论文分母为空，当前明确报错，不悄悄返回零或更改原采样器。FACED 二分类过滤后也受此条件约束。由于从批均值改为批求和，即使其余权重不变，对比项相对探针和辅助项的尺度也已改变；本次没有擅自重缩放其他项。尚未取得可核实的作者实现，本版复现的是明确印出的公式及温度设置。

删除两项距离惩罚后，HNM 辅助目标只保留锥项：

```text
L_cone = mean(relu(theta(material, EEG) - psi(material)))
L_HNM  = 0.01 * L_cone
```

锥角使用 FP64 的 `atan2` 计算，保留方向和半径梯度；映射和原距离计算保持 FP32。数值防护沿用 `cone.tangent_epsilon=1e-10`，将其由原先的退化报错阈值改为平滑极小量：若原切向分量为 `along`、`across`，则计算 `theta = atan2(hypot(across, epsilon), along)`，其中 `hypot(across, epsilon) = sqrt(across**2 + epsilon**2)`。有限的重合输入不再触发退化异常；理想重合时角度取数值近似 `pi/2`，仍有正的锥惩罚和推动径向分离的梯度。非有限输入和非法锥定义域仍报错。此处是本实现的数值近似，不是已核实的 HNM 官方处理，也不构成父子最小距离保证；近重合梯度仍可能较大，继续使用原训练入口已有的梯度裁剪。没有修改 epsilon 数值或其他超参数。

原训练、验证函数没有重写。动态融合模式下，新增项通过原辅助损失接口进入总目标：

```text
训练：w_clip * (L_con_paper + probe_loss_weight * L_probes
               + L_Hyper_optional + L_HNM)
验证：w_clip * (L_con_paper + L_Hyper_optional + L_HNM)
```

原验证不将探针 CE 加入总损失，本版保持这一行为。原训练与验证的材料归一化顺序及 alpha 聚合方式也保持各自实现，不为 HNM 统一它们。

与上一 HNM 实现相比，HNM 项现在随原辅助损失一起乘以 `w_clip`，以避免改写原训练、验证步骤；原默认 `w_clip=1` 时权重数值没有变化。

从第 1 轮起，训练和验证的锥权重始终为 `0.01`，不再受轮数影响。`hnm_loss` 仍接收原接口传入的 `epoch`，但不使用它计算权重，以保持原 Hyper 接口文件不变。原早停和选点规则保持不变。

## 4. 运行

在 `D:\pytorch_study\Guolab\XiongIdea\HyperAMA-EEG-main` 目录内，使用原 Hyper 可用的 Python 环境。GPU 训练仍需要原环境的 CUDA PyTorch，本次独立 CPU 测试环境仅用于工程验证。没有新增第三方依赖。

默认路径见 `cfgs/data/FACED.yaml`。FACED 需要原 EEG 数据、28 个文本特征文件、28 个图像特征文件和原冻结 Align 权重。本机现有材料文件每视频分别为 `[13,5,1024]` 和 `[13,1024]`，配对和切片由原加载器执行。

先查看配置，不启动训练：

```bash
python train_ext.py data=FACED +experiment=hnm_h2 --cfg job --resolve
```

数据位于原配置路径时，依次执行原三个入口：

```bash
python train_ext.py data=FACED +experiment=hnm_h2
python extract_features.py data=FACED +experiment=hnm_h2
python train_mlp.py data=FACED +experiment=hnm_h2
```

本机数据实际位于相邻 `AmaProject`，可通过 PowerShell 参数数组显式传路径，无须改动原配置文件：

```powershell
Set-Location 'D:\pytorch_study\Guolab\XiongIdea\HyperAMA-EEG-main'
$hnmArgs = @(
  'data=FACED'
  '+experiment=hnm_h2'
  'data.data_dir=../AmaProject/data/FACED'
  'data.text_feat_dir=../AmaProject/features/text_timelen5_timestep2_1024_objective'
  'data.image_feat_dir=../AmaProject/features/image_features_clip_vit_centercrop_timelen5_timestep2'
  'data.pretrained_text_proj=../AmaProject/source/multimodel_fusion/fusion10/projector_text.pt'
  'data.pretrained_image_proj=../AmaProject/source/multimodel_fusion/fusion10/projector_image.pt'
)
python train_ext.py @hnmArgs
if ($LASTEXITCODE -ne 0) { throw 'Pretraining failed' }
python extract_features.py @hnmArgs
if ($LASTEXITCODE -ne 0) { throw 'Feature extraction failed' }
python train_mlp.py @hnmArgs
if ($LASTEXITCODE -ne 0) { throw 'MLP training failed' }
```

以上数组仅显式调整文件路径，训练超参数保持原值。三阶段必须使用相同的 `log.run`、`log.exp_name`、数据集和 HNM 配置。若要运行 FACED 二分类，在执行三个入口之前，将数组中的 `data=FACED` 改为 `data=FACED_def_c2`，并在数组内追加以下两个参数：

```powershell
  'log.run=222'
  'log.exp_name=hnm_h2_faced2_r222'
```

此时检查点和提取特征分别进入 `daest_cp/FACED/run222`、`extracted_features/FACED/run222`，最终指标进入 `runs/hnm_h2_faced2_r222/results.json`。只更换 `log.run` 不会改变默认 `log.exp_name=hnm_h2`，仍会覆盖 `runs/hnm_h2/results.json`。已有结果时，应在运行前同时选择未使用的运行编号和实验名称；这些是输出标识，不是训练超参数。

SEED 使用原 SEED 类和数据协议，切换为 `data=SEED` 并提供对应数据，同时在三个入口中使用独立的输出标识，例如 `log.run=232 log.exp_name=hnm_h2_seed_r232`，使最终指标进入 `runs/hnm_h2_seed_r232/results.json`；已检查原接口兼容和合成训练/验证，未完成真实 SEED 效果实验。不能把 FACED 的输入路径用于 SEED。

原 Hyper 可继续用 `+experiment=hyper_distance` 运行；没有 `train.hnm.enabled=true` 时不会构造半径头或追加 HNM 项。不要同时混用两份 experiment 配置。

本次规范号更新为 `hyperama_hnm_constant_cone_v4`，用于区分锥权重从第一轮恒定生效的目标。新检查点仍包含学习到的对比温度及其配置。旧 v1、v2、`hyperama_hnm_no_distance_guards_v3` 检查点不能直接续训或混入本版提取流程，配置校验会拒绝；请使用对应旧代码，或为本版重新训练。若 run212 已有旧结果，运行前须同时换用未使用的 `log.run` 和 `log.exp_name`；默认路径和原串联脚本没有改动。

检查点默认在 `daest_cp/FACED/run212`，提取特征在 `extracted_features/FACED/run212`；具体文件名沿用原入口。提取阶段仍输出原 backbone 的 1024 维 EEG 特征，HNM 半径头只参与预训练几何约束，不改变 MLP 输入维度。原入口对 residual projector、FACED backbone LayerNorm 及提取时 stratified norm 的处理保持原样。

## 5. 验证与问题记录

本地工程测试覆盖原回归测试、配置值对比、FACED 九/二分类及 SEED 的原训练和验证复用、材料输出逐元素一致性、原 Hyper InfoNCE 掩码回归、初始映射、锥角独立公式及梯度、检查点加载，以及真实 FACED backbone 在合成 `[6,1,32,625]` 输入上的一次前向/反向。保留对论文公式值及梯度、批求和、正样本排除、全批负样本、可学习温度及下限、极端分数稳定性、单样本报错、各预训练模式的 HNM 路由和温度检查点恢复的验证。距离惩罚删除检查继续保留；本次改为验证第一轮即有锥损失、不同轮数具有相同锥权重和损失值、旧版检查点被拒绝，并在不覆盖预热参数的情况下执行原训练接口及 backbone 前向/反向。

2026-10-11 删除锥损失预热后的检查：43 项 CPU 测试全部通过（11.499 秒），包含第一轮恒定锥权重、原接口兼容及真实 FACED backbone 的合成输入前向/反向。对比目标、锥角和半径映射函数与本次修改前备份逐字比较一致，其余配置值也一致。本次修改 HNM 算法、配置、测试和本说明，新增根目录 `HNM融合说明.md`。原 Hyper 的 208 个文件中，仍只有此前已改动的 `model/loss/hyperbolic.py` 接口文件不同，其余 207 个字节一致；本次没有再修改原 Hyper 文件。GPU 及真实数据训练未运行。

同目录 `HNM实现检查报告.json` 和 `HNM测试输出.txt` 保留初次接入时的历史检查结果，不包含后续数值防护、对比目标修订、距离惩罚及预热删除。当前测试另覆盖重合、近重合的前向与反向、重合后的半径更新，以及非有限输入仍被拒绝。这些检查不代表模型有效性或正式训练成功。本机可重复工程测试：

```powershell
$env:TEMP='D:\pytorch_study\Guolab\XiongIdea\tools\hnm-test-temp'
$env:TMP=$env:TEMP
& '..\tools\.hnm-test-env\Scripts\python.exe' -B -m unittest discover -s tests -v
```

仍需解决的问题保留记录：

- **Q1：学到材料语义/身份而非情绪。** 仍未解决。保持原 AMA 对齐会保留材料身份捷径的可能；锥和层级本身不能证明模型识别的是情绪。本次按最小改动要求撤回额外跨视频情绪损失，不将该问题标记为已解决。
- **Q2：父子距离过远。** 仍需关注。自行添加的 `1.5` 配对距离软阈值及 `0.1` 径向间隔惩罚均已删除。当前有界半径仍阻止无限径向逃逸：每点原点测地半径不超过 2.5，由三角不等式两点距离不超过 5（数值误差除外）；这不是 HNM 论文的距离阈值，也不能保证配对距离合适。保留对比与锥目标，距离分布和表示质量仍未通过真实数据验证。
- **原评估语义。** 原验证动态融合使用验证标签，检查点仍按原 `val/ZeroShot_Acc` 选取，不能将这个指标解释为完全无标签的零样本能力。本版没有修改该协议。

新增日志位于 `ext/train/hnm_*` 和 `ext/val/hnm_*`，包括锥权重、锥项、加权总项、材料/EEG 平均半径及配对平均距离。间隔项和过远项日志已删除；配对平均距离仅作诊断记录，不构成额外惩罚。日志用于检查新增约束，不新增训练阶段。
