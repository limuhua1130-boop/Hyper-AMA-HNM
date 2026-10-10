# HyperAMA EEG 五段核心代码讲解

先掌握五件事：EEG 怎么变成向量、文本图像怎样融合、双曲距离怎样参与训练、情绪辅助项增加了什么、最终怎样分类。阅读主线是 FACED 九分类、动态融合，先看 hyper_distance，再理解 hyper_emotion。

源码已用 CORE-1 至 CORE-5 标记对应位置。带 B、C 的标记是同一阶段的后续计算。下面只摘核心语句，省略处用注释说明；完整调用与所有分支仍以源码为准。

| 阶段 | 要弄懂的问题 | 源码定位 |
| --- | --- | --- |
| 1 | EEG 怎样变成可比较的特征 | [models.py](model/models.py) 177 行，CORE-1 |
| 2 | EEG 要向什么语义目标学习 | [pl_models.py](model/pl_models.py) 294 行，CORE-2 |
| 3 | 双曲几何怎样改变对齐训练 | [hyperbolic.py](model/loss/hyperbolic.py) 31 行；[con_loss.py](model/loss/con_loss.py) 259、303 行，CORE-3 系列 |
| 4 | 同情绪多正样本如何起作用 | [hyperbolic.py](model/loss/hyperbolic.py) 54 行；[pl_models.py](model/pl_models.py) 365 行，CORE-4 系列 |
| 5 | 预训练好的模型怎样用于分类 | [models.py](model/models.py) 263 行；[pl_models.py](model/pl_models.py) 647 行，CORE-5 系列 |

## 第一阶段 EEG 变成特征向量

先看 [Conv_att_simple_new.forward](model/models.py)。它把一个 EEG 窗口编码成向量，供后续比较或分类。

```python
out = self.timeConv(input)
# 省略 out1、out2、out3、out4 的四个时空卷积分支计算。
out = torch.cat((out1, out2, out3, out4), 1)
# 省略局部与全局注意力、池化和后续卷积。
out = self.global_pool(out)
out_flat = out.flatten(start_dim=1)
```

输入形状是 `[B, 1, 32, 625]`：B 个窗口，每个窗口 32 个通道、每通道 625 个采样点。时间卷积和四个时空分支提取不同时间范围的变化，注意力对特征加权，全局池化将时间维压缩，最后得到 `[B, 1024]` 的主干特征。

`torch.cat(..., 1)` 把四个分支沿特征通道拼起来；`flatten(start_dim=1)` 保留 batch 维，将每个样本的输出整理成一行向量。1024 是特征维数，不是情绪类别数。

**本阶段记住：编码器学习把 EEG 波形表示成一个向量。**预训练时还会经过 EEG 投影头，这个出口与分类阶段的出口在第五阶段对照。

## 第二阶段 动态生成语义学习目标

看 [ExtractorModel.training_step](model/pl_models.py) 的 CORE-2 附近。这一段回答“EEG 应该靠近哪个向量”。

文本、图像的离线特征先经过冻结对齐层，再经过可训练语义投影层，得到 `t_proj` 和 `i_proj`。两个分类探针读取原始语义特征，分别产生文本、图像的情绪预测分数。

```python
with torch.no_grad():
    loss_func_none = nn.CrossEntropyLoss(reduction='none')
    loss_t_sample = loss_func_none(t_logits, labels)
    loss_i_sample = loss_func_none(i_logits, labels)
    diff = (loss_i_sample - loss_t_sample) / self.conf_temp
    inst_alpha = torch.sigmoid(diff).unsqueeze(1)
```

`loss_t_sample` 和 `loss_i_sample` 衡量两种模态对真实情绪标签的预测误差。文本误差更小时，`diff` 为正，文本权重更大；图像误差更小时，文本权重更小。

数学关系为：

\[
\alpha_i=\sigma\left(\frac{\ell_{i,\mathrm{image}}-\ell_{i,\mathrm{text}}}{T_{\mathrm{conf}}}\right).
\]

例如文本损失 0.2、图像损失 0.4、融合温度 0.1，得到的文本权重约为 0.881。这是教学例子；实际 FACED 训练还会将同一视频的样本权重取平均。

得到 `alpha_txt` 后，融合目标由以下两行产生：

```python
fused_target = alpha_txt * t_proj + (1.0 - alpha_txt) * i_proj
fused_target = F.normalize(fused_target, dim=1)
```

`alpha_txt` 是 `[B, 1]`，投影特征是 `[B, 1024]`，广播使每个样本的整条向量共享一个文本权重。融合后重新归一化，作为 EEG 的语义目标。

权重计算处的 `no_grad()` 切断这条权重计算路径的梯度。探针由各自的分类损失训练；语义投影层仍会通过后面的对齐损失训练。因此，融合目标会随训练改变。

**本阶段记住：当前样本上哪种模态的情绪预测误差更小，就更相信哪种模态。**这里的融合温度与下一阶段的对比温度控制不同计算。

## 第三阶段 用双曲距离训练 EEG 与语义的对应关系

先看 [ContrastiveGeometry.project 和 forward](model/loss/hyperbolic.py)。

```python
# project 内的双曲分支：先单位归一化，再映射。
features = self._normalize(features)
if self.geometry == "hyperbolic":
    return self.ball.expmap0(features)

# forward 内：两侧都完成映射后，计算所有 EEG 与目标的两两分数。
eeg, targets = self.project(eeg), self.project(targets)
# 省略维度检查及分支判断。
return -self.ball.dist(eeg[:, None, :], targets[None, :, :])
```

`expmap0` 将向量映射到庞加莱球。`eeg[:, None, :]` 与 `targets[None, :, :]` 利用广播形成两两组合，输出 `[B, B]` 的分数矩阵。第 i 行第 j 列表示 EEG i 与语义目标 j 的匹配程度。

这里取负距离：距离越小，分数越大。距离没有平方；曲率参数默认固定为 `c=0.1`，空间曲率为 `-c`。几何计算使用 FP32。

随后进入 [MultiModalInfoNCELoss.forward](model/loss/con_loss.py)：

```python
logits = self.geometry(preds, targets) / self.temperature
# 省略 mask 的构造：同视频、不同受试者的非对角配对会被屏蔽。
logits = logits.masked_fill(mask, -1e4)
batch_size = preds.shape[0]
labels = torch.arange(batch_size, dtype=torch.long).to(device)
loss_e2t = self.criterion(logits, labels)
# 默认开启双向计算，以下为该分支。
loss_t2e = self.criterion(logits.T, labels)
loss = (loss_e2t + loss_t2e) / 2
```

最容易看错的是这里的 `labels`：它是 `[0, 1, ..., B-1]`，代表正确目标在 batch 中的位置，**不是情绪标签**。对第 i 个 EEG，损失要求第 i 个语义目标的分数相对更高；转置矩阵再算一次，则反过来要求语义目标找到对应 EEG。

假设 B=4，主损失正样本位于 `(0,0)、(1,1)、(2,2)、(3,3)`。同视频、不同受试者的非对角项被屏蔽，避免把共享刺激内容的样本当作普通负样本；屏蔽并不把它变成额外正样本。

双曲几何由 `ContrastiveExperimentMixin._init_contrastive_experiment` 将对象赋给 `criterion.geometry` 后接入这条路径。整体分数可写成 `负双曲距离 / 对比温度`。

**本阶段记住：双曲版本改变匹配分数的计算方式，对比损失仍在学习 EEG 与语义目标的对应关系。**当前先归一化再使用固定曲率映射，不能仅凭双曲距离就认定模型已学习到径向层级。

## 第四阶段 用同情绪的多个目标补充监督

只有启用 `hyper_emotion` 时，才加入这一项。先看 [emotion_positive_mask](model/loss/hyperbolic.py)：

```python
return labels[:, None].eq(labels[None, :])
```

这里的 `labels` 才是情绪标签。两两标签相等的位置就是情绪正样本。例如标签为 `[快乐, 悲伤, 快乐, 悲伤]`，第 0 个 EEG 的情绪正目标是第 0 和第 2 个融合语义向量。

[multi_positive_loss](model/loss/hyperbolic.py) 中的核心计算是：

```python
log_prob = F.log_softmax(scores[valid], dim=1)
return -(log_prob * positives[valid]).sum(dim=1).div(count[valid]).mean()
```

先把匹配分数变成对数概率，再只选正样本求和，除以各行正样本数，最后对样本平均。因此它鼓励每个同情绪目标都获得较高概率。

该项比较的是 EEG 与融合语义，正样本条件是同情绪；不要求来自不同受试者，也不继承主对齐的屏蔽矩阵。它与第三阶段的“按对应位置配对”承担不同任务。

在默认动态融合参数下，总损失为：

\[
L=L_{\mathrm{align}}+2(L_{\mathrm{probe,text}}+L_{\mathrm{probe,image}})
 +\lambda_e L_{\mathrm{emotion}}.
\]

`hyper_distance` 关闭情绪项；`hyper_emotion` 默认取 `lambda_e=0.1`。权重已经在 `MultiScaleContrastiveLoss` 内应用，CORE-4B 处把加权辅助项加入总损失。

**本阶段记住：主对齐项学习正确配对，情绪项进一步鼓励同类语义聚合。**

## 第五阶段 用学到的 EEG 特征做最终分类

回到 [Conv_att_simple_new.forward](model/models.py) 的 CORE-5 附近，比较两个输出分支。以下节选省略了可选的中间层提取分支：

```python
if self.saveFea:
    # 默认 return_layer 未指定时，返回主干特征。
    return out_flat
else:
    projected_fea = self.projector(out_flat)
    return F.normalize(projected_fea, dim=1)
```

预训练使用 `saveFea=False`，因此对比损失读取投影头之后的向量。特征提取时外层 `ExtractorModel.forward` 将 `saveFea` 设为 True，默认读取投影头之前的 1024 维主干特征。

提取程序加载检查点，完成输入归一化、逐 trial 特征平滑及保存；分类入口读取特征并做训练集拟合的标准化等处理。最后，[MLPModel.training_step](model/pl_models.py) 的核心只有三行：

```python
data, labels = batch
logits = self.model(data)
loss = self.criterion(logits, labels.long())
```

这里 `data` 是已经提取的特征，`self.model` 是 MLP，输出为 `[B, 9]` 的类别分数。训练使用普通分类交叉熵；预测取分数最高的类别。

**本阶段记住：双曲对齐影响预训练得到的 EEG 表示，最终默认分类器仍然是普通 MLP。**预训练中的检索准确率与最终分类准确率要分开看；当前验证融合使用真实标签生成权重，且留出折也参与选模，理解报告指标时要结合这些计算路径。

## 五个自检问题

1. EEG 主干输出的 1024 维与九分类的 9 维分别表示什么？
2. 文本损失小于图像损失时，文本权重应该增大还是减小？
3. 主对齐损失中的 labels 为什么是 batch 序号？
4. hyper_emotion 比 hyper_distance 多了什么正样本关系？
5. 最终 MLP 读取的是 EEG 投影头之前还是之后的向量？

参考答案依次是：特征维度与类别分数维度；增大；指出每行的对应目标列；同情绪的多个融合语义目标；默认读取投影头之前的主干特征。能够解释这五项，就已掌握当前实现的核心流程，再按需要回到 [完整学习方案](SOURCE_READING_GUIDE.md) 深入。
