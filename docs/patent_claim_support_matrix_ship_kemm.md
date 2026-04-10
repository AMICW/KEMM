# KEMM 船舶专利权利要求支撑对照表

本文档用于把 [patent_claims_ship_kemm_final.md](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_claims_ship_kemm_final.md#L1) 与 [patent_specification_ship_kemm_draft.md](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1) 逐条对齐，方便后续：

- 和代理人讨论支撑关系；
- 调整权利要求宽窄；
- 准备创造性答复或补正思路。

---

## 1. 使用说明

本表中的“支撑强度”按如下口径判断：

- `强`：说明书已有明确步骤、机制和实施方式支撑
- `中`：说明书已有机制支撑，但正式提交前最好再补一两句明确表述
- `可补强`：目前能支撑，但建议在说明书或交底书中再补一个更直接的实施方式句子

---

## 2. 总表

| 权利要求 | 技术特征摘要 | 说明书支撑章节 | 公式/机制锚点 | 建议附图 | 支撑强度 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 权1 | 数据获取、动态模型、变化诊断、自适应分配、多源候选、统一候选池、约束支配选择、安全优先代表轨迹、滚动重规划 | 3.2，5.2，5.7，5.9，5.10，5.11，5.15，5.17，5.18 | 动态目标模型、变化幅度、UCB分配、候选池构造、约束支配、执行时域 | 图1、图2、图5、图6 | 强 | 主链完整，已能支撑独立方法权利要求 |
| 权2 | 综合风险的组成项和约束违背度的组成项 | 5.6，5.7 | 风险聚合、风险目标函数、约束违背度建模 | 图5、图7、图8 | 强 | 适合作为风险/约束建模从属点 |
| 权3 | 前沿特征、变化幅度、知识可迁移性、预测置信度 | 5.10，5.13.3 | 前沿特征向量、变化幅度公式、可迁移性估计、置信度估计 | 图3 | 强 | 变化诊断支撑明确 |
| 权4 | 上下文划分、策略评分、比例转换、恢复质量更新 | 5.11，5.16 | 上下文映射、UCB评分、softmax比例化、奖励更新 | 图3 | 强 | 如需再收紧，可在正式稿中写成“上下文多臂策略分配” |
| 权5 | 环境指纹、历史记忆库检索、压缩编码存储、记忆候选恢复 | 5.12 | 环境指纹、VAE压缩、Top-K检索、解码恢复 | 图4 | 强 | 若代理人希望更实，可补“潜空间统计量”字样 |
| 权6 | 前沿演化预测模型、预测下一规划周期前沿特征、低置信回退 | 5.13 | GP预测、预测置信度、线性外推 | 图3、图4 | 强 | 支撑充分 |
| 权7 | 多源环境选择、局部结构、子空间映射/流形映射、加权融合 | 5.14 | 相似度权重、聚类、PCA子空间、Grassmann测地流 | 图4 | 强 | 若后续审查压力大，可进一步从属到“多源加权” |
| 权8 | 约束优先比较、非支配排序、多样性保持、安全优先代表轨迹、执行时域 | 5.7，5.8，5.17，5.18 | 约束支配规则、归一化得分、安全排序、H_plan/H_exec | 图5、图6 | 强 | 与 ship 场景的技术效果联系最强 |
| 权9 | 装置模块化表达 | 5.2，5.21 | 系统模块分工 | 图1 | 强 | 装置模块与方法步骤一一对应 |
| 权10 | 电子设备 | 5.22 | 处理器、存储器、程序执行 | 图1 | 强 | 常规计算机程序相关发明写法 |
| 权11 | 计算机可读存储介质 | 5.23 | 程序存储与执行 | 无需专图 | 强 | 标准配套权利要求 |
| 权12 | 计算机程序产品 | 5.23 | 程序产品实施方式 | 无需专图 | 强 | 标准配套权利要求 |

---

## 3. 分条细化说明

### 3.1 权利要求1

直接支撑点：

- [3.2 技术方案概述](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L165)
- [5.2 系统总体架构](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L307)
- [5.9 环境变化响应主流程](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L711)
- [5.15 统一候选池构造](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1239)
- [5.17 代表轨迹选择](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1282)
- [5.18 滚动重规划执行机制](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1322)

需要特别注意的支撑点：

- “精英保留候选和/或历史种群保留候选”主要由 `S900` 和 `5.15` 支撑；
- “安全优先原则”主要由 `5.17.2` 支撑；
- “执行前段轨迹进入下一规划周期”主要由 `5.18.1` 支撑。

### 3.2 权利要求2

直接支撑点：

- [5.6 风险目标建模](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L486)
- [5.7 约束违背度建模](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L630)

补强建议：

- 如果后续需要把权2写得更稳，可在说明书正式版中再加一句“综合风险由上述至少两项风险分量线性或非线性聚合得到”。

### 3.3 权利要求3

直接支撑点：

- [5.10.1 前沿特征构造](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L733)
- [5.10.2 变化幅度计算](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L755)
- [5.10.3 可迁移性估计](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L771)
- [5.13.3 预测置信度](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1060)

### 3.4 权利要求4

直接支撑点：

- [5.11.1 上下文划分](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L816)
- [5.11.2 UCB评分](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L829)
- [5.11.3 比例化输出](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L845)
- [5.11.5 奖励更新](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L882)
- [5.16 恢复质量估计](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1266)

补强建议：

- 如果后续准备更窄版本，可将“策略评分”明确写为“上置信界评分”。

### 3.5 权利要求5

直接支撑点：

- [5.12.1 记忆库存储对象](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L911)
- [5.12.2 环境指纹](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L924)
- [5.12.3 VAE 压缩原理](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L937)
- [5.12.5 相似环境检索](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L987)
- [5.12.6 解码恢复](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1003)

### 3.6 权利要求6

直接支撑点：

- [5.13.1 特征序列](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1027)
- [5.13.2 高斯过程回归](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1033)
- [5.13.3 预测置信度](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1060)
- [5.13.4 预测候选生成](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1077)
- [5.13.5 线性外推](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1101)

### 3.7 权利要求7

直接支撑点：

- [5.14.1 多源环境选择](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1125)
- [5.14.2 相似度归一化权重](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1135)
- [5.14.3 聚类与局部结构提取](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1147)
- [5.14.4 PCA 子空间构建](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1157)
- [5.14.5 Grassmann 流形测地流](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1177)
- [5.14.7 多源加权输出](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1223)

### 3.8 权利要求8

直接支撑点：

- [5.7 约束违背度建模](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L630)
- [5.8.1 约束支配规则](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L685)
- [5.8.2 拥挤距离](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L699)
- [5.17.1 目标归一化分数](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1288)
- [5.17.2 安全优先排序](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1300)
- [5.18.1 局部规划与局部执行](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1326)

### 3.9 权利要求9-12

直接支撑点：

- [5.21 装置实施方式](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1411)
- [5.22 电子设备实施方式](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1433)
- [5.23 计算机可读存储介质与程序产品实施方式](d:/VS%20Code/code/KEMM%28renew%29/docs/patent_specification_ship_kemm_draft.md#L1452)

---

## 4. 当前仍值得补强的点

虽然整体支撑已经较强，但为了进一步提高稳健性，建议在正式说明书终稿中再补下面几句：

1. 在发明内容或具体实施方式中补一句：

“所述统一候选池还可包括精英保留候选和历史种群保留候选中的至少一种。”

2. 在记忆模块中补一句：

“所述压缩编码存储可以是变分自编码器编码、主成分压缩编码或其他低维结构编码方式。”

3. 在预测模块中补一句：

“当前种群中心和当前精英中心均可作为预测候选生成的锚点。”

4. 在装置实施方式中补一句：

“所述候选生成模块可以由多个并行子模块实现，分别对应记忆候选、预测候选、迁移候选和重初始化候选。”

---

## 5. 后续使用建议

如果代理人后续反馈“主权项还可以更宽”，优先放宽：

- 权4 的“策略评分”写法
- 权5 的“压缩编码存储”写法
- 权7 的“子空间映射和/或流形映射”写法

如果代理人反馈“主权项还需要更稳”，优先收紧：

- 在权1中明确“滚动重规划”
- 在权4中明确“上下文策略分配”
- 在权8中明确“安全优先排序 + 执行前段轨迹”

