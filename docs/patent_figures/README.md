# KEMM 船舶专利图稿输出索引

本文档由 `docs/patent_figures/generate_patent_figures.py` 自动生成。
它不只是文件清单，还用于说明每张图表达的技术含义、关键视觉元素、阅读顺序和论文 / 专利中的使用位置。

## 输出目录

- `paper_svg/`：论文版可编辑 SVG 主稿
- `paper_png/`：论文版预览 PNG
- `paper_pdf/`：论文版排版用 PDF
- `patent_bw_svg/`：专利黑白线稿 SVG
- `patent_bw_png/`：专利黑白预览 PNG
- `patent_bw_pdf/`：专利黑白 PDF

## 使用说明

- 论文版优先服务于方法表达、汇报展示和论文排版，默认弱化或隐藏专利构件号，减少视觉干扰。
- 专利黑白版优先服务于附图提交，保留必要构件号，要求在灰度打印下仍可区分层级。
- 图1到图8是专利主案核心图号；图9到图11是论文 / 汇报专用解释图，不进入专利黑白版。
- 若后续继续修改图稿，请优先修改 `generate_patent_figures.py` 后再重新导出，避免 README 与实际图稿脱节。

## 图稿详解

### 图形语言规范板

**图义概述**

该图不是技术方案本身，而是整套图稿的视觉规范板，用于统一图1-图8的形状语言、编号方式和黑白导出规则。

**这张图在表达什么**

- 说明不同类型的框分别代表主流程、机制模块、候选汇聚单元和输出单元。
- 说明附图标号、步骤编号和箭头样式如何在整套图中保持一致。
- 说明论文版与专利版之间的差异：论文版允许轻量强调色，专利版必须可退化为黑白线稿。

**关键视觉元素**

- 核心流程框、机制模块框、候选池框、输出框
- 编号徽标示例，如 `S100`、`140`、`340`、`250`
- 配色样本、线宽层级、字体层级、禁用项提示

**推荐阅读顺序**

- 先看左侧，理解不同框型分别承载什么语义。
- 再看中部，理解标题、副标题、框内文字和编号的层级关系。
- 最后看右侧，理解论文版和专利版的输出边界。

**建议用途**

主要用于内部制图规范、老师沟通和后续继续扩图时保持风格统一，通常不直接进论文正文或专利正文。

**相关文档**

- `本目录全部图稿`
- `../patent_figures_plan_ship_kemm.md`

**输出文件**

- `paper_svg/fig00_visual_language_board.svg`
- `paper_png/fig00_visual_language_board.png`
- `paper_pdf/fig00_visual_language_board.pdf`
- `patent_bw_svg/fig00_visual_language_board.svg`
- `patent_bw_png/fig00_visual_language_board.png`
- `patent_bw_pdf/fig00_visual_language_board.pdf`

### 图1 本发明总体系统框图

**图义概述**

该图展示发明的总体系统架构，回答“系统由哪些输入、模块和输出组成”，是整套方法的总览图。

**这张图在表达什么**

- 左侧输入层对应真实船舶规划环境中的四类原始信息：本船、目标船、障碍物和环境场。
- 中部处理层说明 KEMM 不是单一算子，而是由建模、诊断、分配、四类知识模块、候选池和环境选择组成的完整决策链。
- 右侧输出层说明系统最终不是只输出一个抽象评分，而是输出轨迹解集、代表轨迹和可执行控制结果。
- 底部反馈回路强调该方法运行在滚动重规划闭环中，而不是一次性静态求解。

**关键视觉元素**

- 输入对象 `210/220/230/240`
- 核心模块 `100 -> 110 -> 120 -> 130`
- 并联机制模块 `140/150/160/170`
- 候选池与选择模块 `180/190`，以及执行模块 `200`
- 输出对象 `250/260/290` 和执行反馈回路
- 论文版默认隐藏构件号，专利黑白版保留构件号

**推荐阅读顺序**

- 从左到右读：先看环境数据进入 `100` 和 `110`，再看 `120/130` 如何触发多源知识机制。
- 再看中下部：`140-170` 只是并行生成候选，不直接决定最终轨迹。
- 最后看右侧：真正输出由 `180/190/200` 这一统一竞争与执行链给出。

**建议用途**

论文中适合作为方法总览图；专利中适合作为装置实施方式和系统总体架构图，也是摘要附图备选。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../patent_figures_plan_ship_kemm.md`

**输出文件**

- `paper_svg/fig01_system_architecture.svg`
- `paper_png/fig01_system_architecture.png`
- `paper_pdf/fig01_system_architecture.pdf`
- `patent_bw_svg/fig01_system_architecture.svg`
- `patent_bw_png/fig01_system_architecture.png`
- `patent_bw_pdf/fig01_system_architecture.pdf`

### 图2 本发明方法流程图

**图义概述**

该图展示方法执行顺序，回答“系统实际是按什么步骤运行的”，是主权项最直接的图示表达。

**这张图在表达什么**

- 上排步骤说明问题是如何从动态环境数据开始，被转化为可求解的动态多目标轨迹优化问题。
- 中排步骤说明环境变化后，KEMM 不会固定地只做一种恢复，而是按比例调度多种候选来源。
- 下排步骤说明最终是通过统一候选池与约束支配环境选择得到代表轨迹并执行。
- 虚线回路说明执行后的状态会回到下一规划周期，形成持续闭环。

**关键视觉元素**

- 步骤 `S100-S1100`
- 从 `S500` 发散到 `S600/S700/S800` 的分支箭头
- `S900` 候选池、`S1000` 环境选择、`S1100` 滚动重规划
- 从 `S1100` 返回 `S100` 的循环虚线
- 论文版默认隐藏步骤编号，专利黑白版保留步骤编号

**推荐阅读顺序**

- 先横向看上排：建立动态问题和环境变化信息。
- 再看从 `S500` 向下发散的三条线，理解预算分配后并行生成候选的逻辑。
- 最后看下排和底部虚线，理解“统一筛选 + 局部执行 + 再规划”的闭环。

**建议用途**

这是最适合放在摘要页和专利摘要附图位置的一张图；论文里也可以作为方法章节第一张图。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../patent_abstract_ship_kemm_draft.md`

**输出文件**

- `paper_svg/fig02_method_flow.svg`
- `paper_png/fig02_method_flow.png`
- `paper_pdf/fig02_method_flow.pdf`
- `patent_bw_svg/fig02_method_flow.svg`
- `patent_bw_png/fig02_method_flow.png`
- `patent_bw_pdf/fig02_method_flow.pdf`

### 图3 环境变化诊断与自适应分配示意图

**图义概述**

该图聚焦 KEMM 的第一个核心创新：环境变化诊断与自适应分配。

**这张图在表达什么**

- 说明算法不是使用固定比例的 memory/prediction/transfer/reinit，而是先根据前沿特征估计当前环境变化状态。
- 变化幅度、可迁移性和预测置信度共同构成分配依据，这让不同环境阶段可以使用不同恢复策略。
- 右侧输出的 `N_mem/N_pred/N_trans/N_reinit` 表示四类候选预算，是变化响应的直接控制量。

**关键视觉元素**

- 历史前沿特征序列 `310` 与当前前沿特征
- 诊断量 `M(t)`、`T(t)`、`Cpred`
- 上下文映射模块 `130` 与策略评分器 `330`
- 四类预算输出 `N_mem/N_pred/N_trans/N_reinit`

**推荐阅读顺序**

- 先看左侧输入，理解诊断依赖的是前沿统计，而不是人工指定场景标签。
- 再看中间三个估计量，理解系统怎样判断“变化大不大、旧知识值不值得用、预测靠不靠谱”。
- 最后看右侧四个输出，理解预算分配是诊断的结果，而不是先验固定比例。

**建议用途**

论文中用于单独解释算法创新点；专利中用于支撑“先诊断再分配”的技术逻辑。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../patent_claims_ship_kemm_final.md`

**输出文件**

- `paper_svg/fig03_change_diagnosis_allocation.svg`
- `paper_png/fig03_change_diagnosis_allocation.png`
- `paper_pdf/fig03_change_diagnosis_allocation.pdf`
- `patent_bw_svg/fig03_change_diagnosis_allocation.svg`
- `patent_bw_png/fig03_change_diagnosis_allocation.png`
- `patent_bw_pdf/fig03_change_diagnosis_allocation.pdf`

### 图4 多源候选解协同生成示意图

**图义概述**

该图聚焦 KEMM 的第二个核心创新：多源候选协同生成。

**这张图在表达什么**

- 说明四类机制并不是相互替代，而是在同一次环境变化响应中并行给出候选解提名。
- 历史记忆、前沿预测、跨环境迁移和重初始化分别承担不同类型的知识复用与探索职责。
- 所有候选最后都会汇入统一候选池，因此系统保留了可比较、可竞争、可替换的架构特点。

**关键视觉元素**

- 历史记忆库 `300` 与历史记忆模块 `140`
- 前沿特征序列 `310` 与前沿预测模块 `150`
- 历史源环境集合 `320` 与迁移模块 `160`
- 重初始化模块 `170` 和统一候选池 `340`

**推荐阅读顺序**

- 先纵向看每一列，理解每类机制各自的数据来源、处理模块和候选产出。
- 再横向看底部汇聚，理解四类候选最终都进入同一个统一容器。
- 注意该图刻意强调“提名候选”而不是“直接输出结果”。

**建议用途**

论文里适合对应消融实验、模块介绍和创新点拆解；专利里适合作为多源候选技术方案的支撑图。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../patent_figures_plan_ship_kemm.md`

**输出文件**

- `paper_svg/fig04_multisource_candidates.svg`
- `paper_png/fig04_multisource_candidates.png`
- `paper_pdf/fig04_multisource_candidates.pdf`
- `patent_bw_svg/fig04_multisource_candidates.svg`
- `patent_bw_png/fig04_multisource_candidates.png`
- `patent_bw_pdf/fig04_multisource_candidates.pdf`

### 图5 统一候选池与约束支配环境选择示意图

**图义概述**

该图解释 KEMM 的关键收束机制，回答“多源候选为什么不会失控”。

**这张图在表达什么**

- 说明候选池中的解必须先经过目标值评价和约束违背度评价，不能只看目标值。
- 第一层比较的是可行性和安全性，第二层才比较 Pareto 非支配关系与多样性。
- 最终输出的不只是一个代表解，还有当前规划周期的轨迹解集，因此兼顾了解集质量和执行决策。

**关键视觉元素**

- 统一候选池 `340`
- 目标值评价与约束违背度评价
- 第一层约束支配比较
- 第二层非支配排序与多样性保持
- 输出 `250` 轨迹解集 与 `260` 代表轨迹

**推荐阅读顺序**

- 从左向右看：候选先进入统一容器，再进入两层筛选。
- 重点理解中间两层的先后关系：先可行，再优选。
- 右侧的解集与代表轨迹并列出现，说明系统既保留 Pareto 信息，也输出实际执行对象。

**建议用途**

论文里适合解释算法稳健性的来源；专利里适合支撑“统一候选池 + 约束支配环境选择”的关键权利要求。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../patent_claims_ship_kemm_final.md`

**输出文件**

- `paper_svg/fig05_candidate_pool_selection.svg`
- `paper_png/fig05_candidate_pool_selection.png`
- `paper_pdf/fig05_candidate_pool_selection.pdf`
- `patent_bw_svg/fig05_candidate_pool_selection.svg`
- `patent_bw_png/fig05_candidate_pool_selection.png`
- `patent_bw_pdf/fig05_candidate_pool_selection.pdf`

### 图6 滚动重规划执行示意图

**图义概述**

该图展示 KEMM 在船舶场景中的执行方式，回答“算法如何从一次求解变成持续规划”。

**这张图在表达什么**

- 说明整个系统运行在滚动重规划框架中，每个规划周期都只执行代表轨迹的一部分。
- 执行之后环境状态被更新，下一周期会基于新状态重新规划，因此系统天然适配动态会遇环境。
- 时间轴结构让读者能明确区分规划时域与执行时域，避免把它误解为一次性全局规划。

**关键视觉元素**

- 多个规划周期 `350`
- 每个周期中的局部规划时域 `270` 和执行时域 `280`
- 代表轨迹前段执行曲线
- 回到下一周期的环境更新箭头

**推荐阅读顺序**

- 先看横向时间轴，理解规划周期如何连续排列。
- 再看每个周期上方的两段框，理解“先规划、后执行”的结构。
- 最后看上方虚线回路，理解为什么该方法适合动态海上环境。

**建议用途**

论文中用于把优化框架和应用场景连接起来；专利中用于支撑方法执行机制和实施方式。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../ship_simulation_reference.md`

**输出文件**

- `paper_svg/fig06_rolling_replanning.svg`
- `paper_png/fig06_rolling_replanning.png`
- `paper_pdf/fig06_rolling_replanning.pdf`
- `patent_bw_svg/fig06_rolling_replanning.svg`
- `patent_bw_png/fig06_rolling_replanning.png`
- `patent_bw_pdf/fig06_rolling_replanning.pdf`

### 图7 交叉会遇场景示意图

**图义概述**

该图是实施例一的场景平面图，用于把抽象算法放回交叉会遇的物理语义环境中。

**这张图在表达什么**

- 展示本船、目标船、小岛障碍、禁入区、局部风险场和流场共同构成的复杂交叉会遇环境。
- 代表轨迹展示的是一种合理绕行趋势：既避免静态障碍和风险场，又保持向目标推进。
- 图中的 give-way / stand-on 标注让场景不仅是几何布局，也带有海事规则语义。

**关键视觉元素**

- 本船起点、目标点和代表轨迹
- 目标船 Target A / Target B 及其航向关系
- 小岛障碍 `Islet` 与禁入区 `Restricted Zone`
- 局部风险场等值线和流场箭头

**推荐阅读顺序**

- 先看起点、终点和代表轨迹，理解本船总体航迹趋势。
- 再看目标船和规则关系，理解交叉会遇的风险来源。
- 最后看障碍、风险场和流场，理解为什么轨迹会产生特定偏转。

**建议用途**

论文里适合作为案例场景图；专利里对应实施例一，用于说明方法在交叉会遇场景中的落地。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../../ship_simulation/scenario/generator.py`

**输出文件**

- `paper_svg/fig07_crossing_scene.svg`
- `paper_png/fig07_crossing_scene.png`
- `paper_pdf/fig07_crossing_scene.pdf`
- `patent_bw_svg/fig07_crossing_scene.svg`
- `patent_bw_png/fig07_crossing_scene.png`
- `patent_bw_pdf/fig07_crossing_scene.pdf`

### 图8 港口高密障碍场景示意图

**图义概述**

该图是实施例二的场景平面图，用于说明方法在港口高密障碍、受限航道和多目标船会遇条件下的应用。

**这张图在表达什么**

- 该场景通过上下边界、密集圆形障碍、多个禁入区和多艘目标船构造出典型的受限机动环境。
- 代表轨迹展示系统在狭窄通道中如何平衡安全净距、风险绕避和总体推进效率。
- 风险场与流场叠加说明港口环境中的风险并不只来自障碍物，还来自环境暴露和局部冲突区。

**关键视觉元素**

- 港区上下边界与受限航道
- 多类静态障碍物和多个禁入区
- Target A / B / C 三艘目标船
- 局部 Harbor Conflict Zone、流场箭头和代表轨迹

**推荐阅读顺序**

- 先看上下边界和大块禁入区，理解场景中的总体机动空间有多受限。
- 再看障碍和目标船分布，理解为什么这是高密拥挤场景。
- 最后看代表轨迹穿越窄通道的方式，理解该方法在复杂场景中的规划策略。

**建议用途**

论文里适合作为高密障碍案例主图；专利里对应实施例二，用于说明方法在港口受限水域中的适用性。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../../ship_simulation/scenario/generator.py`

**输出文件**

- `paper_svg/fig08_harbor_clutter_scene.svg`
- `paper_png/fig08_harbor_clutter_scene.png`
- `paper_pdf/fig08_harbor_clutter_scene.pdf`
- `patent_bw_svg/fig08_harbor_clutter_scene.svg`
- `patent_bw_png/fig08_harbor_clutter_scene.png`
- `patent_bw_pdf/fig08_harbor_clutter_scene.pdf`

### 论文专用 Graphical Abstract

**图义概述**

该图是论文专用视觉摘要，用一页横向信息图压缩表达研究动机、方法核心和结果价值。

**这张图在表达什么**

- 第一栏给出动态海上环境，说明问题输入来自真实动态航行场景。
- 第二栏给出 KEMM 的适应性响应引擎，突出 memory、prediction、transfer、reinit 与 adaptive allocation 的关系。
- 第三栏给出统一候选池与约束优先选择，突出方法不是简单堆模块，而是统一竞争筛选。
- 第四栏给出安全滚动重规划结果，浓缩表达‘更快恢复、更稳轨迹、风险下降’的研究结论。

**关键视觉元素**

- 四栏式信息结构
- 动态环境、KEMM 响应引擎、统一选择、结果收益
- 底部一句话总括研究逻辑链

**推荐阅读顺序**

- 按从左到右顺序读，分别对应问题背景、方法核心、收束机制和结果价值。
- 这张图不承担详细技术细节，而承担论文快速沟通与摘要展示任务。
- 它与专利附图不同，更强调概念压缩和视觉可读性。

**建议用途**

仅用于论文、汇报或海报展示，不用于专利黑白附图。

**相关文档**

- `../patent_abstract_ship_kemm_draft.md`
- `../figure_catalog.md`

**输出文件**

- `paper_svg/fig09_graphical_abstract.svg`
- `paper_png/fig09_graphical_abstract.png`
- `paper_pdf/fig09_graphical_abstract.pdf`

### 论文专用 KEMM 原理解释图

**图义概述**

该图是论文专用的算法原理解释图，用更强的阶段叙事解释 KEMM 的核心工作逻辑。

**这张图在表达什么**

- 第一阶段说明系统先从环境与前沿变化中感知“发生了什么变化”。
- 第二阶段说明系统不是固定配比，而是根据诊断结果分配四类候选预算。
- 第三阶段说明四类机制只负责提名候选，最终统一进入候选池。
- 第四阶段说明系统通过约束优先的竞争筛选，输出代表轨迹并进入下一轮滚动重规划。

**关键视觉元素**

- 四阶段主链：变化感知、预算分配、多源候选、统一竞争
- 每阶段下方的简要解释语句
- 连接输出与下一规划周期的闭环箭头

**推荐阅读顺序**

- 先从左到右看四个阶段，建立整体原理图景。
- 再看阶段下方短句，理解每一步解决的问题。
- 最后看底部闭环，理解算法为什么适用于动态环境。

**建议用途**

适合作为论文方法章节里的“原理总览图”，也适合老师汇报和答辩时先讲思路再讲实现。

**相关文档**

- `../patent_specification_ship_kemm_draft.md`
- `../kemm_reference.md`

**输出文件**

- `paper_svg/fig10_kemm_principle_chain.svg`
- `paper_png/fig10_kemm_principle_chain.png`
- `paper_pdf/fig10_kemm_principle_chain.pdf`

### 论文专用 模块职责解释图

**图义概述**

该图是论文专用的模块职责解释图，用于解释 memory、prediction、transfer、reinit 各自在什么变化情形下发挥作用。

**这张图在表达什么**

- 把四个模块按“适用变化类型”和“主要作用”拆开，而不是只画成并列功能块。
- 帮助读者理解四模块不是简单堆叠，而是分别对应重复场景、平滑漂移、结构迁移和大幅扰动。
- 右侧总结框强调协同逻辑：四模块负责提名候选，最终仍然回到统一竞争。

**关键视觉元素**

- 四张模块卡片：memory / prediction / transfer / reinit
- 每张卡片中的适用条件、核心动作和典型收益
- 右侧协同总结框

**推荐阅读顺序**

- 逐张看四个模块卡片，先理解各自解决什么问题。
- 再看右侧总结框，理解为什么它们最终会被统一候选池收束。
- 把这张图和图4配合使用，可以同时解释模块职责和系统协同。

**建议用途**

适合论文中的模块介绍、老师汇报时解释创新点，也适合和消融实验一起展示。

**相关文档**

- `../kemm_reference.md`
- `../figure_catalog.md`

**输出文件**

- `paper_svg/fig11_module_role_map.svg`
- `paper_png/fig11_module_role_map.png`
- `paper_pdf/fig11_module_role_map.pdf`
