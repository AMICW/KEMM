from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
PRIVATE_PATENT_DIR = ROOT.parents[1] / ".private" / "patent_ship_kemm"
PRIVATE_FIGURE_INDEX = PRIVATE_PATENT_DIR / "04_附图索引与图稿说明_详细版.md"


@dataclass(frozen=True)
class Theme:
    name: str
    is_bw: bool
    background: str
    text: str
    muted_text: str
    border: str
    flow_fill: str
    module_fill: str
    source_fill: str
    candidate_fill: str
    output_fill: str
    accent: str
    secondary_accent: str
    risk: str
    water: str
    obstacle: str
    restricted: str
    current: str
    guide: str
    shadow: str


TITLE_FONT = 18.0
SUBTITLE_FONT = 10.6
BODY_FONT = 10.2
CAPTION_FONT = 9.4
BADGE_FONT = 7.8


THEME_PAPER = Theme(
    name="paper",
    is_bw=False,
    background="#ffffff",
    text="#1f2937",
    muted_text="#667085",
    border="#475467",
    flow_fill="#f4f8fe",
    module_fill="#fff2e8",
    source_fill="#edf4ff",
    candidate_fill="#fff7f0",
    output_fill="#eef9f1",
    accent="#4a78b8",
    secondary_accent="#b07243",
    risk="#c45b3c",
    water="#fbfdff",
    obstacle="#7b8794",
    restricted="#a14d62",
    current="#5b8def",
    guide="#d7dee8",
    shadow="#dfe7f1",
)

THEME_BW = Theme(
    name="patent_bw",
    is_bw=True,
    background="#ffffff",
    text="#101010",
    muted_text="#3d3d3d",
    border="#111111",
    flow_fill="#f7f7f7",
    module_fill="#eeeeee",
    source_fill="#f1f1f1",
    candidate_fill="#fafafa",
    output_fill="#e6e6e6",
    accent="#1a1a1a",
    secondary_accent="#555555",
    risk="#2b2b2b",
    water="#ffffff",
    obstacle="#666666",
    restricted="#333333",
    current="#4d4d4d",
    guide="#bdbdbd",
    shadow="#ededed",
)


FIGURE_NAMES = {
    "fig00_visual_language_board": "图形语言规范板",
    "fig01_system_architecture": "图1 本发明总体系统框图",
    "fig02_method_flow": "图2 本发明方法流程图",
    "fig03_change_diagnosis_allocation": "图3 环境变化诊断与自适应分配示意图",
    "fig04_multisource_candidates": "图4 多源候选解协同生成示意图",
    "fig05_candidate_pool_selection": "图5 统一候选池与约束支配环境选择示意图",
    "fig06_rolling_replanning": "图6 滚动重规划执行示意图",
    "fig07_crossing_scene": "图7 交叉会遇场景示意图",
    "fig08_harbor_clutter_scene": "图8 港口高密障碍场景示意图",
    "fig09_graphical_abstract": "论文专用 Graphical Abstract",
    "fig10_kemm_principle_chain": "论文专用 KEMM 原理解释图",
    "fig11_module_role_map": "论文专用 模块职责解释图",
}


FIGURE_DOCS = {
    "fig00_visual_language_board": {
        "summary": "该图不是技术方案本身，而是整套图稿的视觉规范板，用于统一图1-图8的形状语言、编号方式和黑白导出规则。",
        "meaning": [
            "说明不同类型的框分别代表主流程、机制模块、候选汇聚单元和输出单元。",
            "说明附图标号、步骤编号和箭头样式如何在整套图中保持一致。",
            "说明论文版与专利版之间的差异：论文版允许轻量强调色，专利版必须可退化为黑白线稿。",
        ],
        "elements": [
            "核心流程框、机制模块框、候选池框、输出框",
            "编号徽标示例，如 `S100`、`140`、`340`、`250`",
            "配色样本、线宽层级、字体层级、禁用项提示",
        ],
        "reading": [
            "先看左侧，理解不同框型分别承载什么语义。",
            "再看中部，理解标题、副标题、框内文字和编号的层级关系。",
            "最后看右侧，理解论文版和专利版的输出边界。",
        ],
        "usage": "主要用于内部制图规范、老师沟通和后续继续扩图时保持风格统一，通常不直接进论文正文或专利正文。",
        "refs": ["本目录全部图稿", "90_撰写工作台与删减建议.md"],
    },
    "fig01_system_architecture": {
        "summary": "该图展示发明的总体系统架构，回答“系统由哪些输入、模块和输出组成”，同时明确 ship 实施方式中场景感知绕行初值与统一候选竞争的关系，是整套方法的总览图。",
        "meaning": [
            "左侧输入层对应真实船舶规划环境中的四类原始信息：本船、目标船、障碍物和环境场。",
            "中部处理层说明 KEMM 不是单一算子，而是由建模、诊断、分配、四类知识模块、场景感知绕行初值和候选池竞争组成的完整决策链。",
            "右侧输出层说明系统最终不是只输出一个抽象评分，而是输出轨迹解集、代表轨迹和可执行控制结果。",
            "底部反馈回路强调该方法运行在滚动重规划闭环中，而不是一次性静态求解；其中风险维表达沿轨迹安全暴露，终端推进由 fuel/time 维和代表解排序共同承担。",
        ],
        "elements": [
            "输入对象 `210/220/230/240`",
            "核心模块 `100 -> 110 -> 120 -> 130`",
            "并联机制模块 `140/150/160/170`",
            "候选池与选择模块 `180/190`，以及执行模块 `200`",
            "场景感知绕行初值的说明文字，用于交代 ship 实施方式中的 warm-start 注入位置",
            "输出对象 `250/260/290` 和执行反馈回路",
            "论文版默认隐藏构件号，专利黑白版保留构件号",
        ],
        "reading": [
            "从左到右读：先看环境数据进入 `100` 和 `110`，再看 `120/130` 如何触发多源知识机制。",
            "再看中下部：`140-170` 只是并行生成候选，不直接决定最终轨迹；ship 场景中的绕行初值也在这一阶段并入候选池。",
            "最后看右侧：真正输出由 `180/190/200` 这一统一竞争与执行链给出。",
        ],
        "usage": "论文中适合作为方法总览图；专利中适合作为装置实施方式和系统总体架构图，也是摘要附图备选。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "90_撰写工作台与删减建议.md"],
    },
    "fig02_method_flow": {
        "summary": "该图展示方法执行顺序，回答“系统实际是按什么步骤运行的”，并补充说明 ship 实施方式中启发式绕行初值如何插入主流程，是主权项最直接的图示表达。",
        "meaning": [
            "上排步骤说明问题是如何从动态环境数据开始，被转化为可求解的动态多目标轨迹优化问题。",
            "中排步骤说明环境变化后，KEMM 不会固定地只做一种恢复，而是按比例调度多种候选来源，并可叠加场景感知绕行初值。",
            "下排步骤说明最终是通过统一候选池与约束支配环境选择得到代表轨迹并执行；终端推进不足不再伪装成风险，而是通过 fuel/time 与代表解排序体现。",
            "虚线回路说明执行后的状态会回到下一规划周期，形成持续闭环。",
        ],
        "elements": [
            "步骤 `S100-S1100`",
            "从 `S500` 发散到 `S600/S700/S800` 的分支箭头",
            "`S900` 候选池、`S1000` 环境选择、`S1100` 滚动重规划",
            "候选生成阶段对绕行初值注入策略的说明文字",
            "从 `S1100` 返回 `S100` 的循环虚线",
            "论文版默认隐藏步骤编号，专利黑白版保留步骤编号",
        ],
        "reading": [
            "先横向看上排：建立动态问题和环境变化信息。",
            "再看从 `S500` 向下发散的三条线，理解预算分配后并行生成候选的逻辑，以及绕行初值如何与这些候选一并汇入候选池。",
            "最后看下排和底部虚线，理解“统一筛选 + 局部执行 + 再规划”的闭环。",
        ],
        "usage": "这是最适合放在摘要页和专利摘要附图位置的一张图；论文里也可以作为方法章节第一张图。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "03_摘要与摘要附图说明_详细版.md"],
    },
    "fig03_change_diagnosis_allocation": {
        "summary": "该图聚焦 KEMM 的第一个核心创新：环境变化诊断与自适应分配。",
        "meaning": [
            "说明算法不是使用固定比例的 memory/prediction/transfer/reinit，而是先根据前沿特征估计当前环境变化状态。",
            "变化幅度、可迁移性和预测置信度共同构成分配依据，这让不同环境阶段可以使用不同恢复策略。",
            "右侧输出的 `N_mem/N_pred/N_trans/N_reinit` 表示四类候选预算，是变化响应的直接控制量。",
        ],
        "elements": [
            "历史前沿特征序列 `310` 与当前前沿特征",
            "诊断量 `M(t)`、`T(t)`、`Cpred`",
            "上下文映射模块 `130` 与策略评分器 `330`",
            "四类预算输出 `N_mem/N_pred/N_trans/N_reinit`",
        ],
        "reading": [
            "先看左侧输入，理解诊断依赖的是前沿统计，而不是人工指定场景标签。",
            "再看中间三个估计量，理解系统怎样判断“变化大不大、旧知识值不值得用、预测靠不靠谱”。",
            "最后看右侧四个输出，理解预算分配是诊断的结果，而不是先验固定比例。",
        ],
        "usage": "论文中用于单独解释算法创新点；专利中用于支撑“先诊断再分配”的技术逻辑。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "02_权利要求书_详细扩展版.md"],
    },
    "fig04_multisource_candidates": {
        "summary": "该图聚焦 KEMM 的第二个核心创新：多源候选协同生成。",
        "meaning": [
            "说明四类机制并不是相互替代，而是在同一次环境变化响应中并行给出候选解提名。",
            "历史记忆、前沿预测、跨环境迁移和重初始化分别承担不同类型的知识复用与探索职责。",
            "所有候选最后都会汇入统一候选池，因此系统保留了可比较、可竞争、可替换的架构特点。",
        ],
        "elements": [
            "历史记忆库 `300` 与历史记忆模块 `140`",
            "前沿特征序列 `310` 与前沿预测模块 `150`",
            "历史源环境集合 `320` 与迁移模块 `160`",
            "重初始化模块 `170` 和统一候选池 `340`",
        ],
        "reading": [
            "先纵向看每一列，理解每类机制各自的数据来源、处理模块和候选产出。",
            "再横向看底部汇聚，理解四类候选最终都进入同一个统一容器。",
            "注意该图刻意强调“提名候选”而不是“直接输出结果”。",
        ],
        "usage": "论文里适合对应消融实验、模块介绍和创新点拆解；专利里适合作为多源候选技术方案的支撑图。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "90_撰写工作台与删减建议.md"],
    },
    "fig05_candidate_pool_selection": {
        "summary": "该图解释 KEMM 的关键收束机制，回答“多源候选为什么不会失控”。",
        "meaning": [
            "说明候选池中的解必须先经过目标值评价和约束违背度评价，不能只看目标值。",
            "第一层比较的是可行性和安全性，第二层才比较 Pareto 非支配关系与多样性。",
            "最终输出的不只是一个代表解，还有当前规划周期的轨迹解集，因此兼顾了解集质量和执行决策。",
        ],
        "elements": [
            "统一候选池 `340`",
            "目标值评价与约束违背度评价",
            "第一层约束支配比较",
            "第二层非支配排序与多样性保持",
            "输出 `250` 轨迹解集 与 `260` 代表轨迹",
        ],
        "reading": [
            "从左向右看：候选先进入统一容器，再进入两层筛选。",
            "重点理解中间两层的先后关系：先可行，再优选。",
            "右侧的解集与代表轨迹并列出现，说明系统既保留 Pareto 信息，也输出实际执行对象。",
        ],
        "usage": "论文里适合解释算法稳健性的来源；专利里适合支撑“统一候选池 + 约束支配环境选择”的关键权利要求。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "02_权利要求书_详细扩展版.md"],
    },
    "fig06_rolling_replanning": {
        "summary": "该图展示 KEMM 在船舶场景中的执行方式，回答“算法如何从一次求解变成持续规划”。",
        "meaning": [
            "说明整个系统运行在滚动重规划框架中，每个规划周期都只执行代表轨迹的一部分。",
            "执行之后环境状态被更新，下一周期会基于新状态重新规划，因此系统天然适配动态会遇环境。",
            "时间轴结构让读者能明确区分规划时域与执行时域，避免把它误解为一次性全局规划。",
        ],
        "elements": [
            "多个规划周期 `350`",
            "每个周期中的局部规划时域 `270` 和执行时域 `280`",
            "代表轨迹前段执行曲线",
            "回到下一周期的环境更新箭头",
        ],
        "reading": [
            "先看横向时间轴，理解规划周期如何连续排列。",
            "再看每个周期上方的两段框，理解“先规划、后执行”的结构。",
            "最后看上方虚线回路，理解为什么该方法适合动态海上环境。",
        ],
        "usage": "论文中用于把优化框架和应用场景连接起来；专利中用于支撑方法执行机制和实施方式。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "../../docs/ship_simulation_reference.md"],
    },
    "fig07_crossing_scene": {
        "summary": "该图是实施例一的场景平面图，用于把抽象算法放回交叉会遇的物理语义环境中。",
        "meaning": [
            "展示本船、目标船、小岛障碍、禁入区、局部风险场和流场共同构成的复杂交叉会遇环境。",
            "代表轨迹展示的是一种合理绕行趋势：既避免静态障碍和风险场，又保持向目标推进。",
            "图中的 give-way / stand-on 标注让场景不仅是几何布局，也带有海事规则语义。",
        ],
        "elements": [
            "本船起点、目标点和代表轨迹",
            "目标船 Target A / Target B 及其航向关系",
            "小岛障碍 `Islet` 与禁入区 `Restricted Zone`",
            "局部风险场等值线和流场箭头",
        ],
        "reading": [
            "先看起点、终点和代表轨迹，理解本船总体航迹趋势。",
            "再看目标船和规则关系，理解交叉会遇的风险来源。",
            "最后看障碍、风险场和流场，理解为什么轨迹会产生特定偏转。",
        ],
        "usage": "论文里适合作为案例场景图；专利里对应实施例一，用于说明方法在交叉会遇场景中的落地。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "../../ship_simulation/scenario/generator.py"],
    },
    "fig08_harbor_clutter_scene": {
        "summary": "该图是实施例二的场景平面图，用于说明方法在港口高密障碍、受限航道和多目标船会遇条件下的应用。",
        "meaning": [
            "该场景通过上下边界、密集圆形障碍、多个禁入区和多艘目标船构造出典型的受限机动环境。",
            "代表轨迹展示系统在狭窄通道中如何平衡安全净距、风险绕避和总体推进效率。",
            "风险场与流场叠加说明港口环境中的风险并不只来自障碍物，还来自环境暴露和局部冲突区。",
        ],
        "elements": [
            "港区上下边界与受限航道",
            "多类静态障碍物和多个禁入区",
            "Target A / B / C 三艘目标船",
            "局部 Harbor Conflict Zone、流场箭头和代表轨迹",
        ],
        "reading": [
            "先看上下边界和大块禁入区，理解场景中的总体机动空间有多受限。",
            "再看障碍和目标船分布，理解为什么这是高密拥挤场景。",
            "最后看代表轨迹穿越窄通道的方式，理解该方法在复杂场景中的规划策略。",
        ],
        "usage": "论文里适合作为高密障碍案例主图；专利里对应实施例二，用于说明方法在港口受限水域中的适用性。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "../../ship_simulation/scenario/generator.py"],
    },
    "fig09_graphical_abstract": {
        "summary": "该图是论文专用视觉摘要，用一页横向信息图压缩表达研究动机、方法核心和结果价值。",
        "meaning": [
            "第一栏给出动态海上环境，说明问题输入来自真实动态航行场景。",
            "第二栏给出 KEMM 的适应性响应引擎，突出 memory、prediction、transfer、reinit 与 adaptive allocation 的关系。",
            "第三栏给出统一候选池与约束优先选择，突出方法不是简单堆模块，而是统一竞争筛选。",
            "第四栏给出安全滚动重规划结果，浓缩表达‘更快恢复、更稳轨迹、风险下降’的研究结论。",
        ],
        "elements": [
            "四栏式信息结构",
            "动态环境、KEMM 响应引擎、统一选择、结果收益",
            "底部一句话总括研究逻辑链",
        ],
        "reading": [
            "按从左到右顺序读，分别对应问题背景、方法核心、收束机制和结果价值。",
            "这张图不承担详细技术细节，而承担论文快速沟通与摘要展示任务。",
            "它与专利附图不同，更强调概念压缩和视觉可读性。",
        ],
        "usage": "仅用于论文、汇报或海报展示，不用于专利黑白附图。",
        "refs": ["03_摘要与摘要附图说明_详细版.md", "../../docs/figure_catalog.md"],
    },
    "fig10_kemm_principle_chain": {
        "summary": "该图是论文专用的算法原理解释图，用更强的阶段叙事解释 KEMM 的核心工作逻辑。",
        "meaning": [
            "第一阶段说明系统先从环境与前沿变化中感知“发生了什么变化”。",
            "第二阶段说明系统不是固定配比，而是根据诊断结果分配四类候选预算。",
            "第三阶段说明四类机制只负责提名候选，最终统一进入候选池。",
            "第四阶段说明系统通过约束优先的竞争筛选，输出代表轨迹并进入下一轮滚动重规划。",
        ],
        "elements": [
            "四阶段主链：变化感知、预算分配、多源候选、统一竞争",
            "每阶段下方的简要解释语句",
            "连接输出与下一规划周期的闭环箭头",
        ],
        "reading": [
            "先从左到右看四个阶段，建立整体原理图景。",
            "再看阶段下方短句，理解每一步解决的问题。",
            "最后看底部闭环，理解算法为什么适用于动态环境。",
        ],
        "usage": "适合作为论文方法章节里的“原理总览图”，也适合老师汇报和答辩时先讲思路再讲实现。",
        "refs": ["01_发明专利说明书_详细扩展版.md", "../../docs/kemm_reference.md"],
    },
    "fig11_module_role_map": {
        "summary": "该图是论文专用的模块职责解释图，用于解释 memory、prediction、transfer、reinit 各自在什么变化情形下发挥作用。",
        "meaning": [
            "把四个模块按“适用变化类型”和“主要作用”拆开，而不是只画成并列功能块。",
            "帮助读者理解四模块不是简单堆叠，而是分别对应重复场景、平滑漂移、结构迁移和大幅扰动。",
            "右侧总结框强调协同逻辑：四模块负责提名候选，最终仍然回到统一竞争。",
        ],
        "elements": [
            "四张模块卡片：memory / prediction / transfer / reinit",
            "每张卡片中的适用条件、核心动作和典型收益",
            "右侧协同总结框",
        ],
        "reading": [
            "逐张看四个模块卡片，先理解各自解决什么问题。",
            "再看右侧总结框，理解为什么它们最终会被统一候选池收束。",
            "把这张图和图4配合使用，可以同时解释模块职责和系统协同。",
        ],
        "usage": "适合论文中的模块介绍、老师汇报时解释创新点，也适合和消融实验一起展示。",
        "refs": ["../../docs/kemm_reference.md", "../../docs/figure_catalog.md"],
    },
}


def configure_fonts() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Source Han Sans CN",
        "Noto Sans CJK SC",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def out_dir(theme: Theme, suffix: str) -> Path:
    return ROOT / f"{theme.name}_{suffix}"


def ensure_dirs() -> None:
    for theme in (THEME_PAPER, THEME_BW):
        for suffix in ("svg", "png", "pdf"):
            out_dir(theme, suffix).mkdir(parents=True, exist_ok=True)


def canvas(figsize=(13.5, 8.2), facecolor="#ffffff"):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def add_heading(ax, title: str, subtitle: str, theme: Theme) -> None:
    ax.text(0.03, 0.95, title, fontsize=TITLE_FONT, fontweight="bold", color=theme.text, va="top")
    ax.text(0.03, 0.91, subtitle, fontsize=SUBTITLE_FONT, color=theme.muted_text, va="top")


def ref_badge(theme: Theme, label: str | None, *, always: bool = False) -> str | None:
    if label is None:
        return None
    if always or theme.is_bw:
        return label
    return None


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    theme: Theme,
    *,
    fill: str | None = None,
    edge: str | None = None,
    badge: str | None = None,
    fontsize: float = BODY_FONT,
    rounding: float = 0.026,
    lw: float = 1.0,
    line_spacing: float = 1.3,
    text_color: str | None = None,
    dashed: bool = False,
):
    fill = fill or theme.flow_fill
    edge = edge or theme.border
    if not theme.is_bw:
        shadow = FancyBboxPatch(
            (x + 0.004, y - 0.004),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={rounding}",
            linewidth=0.0,
            edgecolor="none",
            facecolor=theme.shadow,
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(shadow)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2 - (0.012 if badge else 0.0),
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color or theme.text,
        linespacing=line_spacing,
        zorder=3,
    )
    if badge:
        badge_w = min(0.065, w * 0.35)
        badge_h = min(0.045, h * 0.38)
        badge_patch = FancyBboxPatch(
            (x + 0.012, y + h - badge_h - 0.012),
            badge_w,
            badge_h,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=theme.background,
            zorder=4,
        )
        ax.add_patch(badge_patch)
        ax.text(
            x + 0.012 + badge_w / 2,
            y + h - 0.012 - badge_h / 2,
            badge,
            ha="center",
            va="center",
            fontsize=max(BADGE_FONT, fontsize - 2.2),
            color=theme.text,
            fontweight="bold",
            zorder=5,
        )
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    theme: Theme,
    *,
    text: str | None = None,
    rad: float = 0.0,
    lw: float = 0.9,
    color: str | None = None,
    style: str = "-|>",
    linestyle: str = "-",
    text_offset: tuple[float, float] = (0.0, 0.0),
):
    color = color or theme.border
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8.0,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=CAPTION_FONT, color=theme.muted_text, ha="center", va="center")
    return patch


def add_elbow_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    theme: Theme,
    *,
    mode: str = "hv",
    mid: float | None = None,
    lw: float = 0.9,
    color: str | None = None,
    linestyle: str = "-",
    text: str | None = None,
    text_pos: tuple[float, float] | None = None,
):
    color = color or theme.border
    x1, y1 = start
    x2, y2 = end
    if mode == "hv":
        pivot = (mid if mid is not None else x2, y1)
    else:
        pivot = (x1, mid if mid is not None else y2)
    ax.plot([x1, pivot[0]], [y1, pivot[1]], color=color, linewidth=lw, linestyle=linestyle, solid_capstyle="round", zorder=1.5)
    add_arrow(ax, pivot, end, theme, lw=lw, color=color, linestyle=linestyle, text=None if text_pos is not None else text, text_offset=(0.0, 0.0))
    if text and text_pos is not None:
        ax.text(text_pos[0], text_pos[1], text, fontsize=CAPTION_FONT, color=theme.muted_text, ha="center", va="center")


def add_group_label(ax, x: float, y: float, w: float, h: float, label: str, theme: Theme) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=theme.guide,
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h + 0.014, label, fontsize=BODY_FONT, color=theme.muted_text, va="bottom", ha="center")


def add_swatch(ax, x: float, y: float, color: str, label: str, theme: Theme) -> None:
    rect = Rectangle((x, y), 0.05, 0.035, facecolor=color, edgecolor=theme.border, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x + 0.06, y + 0.0175, label, va="center", fontsize=9.5, color=theme.text)


def draw_style_board(theme: Theme):
    fig, ax = canvas(figsize=(14, 8.4), facecolor=theme.background)
    add_heading(
        ax,
        "KEMM 船舶专利图稿图形语言规范板",
        "IEEE 系统图风格，统一字体 / 线宽 / 箭头 / 编号 / 灰度规则，用于图1-图8 的批量生成。",
        theme,
    )

    add_group_label(ax, 0.04, 0.12, 0.28, 0.72, "A. 模块与流程单元", theme)
    add_box(ax, 0.07, 0.64, 0.2, 0.11, "核心流程框\n用于主步骤与主模块", theme, fill=theme.flow_fill, badge="S100")
    add_box(ax, 0.07, 0.48, 0.2, 0.1, "机制模块\n用于 memory / prediction", theme, fill=theme.module_fill, badge="140")
    add_box(ax, 0.07, 0.34, 0.2, 0.08, "候选汇聚单元", theme, fill=theme.candidate_fill, badge="340", rounding=0.05)
    add_box(ax, 0.07, 0.22, 0.2, 0.07, "输出解集 / 代表轨迹", theme, fill=theme.output_fill, badge="250")
    add_arrow(ax, (0.17, 0.62), (0.17, 0.585), theme, text="主流程箭头", text_offset=(0.09, 0.0))
    add_arrow(ax, (0.17, 0.47), (0.17, 0.425), theme, color=theme.accent, text="并行 / 汇聚箭头", text_offset=(0.11, 0.0))

    add_group_label(ax, 0.36, 0.12, 0.28, 0.72, "B. 文字与层级", theme)
    ax.text(0.39, 0.72, "标题字体：18 pt 粗体", fontsize=18, fontweight="bold", color=theme.text)
    ax.text(0.39, 0.66, "副标题字体：10.5 pt 常规", fontsize=10.5, color=theme.muted_text)
    ax.text(0.39, 0.58, "框内文字：9.5-10.5 pt", fontsize=10.5, color=theme.text)
    ax.text(0.39, 0.50, "附图标号：专利版显示编号徽标，论文版默认隐藏", fontsize=10.0, color=theme.text)
    ax.text(0.39, 0.42, "说明短句：优先用 1-2 行，不写长段落", fontsize=10.0, color=theme.text)
    ax.text(0.39, 0.34, "灰度规则：不能只靠颜色，必须靠位置 / 形状 / 线型", fontsize=10.0, color=theme.text)
    ax.text(0.39, 0.26, "专利版：去彩色，保留黑白线稿和必要文字", fontsize=10.0, color=theme.text)

    add_group_label(ax, 0.68, 0.12, 0.28, 0.72, "C. 配色与导出", theme)
    add_swatch(ax, 0.71, 0.72, theme.flow_fill, "主流程底色", theme)
    add_swatch(ax, 0.71, 0.66, theme.module_fill, "机制模块底色", theme)
    add_swatch(ax, 0.71, 0.60, theme.candidate_fill, "候选池 / 汇聚底色", theme)
    add_swatch(ax, 0.71, 0.54, theme.output_fill, "输出单元底色", theme)
    add_swatch(ax, 0.71, 0.48, theme.accent, "论文版强调色", theme)
    add_swatch(ax, 0.71, 0.42, theme.secondary_accent, "次级强调色", theme)
    ax.text(0.71, 0.33, "导出格式：SVG / PNG / PDF", fontsize=10.2, color=theme.text)
    ax.text(0.71, 0.27, "推荐页面宽度：单栏可读，灰度打印可分辨", fontsize=10.2, color=theme.text)
    ax.text(0.71, 0.21, "禁用：渐变、厚重阴影、3D 拟物、卡通图标", fontsize=10.2, color=theme.text)
    return fig


def draw_fig01_system_architecture(theme: Theme):
    fig, ax = canvas(figsize=(14.8, 8.8), facecolor=theme.background)
    add_heading(
        ax,
        "图1 本发明总体系统框图",
        "面向动态航行环境的数据输入先进入建模与变化响应主链，再由多源知识模块与场景感知绕行初值共同提名候选，并通过统一竞争得到轨迹解集、代表轨迹与控制结果。",
        theme,
    )

    badge = lambda s: ref_badge(theme, s)

    add_group_label(ax, 0.04, 0.14, 0.18, 0.68, "输入层", theme)
    add_group_label(ax, 0.25, 0.63, 0.54, 0.20, "核心决策链", theme)
    add_group_label(ax, 0.25, 0.36, 0.54, 0.17, "核心增强响应层", theme)
    add_group_label(ax, 0.33, 0.08, 0.40, 0.20, "统一筛选与执行", theme)
    add_group_label(ax, 0.81, 0.14, 0.15, 0.68, "输出层", theme)

    inputs = [
        ("210", "本船状态数据", 0.68),
        ("220", "目标船状态数据", 0.565),
        ("230", "静态障碍物数据", 0.455),
        ("240", "环境场数据", 0.315),
    ]
    for ref, label, y in inputs:
        add_box(ax, 0.06, y, 0.14, 0.09, label, theme, fill=theme.source_fill, badge=badge(ref), fontsize=BODY_FONT)

    core = [
        ("100", "数据获取模块", 0.27),
        ("110", "轨迹建模模块", 0.42),
        ("120", "变化诊断模块", 0.57),
        ("130", "自适应分配模块", 0.72),
    ]
    for ref, label, x in core:
        add_box(ax, x, 0.68, 0.10, 0.12, label, theme, fill=theme.flow_fill, badge=badge(ref), fontsize=BODY_FONT)

    modules = [
        ("140", "历史记忆模块", 0.27),
        ("150", "前沿预测模块", 0.41),
        ("160", "跨环境迁移模块", 0.55),
        ("170", "重初始化模块", 0.69),
    ]
    for ref, label, x in modules:
        add_box(ax, x, 0.40, 0.10, 0.10, label, theme, fill=theme.module_fill, badge=badge(ref), fontsize=BODY_FONT)

    add_box(ax, 0.37, 0.18, 0.15, 0.10, "候选池构建模块", theme, fill=theme.candidate_fill, badge=badge("180"), fontsize=BODY_FONT)
    add_box(ax, 0.57, 0.18, 0.15, 0.10, "环境选择模块", theme, fill=theme.candidate_fill, badge=badge("190"), fontsize=BODY_FONT)
    add_box(ax, 0.46, 0.05, 0.18, 0.09, "轨迹输出执行模块", theme, fill=theme.output_fill, badge=badge("200"), fontsize=BODY_FONT)

    outputs = [
        ("250", "轨迹解集", 0.59),
        ("260", "代表轨迹", 0.445),
        ("290", "控制结果", 0.275),
    ]
    for ref, label, y in outputs:
        add_box(ax, 0.84, y, 0.10, 0.09, label, theme, fill=theme.output_fill, badge=badge(ref), fontsize=BODY_FONT)

    # Input merge spine.
    ax.plot([0.22, 0.22], [0.36, 0.74], color=theme.guide, linewidth=1.0, solid_capstyle="round")
    for _, _, y in inputs:
        ax.plot([0.20, 0.22], [y + 0.045, y + 0.045], color=theme.guide, linewidth=1.0, solid_capstyle="round")
    add_arrow(ax, (0.22, 0.74), (0.27, 0.74), theme, color=theme.guide, lw=0.9)
    ax.text(0.13, 0.81, "航行环境输入", fontsize=BODY_FONT, color=theme.muted_text, ha="center")

    # Core chain.
    add_arrow(ax, (0.37, 0.74), (0.42, 0.74), theme, lw=0.9)
    add_arrow(ax, (0.52, 0.74), (0.57, 0.74), theme, lw=0.9)
    add_arrow(ax, (0.67, 0.74), (0.72, 0.74), theme, lw=0.9)

    # Allocation bus to four knowledge modules.
    ax.plot([0.77, 0.77], [0.68, 0.56], color=theme.accent if not theme.is_bw else theme.border, linewidth=1.0, solid_capstyle="round")
    ax.plot([0.32, 0.77], [0.56, 0.56], color=theme.accent if not theme.is_bw else theme.border, linewidth=1.0, solid_capstyle="round")
    for _, _, x in modules:
        add_arrow(ax, (x + 0.05, 0.56), (x + 0.05, 0.50), theme, color=theme.accent if not theme.is_bw else theme.border, lw=0.9)
    ax.text(0.545, 0.588, "依据变化信息调度四类候选来源", fontsize=CAPTION_FONT, color=theme.muted_text, ha="center")

    # Merge bus from modules to candidate pool.
    ax.plot([0.32, 0.74], [0.34, 0.34], color=theme.secondary_accent if not theme.is_bw else theme.border, linewidth=1.0, solid_capstyle="round")
    for _, _, x in modules:
        add_arrow(ax, (x + 0.05, 0.40), (x + 0.05, 0.34), theme, color=theme.secondary_accent if not theme.is_bw else theme.border, lw=0.9)
    add_elbow_arrow(ax, (0.50, 0.34), (0.445, 0.28), theme, mode="hv", mid=0.445, color=theme.secondary_accent if not theme.is_bw else theme.border, lw=0.9)
    ax.text(0.53, 0.313, "并行提名候选并与绕行初值统一汇聚", fontsize=CAPTION_FONT, color=theme.muted_text, ha="center")
    ax.text(0.74, 0.355, "ship实施时可注入\n场景感知绕行初值", fontsize=CAPTION_FONT, color=theme.muted_text, ha="center")

    # Selection and execution.
    add_arrow(ax, (0.52, 0.23), (0.57, 0.23), theme, lw=0.9)
    add_elbow_arrow(ax, (0.64, 0.18), (0.555, 0.14), theme, mode="hv", mid=0.555, lw=0.9)

    add_elbow_arrow(ax, (0.72, 0.23), (0.84, 0.635), theme, mode="hv", mid=0.80, lw=0.9)
    add_elbow_arrow(ax, (0.72, 0.215), (0.84, 0.49), theme, mode="hv", mid=0.79, lw=0.9)
    add_elbow_arrow(ax, (0.64, 0.095), (0.84, 0.32), theme, mode="hv", mid=0.77, lw=0.9)

    add_arrow(ax, (0.555, 0.05), (0.555, 0.025), theme, text="执行反馈", text_offset=(0.06, 0.0), lw=0.9)
    add_arrow(ax, (0.555, 0.025), (0.41, 0.08), theme, rad=-0.30, color=theme.guide, linestyle="--", lw=0.9)

    ax.text(0.52, 0.012, "核心逻辑：先建模并诊断变化，再调度记忆 / 预测 / 迁移 / 探索并融合绕行初值；风险维表达安全暴露，终端推进由 fuel/time 与代表解排序体现。", fontsize=CAPTION_FONT, color=theme.muted_text, ha="center")
    return fig


def draw_fig02_method_flow(theme: Theme):
    fig, ax = canvas(figsize=(14.5, 9.0), facecolor=theme.background)
    add_heading(
        ax,
        "图2 本发明方法流程图",
        "主流程覆盖数据获取、建模、环境变化响应、绕行初值注入、候选池筛选和滚动重规划执行，可直接作为摘要附图使用。",
        theme,
    )

    positions = {
        "S100": (0.04, 0.72),
        "S200": (0.23, 0.72),
        "S300": (0.42, 0.72),
        "S400": (0.61, 0.72),
        "S500": (0.80, 0.72),
        "S600": (0.15, 0.42),
        "S700": (0.41, 0.42),
        "S800": (0.67, 0.42),
        "S900": (0.31, 0.17),
        "S1000": (0.53, 0.17),
        "S1100": (0.75, 0.17),
    }
    labels = {
        "S100": "获取动态航行\n环境数据",
        "S200": "构建动态多目标\n轨迹优化模型",
        "S300": "归档当前环境\n精英信息",
        "S400": "确定环境变化\n信息",
        "S500": "确定候选预算与\n启发式注入策略",
        "S600": "生成记忆候选",
        "S700": "生成预测候选",
        "S800": "生成迁移候选",
        "S900": "构建统一\n候选池",
        "S1000": "执行约束支配\n环境选择",
        "S1100": "确定代表轨迹并\n执行滚动重规划",
    }
    fills = {
        "S100": theme.flow_fill,
        "S200": theme.flow_fill,
        "S300": theme.flow_fill,
        "S400": theme.flow_fill,
        "S500": theme.module_fill,
        "S600": theme.module_fill,
        "S700": theme.module_fill,
        "S800": theme.module_fill,
        "S900": theme.candidate_fill,
        "S1000": theme.candidate_fill,
        "S1100": theme.output_fill,
    }

    add_group_label(ax, 0.03, 0.66, 0.92, 0.20, "问题建模与变化诊断", theme)
    add_group_label(ax, 0.12, 0.36, 0.74, 0.18, "多源候选与启发式初值", theme)
    add_group_label(ax, 0.28, 0.11, 0.65, 0.18, "统一筛选与滚动执行", theme)

    for key, (x, y) in positions.items():
        add_box(ax, x, y, 0.14, 0.11, labels[key], theme, fill=fills[key], badge=ref_badge(theme, key), fontsize=BODY_FONT)

    add_arrow(ax, (0.18, 0.775), (0.23, 0.775), theme)
    add_arrow(ax, (0.37, 0.775), (0.42, 0.775), theme)
    add_arrow(ax, (0.56, 0.775), (0.61, 0.775), theme)
    add_arrow(ax, (0.75, 0.775), (0.80, 0.775), theme)

    bus_color = theme.accent if not theme.is_bw else theme.border
    ax.plot([0.87, 0.87], [0.72, 0.58], color=bus_color, linewidth=1.0, solid_capstyle="round")
    ax.plot([0.22, 0.87], [0.58, 0.58], color=bus_color, linewidth=1.0, solid_capstyle="round")
    for end_x in (0.22, 0.48, 0.74):
        add_arrow(ax, (end_x, 0.58), (end_x, 0.53), theme, color=bus_color)
    ax.text(0.89, 0.62, "记忆 / 预测 /\n迁移 / 重初始化\n与绕行初值协同调度", ha="center", va="center", fontsize=CAPTION_FONT, color=theme.muted_text)

    merge_color = theme.secondary_accent if not theme.is_bw else theme.border
    ax.plot([0.22, 0.74], [0.34, 0.34], color=merge_color, linewidth=1.0, solid_capstyle="round")
    for start_x in (0.22, 0.48, 0.74):
        add_arrow(ax, (start_x, 0.42), (start_x, 0.34), theme, color=merge_color)
    add_elbow_arrow(ax, (0.48, 0.34), (0.38, 0.28), theme, mode="hv", mid=0.38, color=merge_color, lw=0.9)
    add_arrow(ax, (0.45, 0.225), (0.53, 0.225), theme)
    add_arrow(ax, (0.67, 0.225), (0.75, 0.225), theme)

    loop_color = theme.guide
    ax.plot([0.82, 0.82], [0.17, 0.07], color=loop_color, linewidth=0.9, linestyle="--", solid_capstyle="round")
    ax.plot([0.82, 0.07], [0.07, 0.07], color=loop_color, linewidth=0.9, linestyle="--", solid_capstyle="round")
    add_arrow(ax, (0.07, 0.07), (0.07, 0.72), theme, color=loop_color, linestyle="--", lw=0.9)
    ax.text(
        0.77,
        0.055,
        "下一规划周期返回起始步骤" if not theme.is_bw else "下一规划周期返回 S100",
        fontsize=CAPTION_FONT,
        color=theme.muted_text,
        ha="right",
    )
    ax.text(0.05, 0.08, "核心逻辑：先诊断环境变化，再融合绕行初值与多源候选，最后以约束优先完成筛选并执行局部前段。", fontsize=CAPTION_FONT, color=theme.muted_text)
    return fig


def draw_fig03_change_diagnosis(theme: Theme):
    fig, ax = canvas(figsize=(14.2, 8.2), facecolor=theme.background)
    add_heading(
        ax,
        "图3 环境变化诊断与自适应分配示意图",
        "通过前沿特征构造、变化幅度估计、可迁移性估计和预测置信度估计，驱动 Contextual UCB 分配四类恢复预算。",
        theme,
    )

    add_box(
        ax,
        0.05,
        0.58,
        0.18,
        0.18,
        "310 历史前沿特征序列\nphi(t-k ... t-1)" if theme.is_bw else "历史前沿特征序列\nphi(t-k ... t-1)",
        theme,
        fill=theme.source_fill,
        fontsize=BODY_FONT,
    )
    add_box(ax, 0.05, 0.32, 0.18, 0.12, "当前前沿特征\nphi(t)", theme, fill=theme.source_fill, fontsize=BODY_FONT)

    add_box(ax, 0.30, 0.62, 0.15, 0.12, "变化幅度计算\nM(t)", theme, fill=theme.flow_fill, badge=ref_badge(theme, "120"))
    add_box(ax, 0.30, 0.46, 0.15, 0.12, "可迁移性估计\nT(t)", theme, fill=theme.flow_fill, badge=ref_badge(theme, "120"))
    add_box(ax, 0.30, 0.30, 0.15, 0.12, "预测置信度估计\nCpred", theme, fill=theme.flow_fill, badge=ref_badge(theme, "120"))

    add_box(ax, 0.53, 0.46, 0.16, 0.18, "上下文区间映射\nctx(M(t))", theme, fill=theme.module_fill, badge=ref_badge(theme, "130"))
    add_box(
        ax,
        0.74,
        0.46,
        0.16,
        0.18,
        "330 策略评分器\nContextual UCB" if theme.is_bw else "策略评分器\nContextual UCB",
        theme,
        fill=theme.module_fill,
        fontsize=BODY_FONT,
    )

    outputs = [
        ("N_mem", 0.68),
        ("N_pred", 0.55),
        ("N_trans", 0.42),
        ("N_reinit", 0.29),
    ]
    for label, y in outputs:
        add_box(ax, 0.92, y, 0.06, 0.08, label, theme, fill=theme.output_fill, fontsize=CAPTION_FONT, rounding=0.015)

    add_arrow(ax, (0.23, 0.67), (0.30, 0.68), theme)
    add_arrow(ax, (0.23, 0.64), (0.30, 0.52), theme)
    add_arrow(ax, (0.23, 0.38), (0.30, 0.36), theme)
    add_arrow(ax, (0.45, 0.68), (0.53, 0.56), theme)
    add_arrow(ax, (0.45, 0.52), (0.53, 0.55), theme)
    add_arrow(ax, (0.45, 0.36), (0.53, 0.54), theme)
    add_arrow(ax, (0.69, 0.55), (0.74, 0.55), theme)

    for _, y in outputs:
        add_arrow(ax, (0.90, 0.55), (0.92, y + 0.04), theme, color=theme.accent if not theme.is_bw else theme.border)

    ax.text(0.53, 0.36, "先诊断再分配\n不是固定比例", fontsize=BODY_FONT, color=theme.secondary_accent if not theme.is_bw else theme.muted_text, ha="center", va="center", linespacing=1.2)
    ax.text(0.05, 0.12, "诊断量建议：M(t) 表示变化强度，T(t) 表示历史知识可迁移性，Cpred 表示预测可靠性。", fontsize=CAPTION_FONT, color=theme.muted_text)
    return fig


def draw_fig04_multisource_candidates(theme: Theme):
    fig, ax = canvas(figsize=(14.2, 8.4), facecolor=theme.background)
    add_heading(
        ax,
        "图4 多源候选解协同生成示意图",
        "记忆、预测、迁移和重初始化四类机制只负责提名候选，所有候选统一汇入候选池，任何单模块都不直接决定最终轨迹。",
        theme,
    )

    xs = [0.06, 0.29, 0.52, 0.75]
    sources = [
        ("300", "历史记忆库", "140", "历史记忆模块", "记忆候选"),
        ("310", "前沿特征序列", "150", "前沿预测模块", "预测候选"),
        ("320", "历史源环境集合", "160", "跨环境迁移模块", "迁移候选"),
        (None, "探索预算触发", "170", "重初始化模块", "重初始化候选"),
    ]
    for x, (source_badge, source_label, badge, module_label, cand_label) in zip(xs, sources):
        source_text = source_label if not (theme.is_bw and source_badge is None) else source_label
        add_box(ax, x, 0.70, 0.15, 0.10, source_text, theme, fill=theme.source_fill, badge=ref_badge(theme, source_badge), fontsize=BODY_FONT)
        add_box(ax, x, 0.51, 0.15, 0.11, module_label, theme, fill=theme.module_fill, badge=ref_badge(theme, badge), fontsize=BODY_FONT)
        add_box(ax, x, 0.28, 0.15, 0.12, cand_label, theme, fill=theme.candidate_fill, fontsize=BODY_FONT, rounding=0.03)

        add_arrow(ax, (x + 0.075, 0.70), (x + 0.075, 0.62), theme)
        add_arrow(ax, (x + 0.075, 0.51), (x + 0.075, 0.40), theme)

        for dx, dy in [(0.035, 0.04), (0.075, 0.065), (0.115, 0.04)]:
            circ = Circle((x + dx, 0.31 + dy), 0.007, facecolor=theme.accent, edgecolor=theme.border, linewidth=0.5)
            ax.add_patch(circ)

    add_box(ax, 0.34, 0.06, 0.32, 0.12, "统一候选池", theme, fill=theme.output_fill, badge=ref_badge(theme, "340"), fontsize=BODY_FONT + 0.8, rounding=0.04)
    merge_color = theme.secondary_accent if not theme.is_bw else theme.border
    ax.plot([0.135, 0.825], [0.22, 0.22], color=merge_color, linewidth=1.0, solid_capstyle="round")
    for x in xs:
        add_arrow(ax, (x + 0.075, 0.28), (x + 0.075, 0.22), theme, color=merge_color)
    add_arrow(ax, (0.50, 0.22), (0.50, 0.18), theme, color=merge_color)

    ax.text(0.50, 0.24, "并行提名", ha="center", fontsize=BODY_FONT, color=theme.muted_text)
    ax.text(0.50, 0.015, "说明：候选池统一承接四类来源，最终优劣由环境选择决定，而非由单个模块直接输出。", ha="center", fontsize=CAPTION_FONT, color=theme.muted_text)
    return fig


def draw_fig05_candidate_pool(theme: Theme):
    fig, ax = canvas(figsize=(14.3, 8.5), facecolor=theme.background)
    add_heading(
        ax,
        "图5 统一候选池与约束支配环境选择示意图",
        "先比较约束违背度，再比较非支配关系与多样性，由统一竞争得到轨迹解集与安全优先的代表轨迹。",
        theme,
    )

    add_box(ax, 0.05, 0.32, 0.20, 0.26, "统一候选池", theme, fill=theme.candidate_fill, badge=ref_badge(theme, "340"), fontsize=BODY_FONT + 0.8, rounding=0.04)
    rng = np.random.default_rng(42)
    dots = 24
    xs = 0.075 + rng.random(dots) * 0.15
    ys = 0.36 + rng.random(dots) * 0.18
    colors = [theme.accent, theme.secondary_accent, theme.risk, theme.border]
    for i, (x, y) in enumerate(zip(xs, ys)):
        circ = Circle((float(x), float(y)), 0.0058, facecolor=colors[i % len(colors)], edgecolor=theme.border, linewidth=0.45)
        ax.add_patch(circ)

    add_box(ax, 0.33, 0.46, 0.16, 0.11, "目标值评价", theme, fill=theme.flow_fill, badge=ref_badge(theme, "F"))
    add_box(ax, 0.33, 0.28, 0.16, 0.11, "约束违背度评价", theme, fill=theme.flow_fill, badge=ref_badge(theme, "CV"))
    add_box(ax, 0.57, 0.46, 0.18, 0.11, "第一层：约束支配比较", theme, fill=theme.module_fill, fontsize=BODY_FONT)
    add_box(ax, 0.57, 0.28, 0.18, 0.11, "第二层：非支配排序\n+ 多样性保持", theme, fill=theme.module_fill, fontsize=BODY_FONT)

    add_box(ax, 0.82, 0.46, 0.12, 0.11, "轨迹解集", theme, fill=theme.output_fill, badge=ref_badge(theme, "250"), fontsize=BODY_FONT)
    add_box(ax, 0.82, 0.28, 0.12, 0.11, "代表轨迹", theme, fill=theme.output_fill, badge=ref_badge(theme, "260"), fontsize=BODY_FONT)
    ax.text(0.88, 0.23, "安全优先 + 折中效率", fontsize=CAPTION_FONT, color=theme.muted_text, ha="center")

    add_arrow(ax, (0.25, 0.45), (0.33, 0.515), theme)
    add_arrow(ax, (0.25, 0.45), (0.33, 0.335), theme)
    add_arrow(ax, (0.49, 0.515), (0.57, 0.515), theme)
    add_arrow(ax, (0.49, 0.335), (0.57, 0.335), theme)
    add_arrow(ax, (0.75, 0.515), (0.82, 0.515), theme)
    add_arrow(ax, (0.75, 0.335), (0.82, 0.335), theme)

    ax.text(0.66, 0.61, "先比较可行性", fontsize=CAPTION_FONT, color=theme.secondary_accent if not theme.is_bw else theme.muted_text, ha="center")
    ax.text(0.66, 0.22, "再比较 Pareto 优势与多样性", fontsize=CAPTION_FONT, color=theme.secondary_accent if not theme.is_bw else theme.muted_text, ha="center")
    return fig


def draw_fig06_rolling_replanning(theme: Theme):
    fig, ax = canvas(figsize=(14.5, 8.2), facecolor=theme.background)
    add_heading(
        ax,
        "图6 滚动重规划执行示意图",
        "规划周期由局部规划时域和执行时域组成；每次只执行代表轨迹前段，再触发下一轮规划。"
        if not theme.is_bw
        else "规划周期 350 由局部规划时域 270 和执行时域 280 组成；每次只执行代表轨迹前段，再触发下一轮规划。",
        theme,
    )

    ax.plot([0.07, 0.93], [0.52, 0.52], color=theme.border, linewidth=1.0, solid_capstyle="round")
    ax.text(0.94, 0.52, "时间", va="center", fontsize=BODY_FONT, color=theme.text)

    cycle_x = [0.12, 0.39, 0.66]
    for idx, x in enumerate(cycle_x, start=1):
        add_box(
            ax,
            x,
            0.62,
            0.10,
            0.09,
            "局部规划时域\n270" if theme.is_bw else "局部规划时域",
            theme,
            fill=theme.flow_fill,
            badge=ref_badge(theme, f"350{idx}"),
            fontsize=BODY_FONT,
        )
        add_box(
            ax,
            x + 0.11,
            0.62,
            0.10,
            0.09,
            "执行时域\n280" if theme.is_bw else "执行时域",
            theme,
            fill=theme.output_fill,
            fontsize=BODY_FONT,
        )
        add_arrow(ax, (x + 0.10, 0.665), (x + 0.11, 0.665), theme)

        plan_center = x + 0.05
        exec_center = x + 0.16
        ax.plot([plan_center, exec_center], [0.52, 0.52], color=theme.guide, linewidth=1.2, solid_capstyle="round")
        ax.plot([plan_center - 0.02, plan_center + 0.03], [0.44 + idx * 0.04, 0.42 + idx * 0.04], color=theme.accent, linewidth=1.2, solid_capstyle="round")
        ax.plot([exec_center - 0.03, exec_center + 0.03], [0.36 + idx * 0.08, 0.34 + idx * 0.08], color=theme.secondary_accent, linewidth=1.2, solid_capstyle="round")
        add_arrow(ax, (x + 0.21, 0.665), (x + 0.27, 0.665), theme, color=theme.guide, linestyle="--")

    route = np.array([
        [0.09, 0.30],
        [0.24, 0.34],
        [0.50, 0.48],
        [0.77, 0.58],
        [0.90, 0.66],
    ])
    ax.plot(route[:, 0], route[:, 1], color=theme.accent, linewidth=2.5, linestyle="-")
    ax.text(0.47, 0.71, "仅执行代表轨迹前段", fontsize=BODY_FONT, color=theme.text, ha="center")
    add_arrow(ax, (0.84, 0.66), (0.17, 0.78), theme, rad=0.28, color=theme.guide, linestyle="--", text="环境更新后再次规划", text_offset=(0.01, 0.03))
    ax.text(0.10, 0.12, "关键点：本发明不是一次性全局规划，而是局部规划、局部执行、持续更新的闭环。", fontsize=CAPTION_FONT, color=theme.muted_text)
    return fig


def add_ship(ax, x: float, y: float, heading_deg: float, color: str, label: str, theme: Theme, scale: float = 140.0):
    ang = np.deg2rad(heading_deg)
    hull = np.array([
        [scale * 0.7, 0.0],
        [-scale * 0.5, scale * 0.25],
        [-scale * 0.3, 0.0],
        [-scale * 0.5, -scale * 0.25],
    ])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    hull = hull @ rot.T
    hull[:, 0] += x
    hull[:, 1] += y
    poly = Polygon(hull, closed=True, facecolor=color, edgecolor=theme.border, linewidth=1.2)
    ax.add_patch(poly)
    ax.text(x, y - scale * 0.55, label, ha="center", va="top", fontsize=9.0, color=theme.text)


def add_ship_axes(ax, x: float, y: float, heading_deg: float, color: str, theme: Theme, scale: float = 0.028):
    ang = np.deg2rad(heading_deg)
    hull = np.array([
        [scale * 0.9, 0.0],
        [-scale * 0.55, scale * 0.32],
        [-scale * 0.25, 0.0],
        [-scale * 0.55, -scale * 0.32],
    ])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    hull = hull @ rot.T
    hull[:, 0] += x
    hull[:, 1] += y
    poly = Polygon(hull, closed=True, facecolor=color, edgecolor=theme.border, linewidth=1.0, transform=ax.transAxes)
    ax.add_patch(poly)


def add_current_field(ax, theme: Theme, xs: Iterable[float], ys: Iterable[float], dx: float, dy: float):
    for x in xs:
        for y in ys:
            ax.arrow(
                x,
                y,
                dx,
                dy,
                width=10 if not theme.is_bw else 6,
                head_width=80,
                head_length=120,
                length_includes_head=True,
                color=theme.current,
                alpha=0.45 if not theme.is_bw else 0.75,
                linewidth=0.8,
            )


def style_scene_axes(ax, theme: Theme, title: str, subtitle: str):
    ax.set_facecolor(theme.water)
    ax.set_xlim(-250, 8000)
    ax.set_ylim(-3400, 2800)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, color=theme.guide, alpha=0.55)
    ax.set_xlabel("x / m", color=theme.text)
    ax.set_ylabel("y / m", color=theme.text)
    ax.tick_params(colors=theme.text)
    for spine in ax.spines.values():
        spine.set_color(theme.border)
        spine.set_linewidth(1.2)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13.5, color=theme.text, pad=12)


def draw_fig07_crossing(theme: Theme):
    fig, ax = plt.subplots(figsize=(12.8, 8.8))
    fig.patch.set_facecolor(theme.background)
    style_scene_axes(ax, theme, "图7 交叉会遇场景示意图", "含本船、两艘目标船、小岛障碍、禁入区、局部风险场和流场箭头。")

    risk_x = np.linspace(0, 8000, 220)
    risk_y = np.linspace(-3200, 2800, 180)
    X, Y = np.meshgrid(risk_x, risk_y)
    Z = (
        0.65 * np.exp(-(((X - 5000.0) / 1100.0) ** 2 + ((Y - 200.0) / 700.0) ** 2))
        + 0.38 * np.exp(-(((X - 2500.0) / 850.0) ** 2 + ((Y + 900.0) / 650.0) ** 2))
    )
    if theme.is_bw:
        ax.contour(X, Y, Z, levels=[0.15, 0.3, 0.45, 0.6], colors=[theme.risk], linewidths=1.0, linestyles="--")
    else:
        ax.contourf(X, Y, Z, levels=np.linspace(0.08, 0.65, 6), cmap="OrRd", alpha=0.18)
        ax.contour(X, Y, Z, levels=[0.2, 0.4, 0.55], colors=[theme.risk], linewidths=1.0, alpha=0.6)

    add_current_field(ax, theme, xs=np.linspace(700, 7200, 6), ys=np.linspace(-2600, 2000, 4), dx=140, dy=95)

    island = Circle((3600, -150), 420, facecolor="#d1d5db" if not theme.is_bw else "#d9d9d9", edgecolor=theme.border, linewidth=1.3)
    ax.add_patch(island)
    ax.text(3600, -150, "Islet", ha="center", va="center", fontsize=9.5, color=theme.text)

    restricted = np.array([[4700, 600], [6100, 900], [5900, 1800], [4500, 1500]])
    restricted_patch = Polygon(
        restricted,
        closed=True,
        facecolor="none" if theme.is_bw else "#fde2e7",
        edgecolor=theme.restricted,
        linewidth=1.5,
        hatch="///" if theme.is_bw else None,
    )
    ax.add_patch(restricted_patch)
    ax.text(5320, 1250, "Restricted Zone", ha="center", va="center", fontsize=9.0, color=theme.text)

    own_color = "#2f5da8" if not theme.is_bw else "#404040"
    target_a_color = "#b45309" if not theme.is_bw else "#666666"
    target_b_color = "#15803d" if not theme.is_bw else "#8a8a8a"
    add_ship(ax, 0.0, -480.0, 2.0, own_color, "Own Ship", theme)
    add_ship(ax, 3250.0, -3200.0, 90.0, target_a_color, "Target A", theme)
    add_ship(ax, 5400.0, 1700.0, -90.0, target_b_color, "Target B", theme)

    route = np.array([
        [0.0, -480.0],
        [1600.0, -760.0],
        [2900.0, -860.0],
        [4300.0, -620.0],
        [5850.0, -80.0],
        [7300.0, 320.0],
    ])
    ax.plot(route[:, 0], route[:, 1], color=theme.accent, linewidth=2.6, label="Representative route")
    ax.scatter([7300.0], [320.0], marker="*", s=180, color=theme.secondary_accent, edgecolor=theme.border, zorder=5)
    ax.text(7300.0, 470.0, "Goal", ha="center", fontsize=9.6, color=theme.text)
    ax.text(280.0, -760.0, "Start", ha="left", fontsize=9.6, color=theme.text)

    ax.annotate("give-way", xy=(3250, -2900), xytext=(2750, -2350), arrowprops=dict(arrowstyle="->", color=theme.border), fontsize=9.5, color=theme.text)
    ax.annotate("stand-on", xy=(5400, 1400), xytext=(5800, 2050), arrowprops=dict(arrowstyle="->", color=theme.border), fontsize=9.5, color=theme.text)
    ax.text(4900, 260, "局部风险场", fontsize=9.5, color=theme.risk)
    ax.text(1200, 2150, "流场方向", fontsize=9.2, color=theme.current)

    legend_elems = [
        Line2D([0], [0], color=theme.accent, lw=2.6, label="代表轨迹"),
        Rectangle((0, 0), 1, 1, facecolor="#d1d5db" if not theme.is_bw else "#d9d9d9", edgecolor=theme.border, label="静态障碍物"),
        Line2D([0], [0], color=theme.risk, lw=1.1, linestyle="--" if theme.is_bw else "-", label="风险场等值线"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=True, fontsize=8.8)
    return fig


def draw_fig08_harbor(theme: Theme):
    fig, ax = plt.subplots(figsize=(13.2, 9.2))
    fig.patch.set_facecolor(theme.background)
    style_scene_axes(ax, theme, "图8 港口高密障碍场景示意图", "含受限航道、密集障碍物、目标船会遇、局部风险场和受限机动空间。")

    risk_x = np.linspace(0, 8000, 240)
    risk_y = np.linspace(-3200, 2800, 200)
    X, Y = np.meshgrid(risk_x, risk_y)
    Z = (
        0.34 * np.exp(-(((X - 3800.0) / 2300.0) ** 2 + ((Y - 80.0) / 980.0) ** 2))
        + 0.62 * np.exp(-(((X - 4950.0) / 1120.0) ** 2 + ((Y + 120.0) / 820.0) ** 2))
        + 0.42 * np.exp(-(((X - 2140.0) / 1180.0) ** 2 + ((Y + 1380.0) / 640.0) ** 2))
    )
    if theme.is_bw:
        ax.contour(X, Y, Z, levels=[0.18, 0.32, 0.46, 0.60], colors=[theme.risk], linewidths=1.0, linestyles="--")
    else:
        ax.contourf(X, Y, Z, levels=np.linspace(0.10, 0.65, 7), cmap="OrRd", alpha=0.15)
        ax.contour(X, Y, Z, levels=[0.18, 0.34, 0.52], colors=[theme.risk], linewidths=0.95, alpha=0.6)

    add_current_field(ax, theme, xs=np.linspace(900, 7000, 5), ys=np.linspace(-1200, 1200, 3), dx=150, dy=58)

    boundaries = [
        np.array([[350, 1850], [7900, 1850], [7900, 2700], [350, 2700]]),
        np.array([[350, -2700], [7900, -2700], [7900, -1850], [350, -1850]]),
    ]
    for poly in boundaries:
        patch = Polygon(poly, closed=True, facecolor="#f2f2f2" if not theme.is_bw else "#f7f7f7", edgecolor=theme.border, linewidth=1.3)
        ax.add_patch(patch)

    circles = [
        ("Beacon A", (1120, -1020), 230),
        ("Beacon B", (1960, 170), 250),
        ("Mooring", (2780, -610), 310),
        ("Islet East", (3800, 860), 280),
        ("Cluster C", (4550, -1120), 250),
        ("Shoal South", (5650, -240), 300),
        ("Breakwater Tip", (6460, 820), 250),
    ]
    for name, center, radius in circles:
        patch = Circle(center, radius, facecolor="#d9dde4" if not theme.is_bw else "#e0e0e0", edgecolor=theme.border, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(center[0], center[1], name, fontsize=7.8, ha="center", va="center", color=theme.text)

    keepouts = [
        ("West Pier Zone", np.array([[820, 760], [1530, 1040], [1390, 1710], [700, 1430]])),
        ("Central Terminal", np.array([[3150, 1180], [4310, 1430], [4140, 2230], [2970, 1990]])),
        ("South Cargo Apron", np.array([[5050, -1710], [6560, -1570], [6480, -980], [4970, -1110]])),
        ("East Dock Basin", np.array([[6760, -180], [7590, 120], [7420, 960], [6640, 720]])),
    ]
    for label, poly in keepouts:
        patch = Polygon(
            poly,
            closed=True,
            facecolor="none" if theme.is_bw else "#e8eefc",
            edgecolor=theme.restricted,
            linewidth=1.4,
            hatch="///" if theme.is_bw else None,
        )
        ax.add_patch(patch)
        center = poly.mean(axis=0)
        ax.text(center[0], center[1], label, fontsize=7.8, ha="center", va="center", color=theme.text)

    own_color = "#2f5da8" if not theme.is_bw else "#404040"
    target_a_color = "#c2410c" if not theme.is_bw else "#666666"
    target_b_color = "#15803d" if not theme.is_bw else "#7b7b7b"
    target_c_color = "#d97706" if not theme.is_bw else "#949494"
    add_ship(ax, 150.0, -1450.0, 5.0, own_color, "Own Ship", theme)
    add_ship(ax, 2180.0, 2150.0, -82.0, target_a_color, "Target A", theme)
    add_ship(ax, 4820.0, -2180.0, 88.0, target_b_color, "Target B", theme)
    add_ship(ax, 6260.0, 1180.0, 180.0, target_c_color, "Target C", theme)

    route = np.array([
        [150.0, -1450.0],
        [950.0, -1320.0],
        [1760.0, -780.0],
        [2720.0, -360.0],
        [3980.0, 120.0],
        [5300.0, 420.0],
        [6750.0, 920.0],
        [7650.0, 1260.0],
    ])
    ax.plot(route[:, 0], route[:, 1], color=theme.accent, linewidth=2.6)
    ax.scatter([7650.0], [1260.0], marker="*", s=180, color=theme.secondary_accent, edgecolor=theme.border, zorder=5)
    ax.text(7650.0, 1440.0, "Goal", ha="center", fontsize=9.5, color=theme.text)
    ax.text(320.0, -1630.0, "Start", ha="left", fontsize=9.5, color=theme.text)

    ax.annotate("窄通道", xy=(4180, 150), xytext=(3280, 520), arrowprops=dict(arrowstyle="->", color=theme.border), fontsize=9.2, color=theme.text)
    ax.annotate("受限机动空间", xy=(6040, 380), xytext=(5600, 1360), arrowprops=dict(arrowstyle="->", color=theme.border), fontsize=9.2, color=theme.text)
    ax.text(4850, -420, "Harbor Conflict Zone", fontsize=9.0, color=theme.risk)
    ax.text(1000, 1450, "Harbor Set", fontsize=9.0, color=theme.current)
    return fig


def draw_fig09_graphical_abstract(theme: Theme):
    fig, ax = canvas(figsize=(13.2, 5.9), facecolor=theme.background)

    panel_x = [0.03, 0.28, 0.54, 0.79]
    panel_w = 0.18
    panel_h = 0.76
    titles = ["Dynamic Maritime\nEnvironment", "Adaptive KEMM\nResponse Engine", "Unified Candidate Pool\nand Selection", "Safe Rolling\nReplanning Outcome"]
    fills = [theme.source_fill, theme.module_fill, theme.candidate_fill, theme.output_fill]
    for x, title, fill in zip(panel_x, titles, fills):
        add_box(ax, x, 0.12, panel_w, panel_h, "", theme, fill=fill, rounding=0.03, lw=1.6)
        ax.text(x + panel_w / 2, 0.82, title, ha="center", va="center", fontsize=12.0, fontweight="bold", color=theme.text)

    for i in range(3):
        add_arrow(ax, (panel_x[i] + panel_w, 0.50), (panel_x[i + 1] - 0.01, 0.50), theme, color=theme.accent, lw=2.1)

    ax.text(0.12, 0.66, "Own ship + targets", fontsize=10.0, color=theme.text, ha="center")
    add_ship_axes(ax, 0.10, 0.52, 15, theme.accent, theme)
    add_ship_axes(ax, 0.16, 0.61, 180, theme.secondary_accent, theme, scale=0.025)
    risk_disc = Circle((0.15, 0.34), 0.05, facecolor="none", edgecolor=theme.risk, linewidth=1.3, transform=ax.transAxes)
    ax.add_patch(risk_disc)
    obstacle = Circle((0.08, 0.30), 0.03, facecolor="#d1d5db" if not theme.is_bw else "#d9d9d9", edgecolor=theme.border, transform=ax.transAxes)
    ax.add_patch(obstacle)
    ax.text(0.12, 0.20, "Crossing / harbor clutter /\nrisk field / obstacles", fontsize=9.2, color=theme.muted_text, ha="center")

    center = (0.37, 0.48)
    add_box(ax, 0.33, 0.39, 0.08, 0.06, "Adaptive\nallocator", theme, fill=theme.background, fontsize=8.8, rounding=0.01)
    ax.text(center[0], 0.33, "Adaptive allocation", fontsize=10.5, color=theme.text, ha="center")
    mech_positions = {
        "Memory": (0.37, 0.69),
        "Prediction": (0.47, 0.48),
        "Transfer": (0.37, 0.23),
        "Reinit": (0.27, 0.48),
    }
    for label, (mx, my) in mech_positions.items():
        circ = Circle((mx, my), 0.045, facecolor=theme.module_fill, edgecolor=theme.border, linewidth=1.4, transform=ax.transAxes)
        ax.add_patch(circ)
        ax.text(mx, my, label, ha="center", va="center", fontsize=9.0, color=theme.text, transform=ax.transAxes)
        add_arrow(ax, center, (mx, my), theme, color=theme.accent, lw=1.6)
    ax.text(0.37, 0.14, "Diagnose change -> allocate budgets", fontsize=9.2, color=theme.muted_text, ha="center")

    add_box(ax, 0.575, 0.64, 0.13, 0.10, "Candidate pool", theme, fill=theme.candidate_fill, fontsize=10.0)
    rng = np.random.default_rng(7)
    for i in range(18):
        circ = Circle((0.60 + float(rng.random()) * 0.08, 0.48 + float(rng.random()) * 0.10), 0.006, facecolor=[theme.accent, theme.secondary_accent, theme.risk][i % 3], edgecolor=theme.border, linewidth=0.4, transform=ax.transAxes)
        ax.add_patch(circ)
    add_box(ax, 0.575, 0.26, 0.13, 0.11, "Constraint-first\nselection", theme, fill=theme.flow_fill, fontsize=9.5)
    add_arrow(ax, (0.64, 0.64), (0.64, 0.56), theme, color=theme.border)
    add_arrow(ax, (0.64, 0.44), (0.64, 0.36), theme, color=theme.border)
    ax.text(0.64, 0.18, "Feasible + diverse + Pareto-efficient", fontsize=9.2, color=theme.muted_text, ha="center")

    route = np.array([
        [0.82, 0.28],
        [0.86, 0.35],
        [0.90, 0.44],
        [0.94, 0.60],
        [0.96, 0.70],
    ])
    ax.plot(route[:, 0], route[:, 1], color=theme.accent, linewidth=2.8, transform=ax.transAxes)
    start = Circle((0.82, 0.28), 0.012, facecolor=theme.background, edgecolor=theme.border, linewidth=1.1, transform=ax.transAxes)
    goal = Circle((0.96, 0.70), 0.014, facecolor=theme.secondary_accent, edgecolor=theme.border, linewidth=1.1, transform=ax.transAxes)
    ax.add_patch(start)
    ax.add_patch(goal)
    pill = FancyBboxPatch((0.865, 0.73), 0.08, 0.05, boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=1.0, edgecolor=theme.border, facecolor=theme.background, transform=ax.transAxes)
    ax.add_patch(pill)
    ax.text(0.905, 0.755, "Risk down", fontsize=9.6, color=theme.text, ha="center", va="center")
    ax.text(0.89, 0.18, "Faster recovery\nstable route", fontsize=10.0, color=theme.text, ha="center")
    ax.text(
        0.50,
        0.04,
        "Dynamic scene -> adaptive knowledge reuse -> unified competition -> safer and more efficient replanning",
        fontsize=11.0,
        color=theme.text,
        ha="center",
        fontweight="bold",
    )
    return fig


def draw_fig10_kemm_principle_chain(theme: Theme):
    fig, ax = canvas(figsize=(14.4, 6.9), facecolor=theme.background)
    add_heading(
        ax,
        "KEMM 原理解释图：从变化感知到统一竞争",
        "这张图面向论文和汇报，强调 KEMM 的核心不是固定算子，而是先诊断变化，再分配预算，再统一竞争筛选。",
        theme,
    )

    stage_x = [0.05, 0.29, 0.53, 0.77]
    stage_w = 0.17
    stage_h = 0.48
    stages = [
        ("变化感知", "读取环境与前沿变化", "提取变化强度\n可迁移性\n预测可靠性", theme.source_fill),
        ("预算分配", "决定四类候选预算", "根据诊断结果分配\nmemory / prediction\ntransfer / reinit", theme.module_fill),
        ("多源候选", "并行提名候选解", "四类机制分别给出\n知识复用或探索候选\n避免单策略失灵", theme.candidate_fill),
        ("统一竞争", "约束优先收束输出", "统一候选池比较\n可行性、Pareto 优势\n与多样性后再执行", theme.output_fill),
    ]

    for x, (title, top, body, fill) in zip(stage_x, stages):
        add_box(ax, x, 0.28, stage_w, stage_h, "", theme, fill=fill, rounding=0.03, lw=1.8)
        ax.text(x + stage_w / 2, 0.67, title, fontsize=13.0, fontweight="bold", color=theme.text, ha="center")
        ax.text(x + stage_w / 2, 0.60, top, fontsize=10.0, color=theme.muted_text, ha="center")
        ax.text(x + stage_w / 2, 0.46, body, fontsize=10.4, color=theme.text, ha="center", va="center", linespacing=1.45)

    for idx in range(3):
        add_arrow(ax, (stage_x[idx] + stage_w, 0.52), (stage_x[idx + 1] - 0.01, 0.52), theme, color=theme.accent, lw=2.2)

    add_box(ax, 0.18, 0.08, 0.64, 0.10, "闭环结果：输出代表轨迹并仅执行前段，环境更新后再次回到变化感知阶段。", theme, fill=theme.flow_fill, rounding=0.03, fontsize=10.5)
    add_arrow(ax, (0.86, 0.28), (0.22, 0.18), theme, rad=0.26, color=theme.guide, linestyle="--")
    ax.text(0.50, 0.02, "一句话理解：KEMM 通过“先判断变化类型，再决定怎么复用知识”的方式提升动态环境下的恢复速度与稳定性。", fontsize=10.4, color=theme.muted_text, ha="center")
    return fig


def draw_fig11_module_role_map(theme: Theme):
    fig, ax = canvas(figsize=(14.5, 8.2), facecolor=theme.background)
    add_heading(
        ax,
        "KEMM 模块职责解释图：四类知识机制分别解决什么问题",
        "这张图不是流程图，而是把 memory、prediction、transfer、reinit 的适用场景、核心动作和收益拆开解释。",
        theme,
    )

    cards = [
        ("Memory", "重复场景 / 熟悉模式", "检索相似历史精英并恢复候选", "适合 recurring harbor、重复港区布局", 0.06, 0.53, theme.module_fill),
        ("Prediction", "平滑漂移 / 连续变化", "预测前沿移动趋势并前瞻采样", "适合 drift profile、目标船渐进偏移", 0.32, 0.53, theme.flow_fill),
        ("Transfer", "结构相似 / 几何偏移", "把相似环境中的知识变换到当前环境", "适合通道宽度变化、障碍群几何偏移", 0.06, 0.21, theme.candidate_fill),
        ("Reinit", "突变 / 旧知识失效", "主动补充探索样本并恢复多样性", "适合 shock profile、突发封航或风险飙升", 0.32, 0.21, theme.output_fill),
    ]

    for title, cond, action, gain, x, y, fill in cards:
        add_box(ax, x, y, 0.22, 0.23, "", theme, fill=fill, rounding=0.03, lw=1.7)
        ax.text(x + 0.11, y + 0.18, title, ha="center", va="center", fontsize=12.8, fontweight="bold", color=theme.text)
        ax.text(x + 0.11, y + 0.13, cond, ha="center", va="center", fontsize=9.8, color=theme.muted_text)
        ax.text(x + 0.11, y + 0.08, action, ha="center", va="center", fontsize=10.2, color=theme.text)
        ax.text(x + 0.11, y + 0.03, gain, ha="center", va="center", fontsize=9.3, color=theme.muted_text)

    add_box(ax, 0.63, 0.25, 0.29, 0.45, "", theme, fill=theme.flow_fill, rounding=0.03, lw=1.8)
    ax.text(0.775, 0.62, "协同原则", ha="center", va="center", fontsize=13.0, fontweight="bold", color=theme.text)
    ax.text(0.775, 0.53, "四模块不是互斥替代，\n而是按变化类型分工提名候选。", ha="center", va="center", fontsize=10.4, color=theme.text, linespacing=1.45)
    ax.text(0.775, 0.40, "最终所有候选仍需进入\n统一候选池并接受\n约束优先的竞争筛选。", ha="center", va="center", fontsize=10.4, color=theme.text, linespacing=1.45)
    ax.text(0.775, 0.28, "因此 KEMM 的本质是\n“多知识来源 + 统一收束机制”。", ha="center", va="center", fontsize=10.4, color=theme.secondary_accent if not theme.is_bw else theme.muted_text, linespacing=1.45)

    add_arrow(ax, (0.54, 0.64), (0.63, 0.58), theme, color=theme.accent, lw=1.8)
    add_arrow(ax, (0.54, 0.58), (0.63, 0.54), theme, color=theme.accent, lw=1.8)
    add_arrow(ax, (0.54, 0.32), (0.63, 0.42), theme, color=theme.secondary_accent if not theme.is_bw else theme.border, lw=1.8)
    add_arrow(ax, (0.54, 0.26), (0.63, 0.38), theme, color=theme.secondary_accent if not theme.is_bw else theme.border, lw=1.8)
    return fig


DRAWERS: dict[str, Callable[[Theme], plt.Figure]] = {
    "fig00_visual_language_board": draw_style_board,
    "fig01_system_architecture": draw_fig01_system_architecture,
    "fig02_method_flow": draw_fig02_method_flow,
    "fig03_change_diagnosis_allocation": draw_fig03_change_diagnosis,
    "fig04_multisource_candidates": draw_fig04_multisource_candidates,
    "fig05_candidate_pool_selection": draw_fig05_candidate_pool,
    "fig06_rolling_replanning": draw_fig06_rolling_replanning,
    "fig07_crossing_scene": draw_fig07_crossing,
    "fig08_harbor_clutter_scene": draw_fig08_harbor,
    "fig09_graphical_abstract": draw_fig09_graphical_abstract,
    "fig10_kemm_principle_chain": draw_fig10_kemm_principle_chain,
    "fig11_module_role_map": draw_fig11_module_role_map,
}


def save_figure(fig: plt.Figure, stem: str, theme: Theme) -> list[Path]:
    saved: list[Path] = []
    for suffix in ("svg", "png", "pdf"):
        path = out_dir(theme, suffix) / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved.append(path)
    plt.close(fig)
    return saved


def themes_for_stem(stem: str) -> tuple[Theme, ...]:
    if stem in {"fig09_graphical_abstract", "fig10_kemm_principle_chain", "fig11_module_role_map"}:
        return (THEME_PAPER,)
    return (THEME_PAPER, THEME_BW)


def write_index(generated: dict[str, list[Path]]) -> None:
    PRIVATE_PATENT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KEMM 船舶专利附图索引与图稿说明详细版",
        "",
        "本文档由 `docs/patent_figures/generate_patent_figures.py` 自动生成，并输出到本地私有目录 `.private/patent_ship_kemm/`。",
        "它不只是文件清单，还用于说明每张图表达的技术含义、关键视觉元素、阅读顺序和论文 / 专利中的使用位置。当前文件名已切换为中文，以便后续直接作为专利材料目录的一部分管理。",
        "",
        "## 输出目录",
        "",
        "- `paper_svg/`：论文版可编辑 SVG 主稿",
        "- `paper_png/`：论文版预览 PNG",
        "- `paper_pdf/`：论文版排版用 PDF",
        "- `patent_bw_svg/`：专利黑白线稿 SVG",
        "- `patent_bw_png/`：专利黑白预览 PNG",
        "- `patent_bw_pdf/`：专利黑白 PDF",
        "",
        "## 使用说明",
        "",
        "- 论文版优先服务于方法表达、汇报展示和论文排版，默认弱化或隐藏专利构件号，减少视觉干扰。",
        "- 专利黑白版优先服务于附图提交，保留必要构件号，要求在灰度打印下仍可区分层级。",
        "- 图1到图8是专利主案核心图号；图9到图11是论文 / 汇报专用解释图，不进入专利黑白版。",
        "- 若后续继续修改图稿，请优先修改 `generate_patent_figures.py` 后再重新导出，避免私有图稿索引与实际图稿脱节。",
        "",
        "## 图稿详解",
        "",
    ]
    for stem in DRAWERS:
        meta = FIGURE_DOCS[stem]
        lines.append(f"### {FIGURE_NAMES[stem]}")
        lines.append("")
        lines.append(f"**图义概述**")
        lines.append("")
        lines.append(meta["summary"])
        lines.append("")
        lines.append(f"**这张图在表达什么**")
        lines.append("")
        for item in meta["meaning"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append(f"**关键视觉元素**")
        lines.append("")
        for item in meta["elements"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append(f"**推荐阅读顺序**")
        lines.append("")
        for item in meta["reading"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append(f"**建议用途**")
        lines.append("")
        lines.append(meta["usage"])
        lines.append("")
        lines.append(f"**相关文档**")
        lines.append("")
        for ref in meta["refs"]:
            lines.append(f"- `{ref}`")
        lines.append("")
        lines.append(f"**输出文件**")
        lines.append("")
        for path in generated.get(stem, []):
            rel = path.relative_to(ROOT).as_posix()
            lines.append(f"- `{rel}`")
        lines.append("")
    PRIVATE_FIGURE_INDEX.write_text("\n".join(lines), encoding="utf-8")


def generate(selected: Iterable[str] | None = None) -> dict[str, list[Path]]:
    configure_fonts()
    ensure_dirs()
    selected_set = set(selected or DRAWERS.keys())
    generated: dict[str, list[Path]] = {}
    for stem, drawer in DRAWERS.items():
        if stem not in selected_set:
            continue
        saved: list[Path] = []
        for theme in themes_for_stem(stem):
            fig = drawer(theme)
            saved.extend(save_figure(fig, stem, theme))
        generated[stem] = saved
    write_index(generated)
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Generate KEMM ship patent figures.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(DRAWERS.keys()),
        help="Only generate the specified figure stems.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = generate(args.only)
    count = sum(len(paths) for paths in generated.values())
    print(f"Generated {len(generated)} figure groups and {count} output files under {ROOT}.")


if __name__ == "__main__":
    main()
