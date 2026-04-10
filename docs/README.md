# 文档导航

这份文件是 `docs/` 目录的首页。

如果你不知道应该先看哪份文档，先看这里，再跳转到对应专题文档。

## 我应该先看哪份

### 1. 我只是想把项目跑起来

按这个顺序：

1. [environment_setup.md](environment_setup.md)
2. [run_commands.md](run_commands.md)
3. [how_to_run.md](how_to_run.md)

### 2. 我想理解仓库结构和改代码入口

按这个顺序：

1. [../README.md](../README.md)
2. [../AGENTS.md](../AGENTS.md)
3. [ai_developer_handoff.md](ai_developer_handoff.md)
4. [codebase_reference.md](codebase_reference.md)

### 3. 我想理解 KEMM 算法本身

按这个顺序：

1. [kemm_reference.md](kemm_reference.md)
2. [formula_audit.md](formula_audit.md)
3. [ship_simulation_reference.md](ship_simulation_reference.md)

### 4. 我想做 ship 场景、物理仿真或实验设计

按这个顺序：

1. [ship_simulation_reference.md](ship_simulation_reference.md)
2. [ship_experiment_playbook.md](ship_experiment_playbook.md)
3. [run_commands.md](run_commands.md)

### 5. 我想出图、写论文或整理图注

按这个顺序：

1. [visualization_guide.md](visualization_guide.md)
2. [figure_catalog.md](figure_catalog.md)
3. [run_commands.md](run_commands.md)

### 6. 我在整理专利材料

核心文档：

- [patent_disclosure_ship_kemm.md](patent_disclosure_ship_kemm.md)
- [patent_specification_ship_kemm_draft.md](patent_specification_ship_kemm_draft.md)
- [patent_claims_ship_kemm_final.md](patent_claims_ship_kemm_final.md)
- [patent_abstract_ship_kemm_draft.md](patent_abstract_ship_kemm_draft.md)
- [patent_figures_plan_ship_kemm.md](patent_figures_plan_ship_kemm.md)
- [patent_claim_support_matrix_ship_kemm.md](patent_claim_support_matrix_ship_kemm.md)
- [patent_embodiment_tables_ship_kemm.md](patent_embodiment_tables_ship_kemm.md)
- [patent_completeness_checklist.md](patent_completeness_checklist.md)
- [patent_figures/README.md](patent_figures/README.md)

## 按主题分类

### 项目入口

- [../README.md](../README.md)：仓库首页，负责项目定位、快速开始和总索引
- [../AGENTS.md](../AGENTS.md)：代码修改边界、真实实现入口和验证命令
- [ai_developer_handoff.md](ai_developer_handoff.md)：新开发者和新 AI 助手的最短接手路径

### 运行与环境

- [environment_setup.md](environment_setup.md)：安装环境、依赖、验证命令
- [run_commands.md](run_commands.md)：中文运行速查表
- [how_to_run.md](how_to_run.md)：英文运行速查表

### 架构与代码

- [codebase_reference.md](codebase_reference.md)：代码级结构说明
- [kemm_reference.md](kemm_reference.md)：KEMM 主线详细说明
- [ship_simulation_reference.md](ship_simulation_reference.md)：ship 子系统详细说明
- [formula_audit.md](formula_audit.md)：公式与实现审查记录

### 实验与图表

- [ship_experiment_playbook.md](ship_experiment_playbook.md)：ship 实验设计思路和 profile 对应关系
- [visualization_guide.md](visualization_guide.md)：图表配置、preset 和输出方式
- [figure_catalog.md](figure_catalog.md)：每张图的含义、图注和论文用途

### 专利与附图

- [patent_disclosure_ship_kemm.md](patent_disclosure_ship_kemm.md)：技术交底书
- [patent_specification_ship_kemm_draft.md](patent_specification_ship_kemm_draft.md)：说明书初稿
- [patent_claims_ship_kemm_final.md](patent_claims_ship_kemm_final.md)：权利要求终稿候选
- [patent_figures_plan_ship_kemm.md](patent_figures_plan_ship_kemm.md)：专利附图方案
- [patent_figures/README.md](patent_figures/README.md)：专利图稿和图义说明

## 推荐阅读路径

### 新用户

1. [../README.md](../README.md)
2. [environment_setup.md](environment_setup.md)
3. [run_commands.md](run_commands.md)

### 新开发者

1. [../README.md](../README.md)
2. [../AGENTS.md](../AGENTS.md)
3. [ai_developer_handoff.md](ai_developer_handoff.md)
4. [codebase_reference.md](codebase_reference.md)

### 写论文

1. [kemm_reference.md](kemm_reference.md)
2. [ship_simulation_reference.md](ship_simulation_reference.md)
3. [visualization_guide.md](visualization_guide.md)
4. [figure_catalog.md](figure_catalog.md)

### 整理专利

1. [patent_disclosure_ship_kemm.md](patent_disclosure_ship_kemm.md)
2. [patent_specification_ship_kemm_draft.md](patent_specification_ship_kemm_draft.md)
3. [patent_claims_ship_kemm_final.md](patent_claims_ship_kemm_final.md)
4. [patent_figures/README.md](patent_figures/README.md)

## 维护约定

- `README.md` 只保留总入口和高层索引
- `docs/README.md` 负责 `docs/` 首页导航
- 运行命令优先集中在 [run_commands.md](run_commands.md) 和 [how_to_run.md](how_to_run.md)
- 图表含义优先集中在 [figure_catalog.md](figure_catalog.md)
- 如果默认图包、默认命令或默认入口变化，请至少同步更新：
  - [../README.md](../README.md)
  - [run_commands.md](run_commands.md)
  - [how_to_run.md](how_to_run.md)
  - [visualization_guide.md](visualization_guide.md)
  - [figure_catalog.md](figure_catalog.md)
