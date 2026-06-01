# Computation 转投改稿清单

## 目标

将当前稿件从偏 `Applied Sciences` 的应用金融叙事，重构为更适合 `Computation` 的计算方法论文，核心定位为：

- reproducible computational framework
- multiscale representation of nonstationary risk series
- hybrid forecasting and early-warning pipeline
- public-data and leakage-free evaluation

## 必改项

1. 期刊与前言信息
- 将 LaTeX 模板从 `applsci` 切换到 `computation`
- 重写标题、摘要、关键词、featured application
- 删除正文中对 `Applied Sciences` 的直接提及

2. 引言定位
- 弱化“金融市场重要性”式开头
- 强化四个计算问题：
  - multiscale nonstationarity
  - forecast-warning integration
  - strong-baseline evaluation
  - public-data reproducibility
- 将“股指波动率”改写为计算框架的应用场景，而不是全文主语

3. 方法章节
- 明确方法是 causal public-data computational pipeline
- 保留小波、多尺度、LightGBM、warning 的实际实现，不引入未做的新模型
- 增加 block-wise feature organization、fused representation 和 model-specific loss 的形式化表达
- 处理图像重复：主文只保留必要的结构图

4. 结果与讨论
- 将结果组织为 computational questions，而不是纯金融 horse-race
- 强调 representation design、robustness、cross-horizon stability、warning usefulness
- 讨论中降低纯经济解释比重，增加 computational interpretation

5. 可复现性
- 强化 data/code availability 叙述
- 明确脚本、配置、绘图和 rolling evaluation 都是可审计的
- 后续准备 public replication archive 或 GitHub/Zenodo 包

## 建议追加项

1. Cover letter 重写为 `Computation` 版本
- 对齐 `Computational Social Science`
- 如果投 Special Issue，则显式对齐 systemic risk / early-warning / explainable AI / reproducibility

2. 图表精简
- 主文保留两张方法图
- 其余重复流程图挪到附录或删除
- 若需要补图，优先增加 mechanism / diagnostic figure，而不是再加流程图

3. 可能的补实验
- 国际指数或跨市场外部验证
- 更长窗口的扩展风险数据稳健性
- 公开仓库版 reproducibility rerun

## 本轮已开始执行

- `main.tex`：模板、标题、摘要、关键词、featured application、data availability
- `sections/01_introduction.tex`：转为 computation-first 叙事
- `sections/03_methodology.tex`：增强计算框架表述，删除重复 workflow 图
- `sections/06_discussion.tex`：转为 computational interpretation
