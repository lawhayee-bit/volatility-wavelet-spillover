# Applied Sciences LaTeX 模板使用说明

## 1. 当前模板来源

- 官方 MDPI ACS LaTeX 模板已下载到：
  - `paper/template_download`
- 稿件工作目录已建立到：
  - `paper/manuscript`

## 2. 当前采用的模板设定

- 主文档类：
  - `\documentclass[applsci,article,submit,moreauthors]{Definitions/mdpi}`
- 这说明：
  - 目标期刊固定为 `Applied Sciences`
  - 稿件类型固定为 `article`
  - 当前仍是投稿版 `submit`
  - 当前编译优先切换为 `tectonic/XeTeX` 兼容模式，因此不再显式写 `pdftex`

## 3. 模板中已确认可直接使用的栏目

- `\featuredapplication{...}`
  - 对 `Applied Sciences` 可用，但不是强制项
- `\dataavailability{...}`
  - 建议保留
- `\conflictsofinterest{...}`
  - 建议保留
- `\authorcontributions{...}`
  - MDPI 文章常规要求
- `\abbreviations{...}{...}`
  - 可选，但对本项目有帮助

## 4. 当前稿件结构

- `paper/manuscript/main.tex`
- `paper/manuscript/sections/01_introduction.tex`
- `paper/manuscript/sections/02_data.tex`
- `paper/manuscript/sections/03_methodology.tex`
- `paper/manuscript/sections/04_experimental_design.tex`
- `paper/manuscript/sections/05_results.tex`
- `paper/manuscript/sections/06_discussion.tex`
- `paper/manuscript/sections/07_conclusions.tex`
- `paper/manuscript/supplement/appendix.tex`
- `paper/manuscript/bib/references.bib`

## 5. 当前需要注意的问题

- 模板文件已经准备好，但 conda 自带的 `pdflatex/latexmk` 环境不完整，缺少 `tlpkg` 相关 Perl 模块，因此不再作为主编译路径使用。
- 当前已改用本地独立二进制：
  - `tools/tectonic/tectonic`
- 当前推荐编译命令已经封装为：
  - `scripts/compile_manuscript.sh`
- 经过这一路径，`paper/manuscript/main.pdf` 已可成功生成。

## 6. 后续写作原则

- 正文采用英文写作
- 对用户的过程汇报继续使用中文
- 正文优先写主文结果包：
  - `outputs/full10y_refinedfinal_merged`
- 扩展实验写入稳健性部分：
  - `outputs/plusdata500_merged`
  - `outputs/parkinson300_merged`
