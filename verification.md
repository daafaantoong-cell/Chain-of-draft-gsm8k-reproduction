# 验证与交付记录

验证日期：2026-11-06。工程目录：`C:\Users\24573\Downloads\CoD-Reproduction`。

本次交付为 Chain of Draft 的小规模核心实验：固定 Qwen2.5-3B-Instruct、FP16、GSM8K 测试集 seed 42 的 100 道题，比较直接回答、CoT、CoD 与 CoD 0-shot。正式运行于北京时间 22:54–23:20 完成，共 400 条实际生成；独立 pilot 保存 48 条，题目与正式集前 12 题重叠。

## 检查结果

| 检查 | 状态 | 证据和范围 |
| --- | --- | --- |
| 重点测试 | passed | 本地和 Kaggle 各通过 26 项 unittest |
| 固定资源 | passed | 1,319 题行序、数据哈希、官方源码、逐字节提示词副本及论文哈希 |
| Notebook 输入 | passed | 38 个内嵌输入文件与本机文件逐字节一致，5 个代码单元可编译 |
| 正式记录 | passed | 400 条、100 个唯一题目，四组各 100 条；计划、提示、评分、计量与 summary 一致 |
| 官方评分复算 | passed | 独立执行冻结官方 AST 原函数，400 条输入、Gold、判定和输出哈希一致；13 条判定发生变化 |
| Pilot 与正式前缀 | passed | 48 条题目、提示、token ID、原文、评分一致；各次实际延迟分别保留 |
| 现场 tokenizer 审计 | passed | 主 Agent 在 Kaggle 对全部 448 条记录重做输入 token 计数和两种输出解码；独立验证核对审计日志与结果哈希 |
| 环境和时间 | passed | manifest、完整 pip freeze、执行 Notebook 交叉一致；Tesla T4、cuda:0、FP16、同步生成计时 |
| 报告、CSV 和样例 | passed | 指标独立重算，报告纯函数重渲染除生成时间外一致，20 个本地链接有效 |
| 图表 | passed | 两组 PNG 可读；SVG 的 12 根柱和 8 个散点坐标对应实际指标；柱图明确使用严格评分 |
| 已执行 Notebook | passed | 执行计数 1–5，448 条 DONE 日志与对应记录一致，无 error 输出 |
| 验证期间文件完整性 | passed | 最终冻结后 84 个文件 SHA-256 前后相同；官方源码目录完整 |
| Kaggle 保存 | passed | Quick Save 的 Version #1 显示 Successful，网页显示 Draft saved |
| GPU 会话停止 | passed | 网页显示 off (run a cell to start)，Start session 按钮存在 |
| 模型答题与格式 | failed | 严格评分共 130/400 条正确，74 条解析错误；全部记录保留在对应组分母和成本中 |
| 输出长度上限 | passed | 正式四组截断均为 0，保存 32,371 个输出 token ID |
| 本机 GPU 推理及 tokenizer 复跑 | not run | 本机验证解释器缺少 Torch、Transformers；现场执行证据来自 Kaggle |
| 完整 1,319 题、其他模型及跨硬件重复实验 | not run | 本次范围为固定 100 题与一个模型 |
| PPT、15 分钟录屏及教师选题审核 | not run | 本次交付为可用于制作汇报的实验、代码和报告材料 |

`failed` 的模型答题项表示已记录的真实答题或格式失败。产物一致性、运行完成及复算检查均通过。独立验证使用归档执行证据交叉核对，400 次推理保留为本次实际运行，独立验证阶段采用只读检查。

## 角色、Skill 与文件归属

| Agent | Skill / 职责 | 文件范围和结果 |
| --- | --- | --- |
| `/root` | PDF、Playwright；调查整合、实际浏览器运行、现场 token 审计与交付 | 作业与论文检查、Kaggle Notebook、config、tools、执行归档与本记录；会话已停止 |
| `/root/cod_resources` | `maintain-technical-documentation`；官方资源准备和报告生成 | prepare_resources.py、data、prompts、reference、report_results.py、README、report、CSV 和图；真实正式报告生成通过 |
| `/root/cod_runner` | 已确认范围的单文件实验运行器和局部测试 | experiment.py、test_experiment.py；26 项测试通过；修复结果尾行恢复与实验入口资源校验 |
| `/root/cod_review` | `independent-code-review`，只读 | 审查运行器、测试及报告脚本；最初两项 P2 已修复并复查通过；报告复查无成立问题 |
| `/root/cod_verify` | `independent-verification`，全新上下文、只读 | 执行本地检查与完整实验产物门禁；零文件修改、零依赖安装 |

运行器及辅助脚本为明确文件范围内的研究实现，局部只读检索和辅助工具按通用规则执行；验证覆盖测试、真实 GPU 输出、持久化恢复、数据与评分契约、环境、图表和原始日志。

## 可复现的主要检查

在工程目录执行，`python` 指具备 requests、PyYAML 与 Matplotlib 的解释器。本次本机解释器为 `D:\DevTools\Runtimes\Python\Python314\python.exe`。

```powershell
python -X utf8 -m unittest -v test_experiment
python -X utf8 prepare_resources.py --verify-only
python -B -X utf8 -c "import json; from pathlib import Path; import report_results as r; p=Path.cwd(); v=r.validate_run(p); assert r.official_rescore(v)==json.loads((p/'runs/full/official_rescore.json').read_text(encoding='utf-8')); print('400 records and official rescore passed')"
```

上述检查与报告生成、GPU 推理各有独立入口。重跑推理使用 README 的 CUDA / Kaggle 步骤，正式配置统一读取 config.json。`runs/full/report-generation.log` 保存真实报告生成输出；`runs/token_audit.json` 保存现场解码审计及结果哈希。

## 执行归档与导出说明

Kaggle：[CoD Qwen3B GSM8K Reproduction](https://www.kaggle.com/code/zhiyongpeng/cod-qwen3b-gsm8k-reproduction)。保存的版本名为 `GSM8K100-4methods-20261106`。

Jupyter 内层导出保留空白模板的 Kaggle 环境字段，外层运行设置、实际 CUDA 输出及 manifest 记录为 T4。本次正式执行版仅移除失配的 `metadata.kaggle`；全部源代码、输出、执行计数及其他元数据与原始导出相同。两份原始导出和说明保存在 [debug/notebook-export-note.md](debug/notebook-export-note.md)。

关键 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| runs/full/results.jsonl | `24ac68b433f04211a1a5f99939b4b5241f5c5ede1c7d1cae77d399caabbba5a6` |
| CoD-Reproduction-executed.ipynb | `db65eea5f0f7b9057a6a8f9a96199a2a0b6cc1c0c196ee509c657736be785e9b` |
| report_results.py | `24339e7f1dfe202ced5ae39f9bf5f59ce7bdbbea5c3a7a62d3af95fb31e217f2` |

`tools/package_project.py` 为选定的源码、输入和证据生成 artifact_manifest.json，并逐文件校验 ZIP 内哈希与 ZIP 完整性。传输日志和中途检查点位于工程本地调试目录，交付包收录正式输入、完整结果和必要失败证据。
