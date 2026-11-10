# Chain of Draft：GSM8K 复现工程

该工程比较 Qwen2.5-3B-Instruct 在 GSM8K 上的直接回答、CoT（Chain of Thought，思维链）、CoD（Chain of Draft，草稿链）和 CoD 0-shot。正式实验采用随机种子 42 抽取的同一批 100 道测试题，四组共 400 次生成。报告依据实际逐条结果计算准确率、输出 token、生成延迟、截断和解析错误。

## 快速开始

> 建议 Python 3.9 及以上；本实验正式运行环境为 Python 3.13.16。

### 方式 A：只重新生成报告（无需 GPU）

仓库已包含 `runs/full/` 的正式运行产物。安装报告依赖后，直接重算：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe report_results.py
```

该命令会重新生成 `report.md`、`figures/full_metrics.png`、`figures/accuracy_tokens.png`、`runs/full/metrics.csv` 和 `runs/full/official_rescore.json`。

### 方式 B：完整重跑（需要 CUDA GPU）

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# 先按 https://pytorch.org/get-started/locally/ 安装与本机 CUDA 匹配的 PyTorch
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.venv\Scripts\python.exe prepare_resources.py
.venv\Scripts\python.exe -m unittest -v test_experiment.py
.venv\Scripts\python.exe experiment.py --config config.json --phase pilot --output runs/pilot
.venv\Scripts\python.exe experiment.py --config config.json --phase full --output runs/full
.venv\Scripts\python.exe report_results.py
```

- `prepare_resources.py` 会下载固定 commit 的官方源码、论文、提示词和 GSM8K 数据；如果仓库已包含这些文件，可改用 `--verify-only`。
- 完整运行在 Tesla T4 上约 26 分钟，共 400 次生成；需要能加载 Qwen2.5-3B-Instruct 的 CUDA 环境。
- 只做报告复算时，不需要 GPU、Transformers 或模型下载。
- `demo.ps1` 只读 `runs/full` 的已保存结果，用于现场演示，不参与实验本身。

## 入口与配置

`config.json` 是唯一正式实验配置。当前配置固定模型 revision、FP16、`cuda:0`、greedy 生成、8-shot、4,096 新 token 上限、16 token 预热，以及 12 题 pilot / 100 题 full。CoD 0-shot 复用 CoD 指令并使用零个示例。模型与数据均以明确修订号固定，官方提示词从冻结源码逐字节复制。

| 文件或目录 | 用途 |
| --- | --- |
| `prepare_resources.py` | 下载、重建并校验官方源码、论文、提示词与完整 GSM8K 数据 |
| `reference/provenance.json` | 官方来源、commit、SHA-256 与请求拼接契约 |
| `data/test.jsonl`、`data/manifest.json` | 原始顺序的 1,319 题测试数据及逐页来源 |
| `prompts/` | 官方三种 8-shot YAML |
| `experiment.py` | CLI 和 Notebook 共用的运行与断点恢复入口 |
| `test_experiment.py` | 评分、提示词、生成计量及持久化测试 |
| `tools/build_notebook.py` | 将冻结输入打包成自包含 Kaggle Notebook |
| `runs/pilot/`、`runs/full/` | 两个独立阶段的 manifest、逐条结果与 summary |
| `report_results.py` | 审计正式结果并生成中文报告及 PNG / SVG 图表 |
| `requirements.txt` / `requirements-gpu.txt` | 报告与资源依赖 / 真实推理依赖 |

命令均在工程根目录执行。

## 环境准备

资源准备依赖 `requests`；本地测试依赖 `PyYAML`；报告依赖 `matplotlib`。`numpy` 随 Matplotlib 的正常安装一并提供。通过独立虚拟环境安装这些依赖：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install requests PyYAML==6.0.3 matplotlib
```

以下命令中的 `python` 指所选环境的解释器。Windows 可直接将其替换为 `.venv\Scripts\python.exe`。

真实模型推理还要求 CUDA GPU、支持 CUDA 的 PyTorch、`transformers`、`tokenizers`、`accelerate` 和 `safetensors`。本工程固定 `transformers==4.57.1`、`accelerate==1.10.1` 和 `PyYAML==6.0.3`。使用与机器 CUDA 环境匹配的 [PyTorch 官方安装方式](https://pytorch.org/get-started/locally/)，然后安装模型加载依赖：

```powershell
python -m pip install transformers==4.57.1 accelerate==1.10.1 PyYAML==6.0.3 tokenizers safetensors
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

本地 Windows Python 环境用于资源准备、测试与报告；CUDA 推理通过实际 GPU 环境运行。首次模型加载需要连接 Hugging Face 下载固定 revision。显存不足、缺少 CUDA、模型 revision 或运行精度不符时，运行会给出明确错误并保存失败上下文。

## 资源重建与校验

```powershell
python prepare_resources.py
python prepare_resources.py --verify-only
```

首条命令获取指定 commit 的官方源码 ZIP、论文 PDF、三个提示词文件，以及 `openai/gsm8k` 的完整 `main/test`。数据接口每页最多获取 100 题，程序逐页核对 `X-Revision`、原始行号、字段、总数与数值 Gold，随后写入 1,319 行 JSONL。该 rows 接口提供当前数据修订；修订号与固定值有差异时程序明确失败。已准备的数据与清单支持离线校验及后续离线读取。

`--verify-only` 根据已存指纹核对资源，覆盖数据行数与 ID 顺序、Gold、源码树、提示词副本和论文文件。`reference/upstream/` 保存官方源码快照；在其他目录做探索与修改可保持该快照的指纹稳定。

## 本地检查与实验运行

```powershell
python -m unittest -v test_experiment.py
python experiment.py --config config.json --phase pilot --output runs/pilot
python experiment.py --config config.json --phase full --output runs/full
```

测试通过模拟生成接口覆盖数据与判分契约、官方提示拼接、CUDA 同步调用、EOS（End of Sequence，序列结束）计数、截断、断点恢复及损坏拒绝，可在缺少 Torch 的环境执行。测试结果与真实模型质量分别由各自输出证明。

每阶段运行首先校验资源与输入指纹，预计算固定样本和方法顺序，加载一次固定 FP16 模型，执行预热，再逐题保存生成结果。Notebook 内 pilot/full 调用复用同一个缓存运行时；CLI 的独立进程分别加载模型。每条结果均保存原文、特殊 token 原文、新 token ID、输入与输出 token 数、延迟、解析状态、正确性和结束原因。

计时包围 CUDA 同步前后的 `model.generate`，tokenizer 与文本 decode 单独执行。输出 token 数包含实际生成的 EOS 等特殊 token。完整输入与输出预算在生成前接受上下文检查。正式报告保留全部错误和截断记录的开销。

## 断点恢复与失败诊断

对同一阶段、同一输出目录重复执行原命令即可恢复。程序核对配置、完整数据、提示词、实现源码、官方来源、采样计划、环境版本与设备元数据；已完成记录按既定顺序验证后跳过。每条结果都 flush 并同步到磁盘。

`runs/<phase>/failure.json` 记录失败阶段、样本/方法上下文、异常信息与 traceback。JSONL 尾部的未提交字节先保存为 SHA 命名的 `.recovery-*.bak` 并生成恢复审计，再修复到完整记录边界；中间记录损坏或已提交 JSON 损坏会明确失败。对运行错误保留原始目录与恢复备份，按记录修复环境后用相同命令继续。

若 `stage=model_load` 报 `Loaded model revision differs: ... actual=None`，检查 Transformers 与 Accelerate 是否为上文固定版本，并在 Notebook 安装完成后重启 kernel，再从环境单元顺序执行。固定模型 revision 的校验在预热与评分前完成；其失败上下文用于定位模型来源元数据问题。

修改正式配置、源码或冻结输入意味着新的实验身份，应选择新的输出目录保存新实验。正式报告入口使用 `runs/full/`；保留旧结果后将待报告的完整正式运行放置于该目录。

## Kaggle Notebook

```powershell
python tools/build_notebook.py
```

该命令将实验脚本、测试、配置、数据、提示词、官方源码和论文打包进 `CoD-Reproduction.ipynb`，同时生成用于当前编辑会话的工具脚本。Notebook 包含五个代码单元：展开并验证冻结输入、运行测试、pilot、full、归档结果。Notebook 使用 Kaggle 已提供的 Torch 环境，bootstrap 固定安装 `transformers==4.57.1`、`accelerate==1.10.1` 和 `PyYAML==6.0.3`；实际依赖版本在运行 manifest 中记录。替换已导入的依赖版本后重启 kernel，再顺序执行 Notebook。

在 Kaggle 导入 Notebook，启用 Internet 与 GPU T4 x2，顺序执行代码单元。实验配置选用 `cuda:0`，模型在第一块 GPU 上运行。完成后下载 `/kaggle/working/cod-results.zip`，将其中 `runs/` 解压回本工程根目录。归档包含逐条输出、摘要、manifest 与 `requirements-freeze.txt`。运行通过以 Notebook 实际输出与完整产物为依据；公开成绩和排名需要独立验证。

## 生成报告

正式运行完成且结果已回收到本机后执行：

```powershell
python report_results.py
```

生成器要求 `runs/full/manifest.json`、`summary.json` 和 `results.jsonl` 同时存在。它校验完整的 4 × 100 条记录、唯一组合、同题样本、执行顺序、输入指纹、原始输出评分及 token ID，重新计算指标并核对 summary；任一差异都会明确失败。校验完成后写出：

- `report.md`：中文独立报告，分列原论文表 6 与本次实测，并解释计量和评分偏差。
- `figures/full_metrics.png`、`figures/full_metrics.svg`：准确率、输出 token 与生成延迟的可导出图。
- `figures/accuracy_tokens.png`、`figures/accuracy_tokens.svg`：token 为横轴、准确率为纵轴的散点图，以不同标记区分严格和官方评分。
- `runs/full/reviewed_examples.json`：固定规则选择的样例及完整四组输出。
- `runs/full/official_rescore.json`：通过 AST 提取冻结官方评分函数，对同一输出与原始数据 Gold 逐条复算，保留严格/官方判定、原序与输出 SHA-256。
- `runs/full/metrics.csv`：四方法的严格/官方正确数与准确率、token、秒制延迟、截断、解析错误和判定变化；准确率采用 0–1 数值比例。

正式表格仅采用 full；pilot 保持独立。主表采用原严格评分，额外官方评分表用于观察判分口径的敏感性。官方复算读入去除特殊 token 后的 `output`，原始数据 `answer` 同样经过官方提取函数；全部计量值与原实验保持一致。选例包含计划前三题，以及计划中首次出现的方法差异或失败类别，规则与样本 ID 一同保留；评分变化样例按正式结果原序选择前三条。报告只在真实正式结果通过校验后生成。

## 来源与使用边界

- 论文：[Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/abs/2502.18600)。
- 官方实现：[sileix/chain-of-draft](https://github.com/sileix/chain-of-draft)，冻结 commit 与文件 SHA 详见 `reference/provenance.json`。
- 数据：[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)。
- 模型：[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。

官方仓库快照中的许可文件情况在生成报告中据实列示。使用与再分发应核对源码、数据和模型各自许可。100 题、小模型、本地 FP16、严格 Decimal 评分和 GPU 生成计时共同限定本次实验范围，完整测试集、论文 API 延迟和其他模型验证需要额外运行证据。

## 许可与再分发

本仓库代码用于课程复现，未单独声明许可证。`reference/paper.pdf` 与 `reference/upstream/` 的版权归原作者；官方仓库快照未提供 LICENSE，公开再分发前请先向作者确认授权。GSM8K 数据、Qwen 模型及第三方依赖分别遵循其原始许可。
