# Notebook 导出与运行环境

`jupyter-model-executed.ipynb` 保存 Jupyter 模型的完整原始导出，`kaggle-downloaded.ipynb` 保存 Kaggle 外层菜单下载的文件。两者的 `metadata.kaggle` 都保留空白模板的 `accelerator=none`、`isGpuEnabled=false`、`isInternetEnabled=false`。

本次外层 Kaggle 会话实际设置为 GPU T4 x2 与 Internet enabled。环境单元打印两块 Tesla T4；正式 manifest 记录 `cuda:0`、`torch.float16`、模型 revision 与 CUDA 版本；全部 448 条输出通过现场 tokenizer 的输入计数与输出解码核对。Kaggle Quick Save 的 Version #1 状态为 Successful。

正式交付的 `CoD-Reproduction-executed.ipynb` 使用通用 Jupyter 元数据，导出时移除上述失配的平台模板字段，全部单元源代码、执行计数及输出逐项保留。真实运行环境以 `runs/full/manifest.json`、环境单元与 `runs/token_audit.json` 交叉记录。重新导入 Kaggle 时按 README 启用 GPU 与 Internet。
