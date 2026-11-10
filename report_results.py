"""Validate real full-run artifacts and produce the Chinese CoD reproduction report."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Union

import experiment as exp
import prepare_resources


METHODS = ("baseline", "cot", "cod", "cod_zero")
LABELS = {"baseline": "Direct (8-shot)", "cot": "CoT (8-shot)", "cod": "CoD (8-shot)", "cod_zero": "CoD (0-shot)"}
CHINESE = {"baseline": "直接回答（8-shot）", "cot": "CoT（8-shot）", "cod": "CoD（8-shot）", "cod_zero": "CoD（0-shot）"}
PAPER_TABLE_6 = {"baseline": (7.2, 3.4), "cot": (59.1, 236.4), "cod": (43.1, 41.2)}


@dataclass
class ValidatedRun:
    """Hold the verified input artifacts, recomputed metrics and their source hashes."""

    root: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    provenance: dict[str, Any]
    data_manifest: dict[str, Any]
    hashes: dict[str, str]


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON object with a path-specific validation error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read required artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def hash_file(path: Path) -> str:
    """Return the SHA-256 fingerprint of an input or generated artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    """Raise a clear validation error when an observed report invariant fails."""
    if not condition:
        raise ValueError(message)


def recompute_summary(rows: list[dict[str, Any]], sample_ids: list[int]) -> dict[str, Any]:
    """Recalculate the formal summary from the validated per-generation measurements."""
    groups = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        count = len(selected)
        correct = sum(row["correct"] for row in selected)
        groups[method] = {
            "N": count,
            "correct": correct,
            "accuracy": correct / count,
            "mean_output_tokens": sum(row["output_tokens"] for row in selected) / count,
            "mean_latency": sum(row["latency_seconds"] for row in selected) / count,
            "truncated": sum(row["truncated"] for row in selected),
            "parse_errors": sum(row["parse_error"] is not None for row in selected),
        }
    return {"phase": "full", "sample_ids": sample_ids, "total_records": len(rows), "methods": groups,
            "scoring_policy": exp.SCORING_POLICY, "latency_unit": "seconds"}


def check_summary(actual: object, expected: object, path: str = "summary") -> None:
    """Require exact persisted summary fields and numerically equivalent recomputed metrics."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Summary fields differ at {path}")
        for key, value in expected.items():
            check_summary(actual[key], value, f"{path}.{key}")
    elif type(expected) is float:
        require(type(actual) in {int, float} and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"Summary metric differs at {path}: saved={actual!r}, recomputed={expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"Summary value differs at {path}: saved={actual!r}, recomputed={expected!r}")


def validate_run(root: Path) -> ValidatedRun:
    """Validate 400 real full-run rows, frozen inputs, model metadata and summary consistency."""
    full = root / "runs" / "full"
    paths = {name: full / name for name in ("manifest.json", "summary.json", "results.jsonl")}
    for path in paths.values():
        require(path.is_file(), f"Formal full-run artifact is missing: {path}")
    config = exp.load_config(root / "config.json")
    require(config["sample_size"] == 100 and config["few_shot_count"] == 8,
            "This report requires the approved 100-question, 8-shot experiment")
    manifest = read_json(paths["manifest.json"])
    require(manifest.get("phase") == "full" and manifest.get("completed") is True,
            "The formal full-run manifest must have phase=full and completed=true")
    require(Path(prepare_resources.ROOT).resolve() == root, "Report root must match the frozen resource root")
    verification = prepare_resources.verify_local_resources()
    data = exp.load_data(root / "data" / "test.jsonl")
    prompts = exp.load_prompts(root, config["few_shot_count"])
    plan = exp.make_plan(data, config, "full")
    identity = exp.make_identity(root, root / "config.json", config, "full", plan)
    identity["resource_verification"] = verification
    for key, value in identity.items():
        require(manifest.get(key) == value, f"Frozen run identity mismatch: manifest.{key}")
    sample_ids = identity["sample_ids"]
    require(len(sample_ids) == 100 and len(set(sample_ids)) == 100, "Expected 100 unique paired sample IDs")
    runtime = manifest.get("runtime")
    require(isinstance(runtime, dict), "Completed run lacks runtime metadata")
    for key in ("gpu_name", "gpu_total_memory_bytes", "python", "platform", "versions", "timing", "generation"):
        require(key in runtime, f"Runtime metadata lacks {key}")
    require(runtime.get("model_id") == config["model_id"] and runtime.get("model_revision") == config["model_revision"],
            "Runtime model identity differs from configuration")
    require(runtime.get("dtype") == "torch.float16" and runtime.get("device") == "cuda:0",
            "Report requires the recorded FP16 cuda:0 runtime")
    require(runtime["generation"] == {"do_sample": False, "temperature_argument": "omitted", "input_truncation": False},
            "Runtime generation contract differs from the approved greedy run")
    require(isinstance(manifest.get("warmups"), list) and manifest["warmups"]
            and all(isinstance(warmup, dict) and warmup.get("status") in {"started", "completed"}
                    for warmup in manifest["warmups"])
            and any(warmup["status"] == "completed" for warmup in manifest["warmups"]),
            "Formal run requires at least one completed recorded warmup")
    content = paths["results.jsonl"].read_bytes()
    require(content.endswith(b"\n"), "Results JSONL ends with an uncommitted line; resume the experiment first")
    rows = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed results.jsonl:{line_number}: {error}") from error
        require(isinstance(row, dict) and set(row) == exp.RECORD_KEYS and row["schema_version"] == 1,
                f"Result schema mismatch at line {line_number}")
        rows.append(row)
    require(len(rows) == 400, f"Expected exactly 4 × 100 results; found {len(rows)}")
    jobs = exp.build_jobs(data, plan)
    seen = set()
    for index, (row, job) in enumerate(zip(rows, jobs, strict=True)):
        sample_index, question, method = job
        label = f"results.jsonl:{index + 1}"
        key = (row["id"], row["method"])
        require(key not in seen, f"Duplicate sample/method at {label}: {key}")
        seen.add(key)
        expected = ("full", sample_index, question["id"], method, question["question"], exp.parse_gold(question["answer"]))
        require(tuple(row[field] for field in ("phase", "sample_index", "id", "method", "question", "gold")) == expected,
                f"Sample, method order, question or gold differs at {label}")
        require(type(row["id"]) is int and type(row["sample_index"]) is int, f"Invalid sample identity type at {label}")
        prompt = exp.job_prompt(prompts, method, question["question"], config["few_shot_count"])
        require(row["prompt_sha256"] == exp.sha256_text(prompt), f"Prompt fingerprint mismatch at {label}")
        require(all(isinstance(row[field], str) for field in ("output", "raw_output", "started_at", "completed_at")),
                f"Invalid saved output or timestamp type at {label}")
        parsed, error = exp.parse_prediction(row["output"])
        require(type(row["correct"]) is bool and type(row["truncated"]) is bool, f"Invalid boolean flags at {label}")
        require((row["parsed_answer"], row["parse_error"], row["correct"]) == (parsed, error, error is None and parsed == row["gold"]),
                f"Saved scoring disagrees with raw decoded output at {label}")
        require(type(row["input_tokens"]) is int and row["input_tokens"] >= 1
                and type(row["output_tokens"]) is int and 1 <= row["output_tokens"] <= config["max_new_tokens"],
                f"Invalid token counts at {label}")
        require(isinstance(row["output_token_ids"], list) and len(row["output_token_ids"]) == row["output_tokens"]
                and all(type(token) is int and token >= 0 for token in row["output_token_ids"]),
                f"Token IDs disagree with output-token count at {label}")
        require(type(row["latency_seconds"]) in {int, float} and math.isfinite(row["latency_seconds"])
                and row["latency_seconds"] >= 0, f"Invalid latency at {label}")
        require(row["end_reason"] in {"eos", "length"} and row["truncated"] == (row["end_reason"] == "length"),
                f"Stop reason disagrees with truncation at {label}")
        require(not row["truncated"] or row["output_tokens"] == config["max_new_tokens"],
                f"Truncated generation lacks the full output budget at {label}")
    for method in METHODS:
        ids = [row["id"] for row in rows if row["method"] == method]
        require(ids == sample_ids, f"Method {method} lacks the same 100 ordered paired samples")
    recalculated = recompute_summary(rows, sample_ids)
    check_summary(read_json(paths["summary.json"]), recalculated)
    provenance = read_json(root / "reference" / "provenance.json")
    data_manifest = read_json(root / "data" / "manifest.json")
    hashes = {str(path.relative_to(root)).replace("\\", "/"): hash_file(path) for path in paths.values()}
    return ValidatedRun(root, manifest, recalculated, rows, provenance, data_manifest, hashes)


def relative_percent(value: float, reference: float) -> str:
    """Render a signed relative change, with an explicit zero-denominator outcome."""
    return f"{(value / reference - 1) * 100:+.2f}%" if reference else "参照值为零，比例未定义"


def comparison_sentence(groups: dict[str, Any], method: str, reference: str) -> str:
    """Describe measured token, latency and accuracy changes against one paired group."""
    target, base = groups[method], groups[reference]
    return (f"{CHINESE[method]} 相对 {CHINESE[reference]}：平均输出 token 变化 "
            f"{relative_percent(target['mean_output_tokens'], base['mean_output_tokens'])}，"
            f"平均生成延迟变化 {relative_percent(target['mean_latency'], base['mean_latency'])}，"
            f"准确率变化 {(target['accuracy'] - base['accuracy']) * 100:+.2f} 个百分点。")


def load_official_scorer(root: Path) -> tuple[Callable[[str], str], Callable[[str, str], bool]]:
    """Execute only the three scoring AST definitions extracted from the frozen official source."""
    task_path = root / "reference" / "upstream" / "tasks" / "gsm8k.py"
    utility_path = root / "reference" / "upstream" / "utils.py"
    task_tree = ast.parse(task_path.read_text(encoding="utf-8"), filename=str(task_path))
    utility_tree = ast.parse(utility_path.read_text(encoding="utf-8"), filename=str(utility_path))
    classes = [node for node in task_tree.body if isinstance(node, ast.ClassDef) and node.name == "GSM8K"]
    require(len(classes) == 1, "Frozen source must define exactly one GSM8K class")
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)
               and node.name in {"extract_answer", "equal"}]
    utilities = [node for node in utility_tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "extract_number_from_string"]
    require({node.name for node in methods} == {"extract_answer", "equal"} and len(methods) == 2
            and len(utilities) == 1, "Frozen official scoring function contract differs")
    module = ast.Module(body=[utilities[0], *methods], type_ignores=[])
    namespace = {"re": re, "Union": Union}
    exec(compile(module, "frozen_official_gsm8k_scoring", "exec"), namespace)
    scorer_type = type("FrozenOfficialGSM8KScorer", (), {
        "extract_answer": namespace["extract_answer"], "equal": namespace["equal"],
    })
    scorer = scorer_type()
    return scorer.extract_answer, scorer.equal


def official_rescore(run: ValidatedRun) -> dict[str, Any]:
    """Rescore each saved decoded output and original dataset answer with the exact official functions."""
    extract_answer, equal = load_official_scorer(run.root)
    dataset = {item["id"]: item for item in exp.load_data(run.root / "data" / "test.jsonl")}
    rows = []
    for result_index, row in enumerate(run.rows):
        original_answer = dataset[row["id"]]["answer"]
        predicted = extract_answer(row["output"])
        expected = extract_answer(original_answer)
        correct = equal(predicted, expected)
        require(type(correct) is bool, f"Official scorer returned a non-boolean at result index {result_index}")
        rows.append({
            "result_index": result_index, "phase": row["phase"], "sample_index": row["sample_index"],
            "id": row["id"], "method": row["method"],
            "output_sha256": exp.sha256_text(row["output"]),
            "raw_output_sha256": exp.sha256_text(row["raw_output"]),
            "dataset_answer_sha256": exp.sha256_text(original_answer),
            "strict_parsed_answer": row["parsed_answer"], "strict_parse_error": row["parse_error"],
            "strict_correct": row["correct"], "official_predicted_answer": predicted,
            "official_expected_answer": expected, "official_correct": correct,
            "changed": correct != row["correct"],
        })
    groups = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        count = len(selected)
        strict_correct = sum(row["strict_correct"] for row in selected)
        official_correct = sum(row["official_correct"] for row in selected)
        groups[method] = {
            "N": count, "strict_correct": strict_correct, "strict_accuracy": strict_correct / count,
            "official_correct": official_correct, "official_accuracy": official_correct / count,
            "changed": sum(row["changed"] for row in selected),
            "strict_only": sum(row["strict_correct"] and not row["official_correct"] for row in selected),
            "official_only": sum(row["official_correct"] and not row["strict_correct"] for row in selected),
        }
    return {
        "schema_version": 1, "phase": "full", "total_records": len(rows),
        "sample_ids": run.summary["sample_ids"], "source_hashes": run.hashes,
        "official_source": {
            "repository": run.provenance["source"]["repository"],
            "commit": run.provenance["source"]["commit"],
            "functions": ["GSM8K.extract_answer", "GSM8K.equal", "extract_number_from_string"],
            "execution": "Compile and execute the original AST function definitions; official modules are read as text.",
            "files": {name: hash_file(run.root / name) for name in
                      ("reference/upstream/tasks/gsm8k.py", "reference/upstream/utils.py")},
        },
        "prediction_input": "results.jsonl output: model text decoded with skip_special_tokens=True",
        "expected_input": "Original data/test.jsonl answer passed through the same official extract_answer",
        "hash_encoding": "Exact UTF-8 text bytes; SHA-256",
        "methods": groups, "changed_total": sum(row["changed"] for row in rows), "records": rows,
    }


def official_rescore_markdown(run: ValidatedRun, rescored: dict[str, Any]) -> str:
    """Describe the official sensitivity table and first three changed verdicts in saved result order."""
    lines = ["## 官方评分敏感性复算", "",
             "以下复算直接执行冻结官方源码的 `GSM8K.extract_answer`、`GSM8K.equal` 和 "
             "`utils.extract_number_from_string`。程序以 AST（Abstract Syntax Tree，抽象语法树）提取三个原始函数，"
             "源码模块仅作为文本读取。预测输入为每条保存的 `output`，对应去除特殊 token 后的模型回复文本；"
             "原始数据集 `answer` 也经过同一官方提取函数，保留其原有数值文本。主表及原 summary 继续采用严格评分。", "",
             "| 方法 | N | 严格准确率 | 官方准确率 | 官方−严格（百分点） | 判定变化数 | 仅严格正确 | 仅官方正确 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        group = rescored["methods"][method]
        lines.append(f"| {CHINESE[method]} | {group['N']} | {group['strict_accuracy'] * 100:.1f}% | "
                     f"{group['official_accuracy'] * 100:.1f}% | "
                     f"{(group['official_accuracy'] - group['strict_accuracy']) * 100:+.1f} | "
                     f"{group['changed']} | {group['strict_only']} | {group['official_only']} |")
    cot, cod = rescored["methods"]["cot"], rescored["methods"]["cod"]
    lines.extend(["", f"官方评分下，CoD 8-shot 相对 CoT 8-shot 的准确率变化为 "
                  f"{(cod['official_accuracy'] - cot['official_accuracy']) * 100:+.2f} 个百分点。"
                  "这一评分表采用与论文实现一致的判定函数，提供更接近论文评分口径的比较；"
                  "100 题抽样、本地 FP16 greedy、输入与计量差异继续限制可比性。评分复算复用原生成，"
                  "因此 token、延迟和截断与主表完全相同。", "",
                  "官方评分的固有限制包括：优先取第一个 `####` 区段；删除逗号、美元符号和百分号；"
                  "完整字符串相等优先，否则取首个无符号数值并尝试 float 相等。重复同值数值或含正确首数值的过程可能获判正确，"
                  "中间值也可能导致误判；负号在后备数值提取中丢失，无分组长整数可能只匹配前三位，"
                  "float 比较还会受到数值精度影响。严格评分与官方评分均有明确适用边界。", "",
                  f"全部 {rescored['total_records']} 条复算中共 {rescored['changed_total']} 条判定发生变化。"
                  "[official_rescore.json](runs/full/official_rescore.json) 保持正式 JSONL 原序，"
                  "逐条记录双方判定、官方预测与 Gold、输出及原始特殊 token 文本 SHA-256、原数据答案 SHA-256。", "",
                  "以下按正式 JSONL 原序选择最先出现的三条判定变化记录；变化不足三条时展示全部，零变化时此类样例为空。"])
    for row in [item for item in rescored["records"] if item["changed"]][:3]:
        original = run.rows[row["result_index"]]
        lines.extend(["", f"### 评分变化：记录 {row['result_index']}，ID {row['id']}，{CHINESE[row['method']]}", "",
                      f"题目：{original['question']}", "",
                      f"严格判定：`{row['strict_correct']}`，解析错误：`{row['strict_parse_error']}`；"
                      f"官方判定：`{row['official_correct']}`，官方 Gold：`{row['official_expected_answer']}`。", "",
                      "官方提取的预测文本：", "", fenced_excerpt(row["official_predicted_answer"]), "",
                      "保存的模型回复摘录：", "", fenced_excerpt(original["output"])])
    return "\n".join(lines)


def write_metrics_csv(run: ValidatedRun, rescored: dict[str, Any]) -> str:
    """Export the four validated method summaries with numeric accuracy ratios and second-based latency."""
    path = run.root / "runs" / "full" / "metrics.csv"
    columns = ("method", "N", "strict_correct", "strict_accuracy", "official_correct", "official_accuracy",
               "mean_output_tokens", "mean_latency_seconds", "truncated", "parse_errors", "verdict_changes",
               "strict_only_correct", "official_only_correct")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for method in METHODS:
            strict, official = run.summary["methods"][method], rescored["methods"][method]
            require(strict["N"] == official["N"] and strict["correct"] == official["strict_correct"],
                    f"CSV strict and official summary identities differ for {method}")
            writer.writerow({
                "method": method, "N": strict["N"], "strict_correct": strict["correct"],
                "strict_accuracy": strict["accuracy"], "official_correct": official["official_correct"],
                "official_accuracy": official["official_accuracy"], "mean_output_tokens": strict["mean_output_tokens"],
                "mean_latency_seconds": strict["mean_latency"], "truncated": strict["truncated"],
                "parse_errors": strict["parse_errors"], "verdict_changes": official["changed"],
                "strict_only_correct": official["strict_only"], "official_only_correct": official["official_only"],
            })
    return path.relative_to(run.root).as_posix()


def make_figures(run: ValidatedRun, rescored: dict[str, Any]) -> list[str]:
    """Export the three-metric bars and accuracy-token sensitivity scatter as PNG and SVG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    groups = run.summary["methods"]
    colors = ["#7b8794", "#3578b5", "#d67d2f", "#6b9b76"]
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), layout="constrained")
    metrics = (("accuracy", "Strict accuracy (%)", 100), ("mean_output_tokens", "Mean generated tokens", 1),
               ("mean_latency", "Mean generation latency (s)", 1))
    x = list(range(4))
    for axis, (metric, title, scale) in zip(axes, metrics, strict=True):
        values = [groups[method][metric] * scale for method in METHODS]
        bars = axis.bar(x, values, color=colors, width=0.65)
        axis.set_xticks(x, ["Direct\n8-shot", "CoT\n8-shot", "CoD\n8-shot", "CoD\n0-shot"])
        axis.set_title(title, fontsize=12, pad=14)
        axis.set_ylim(0, 110 if metric == "accuracy" else max(values + [1]) * 1.22)
        axis.bar_label(bars, labels=[f"{value:.1f}" if metric != "mean_latency" else f"{value:.3f}" for value in values],
                       padding=5, fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("GSM8K: paired 100-question local FP16 greedy reproduction", fontsize=14)
    figure.supxlabel("Same 100 questions per method; generated tokens include EOS; synchronized model.generate timing", fontsize=10)
    directory = run.root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"full_metrics.{suffix}"
        figure.savefig(path, dpi=220, facecolor="white")
        paths.append(path.relative_to(run.root).as_posix())
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9.0, 6.0), layout="constrained")
    offsets = {"baseline": (10, -22), "cot": (-85, 15), "cod": (10, 20), "cod_zero": (10, -34)}
    for method, color in zip(METHODS, colors, strict=True):
        tokens = groups[method]["mean_output_tokens"]
        strict_accuracy = groups[method]["accuracy"] * 100
        official_accuracy = rescored["methods"][method]["official_accuracy"] * 100
        axis.plot([tokens, tokens], [strict_accuracy, official_accuracy], color=color, linestyle="--", alpha=0.7)
        axis.scatter([tokens], [strict_accuracy], color=color, marker="o", s=70, zorder=3)
        axis.scatter([tokens], [official_accuracy], facecolors="none", edgecolors=color, marker="D",
                     s=140, linewidths=1.8, zorder=4)
        axis.annotate(LABELS[method], (tokens, max(strict_accuracy, official_accuracy)), xytext=offsets[method],
                      textcoords="offset points", fontsize=11, color=color,
                      arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.6})
    axis.set(xlabel="Mean generated tokens (including EOS)", ylabel="Accuracy (%)",
             title="Accuracy vs generated tokens: scoring sensitivity", ylim=(0, 110),
             xlim=(0, max(groups[method]["mean_output_tokens"] for method in METHODS) * 1.35))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2)
    axis.legend(handles=[Line2D([], [], color="#39424e", marker="o", linestyle="None", markersize=7,
                                label="Strict scorer"),
                         Line2D([], [], color="#39424e", marker="D", markerfacecolor="none", linestyle="None",
                                markersize=9, label="Official scorer")], loc="upper right", frameon=False)
    figure.supxlabel("Paired 100-question sample; scoring reuses the same generated outputs", fontsize=10)
    for suffix in ("png", "svg"):
        path = directory / f"accuracy_tokens.{suffix}"
        figure.savefig(path, dpi=220, facecolor="white")
        paths.append(path.relative_to(run.root).as_posix())
    plt.close(figure)
    return paths


def select_examples(run: ValidatedRun) -> list[dict[str, Any]]:
    """Select the first three planned IDs and first ordered disagreement or failure categories."""
    sample_ids = run.summary["sample_ids"]
    by_id = {sample_id: {row["method"]: row for row in run.rows if row["id"] == sample_id} for sample_id in sample_ids}
    reasons = {sample_id: ["计划中的前三题"] for sample_id in sample_ids[:3]}
    rules = (
        ("计划顺序中首个 CoT/CoD 正确性差异", lambda rows: rows["cot"]["correct"] != rows["cod"]["correct"]),
        ("计划顺序中首个任一方法错误", lambda rows: any(not row["correct"] for row in rows.values())),
        ("计划顺序中首个解析错误", lambda rows: any(row["parse_error"] is not None for row in rows.values())),
        ("计划顺序中首个截断", lambda rows: any(row["truncated"] for row in rows.values())),
    )
    for reason, predicate in rules:
        for sample_id in sample_ids:
            if predicate(by_id[sample_id]):
                reasons.setdefault(sample_id, []).append(reason)
                break
    return [{"id": sample_id, "selection_reasons": reasons[sample_id],
             "records": [by_id[sample_id][method] for method in METHODS]} for sample_id in sample_ids if sample_id in reasons]


def fenced_excerpt(text: str, limit: int = 1200) -> str:
    """Render an explicitly shortened output excerpt inside a collision-safe Markdown code fence."""
    shortened = text[:limit]
    if len(text) > limit:
        shortened += f"\n[报告摘录至前 {limit} 字符；完整输出保存在 results.jsonl 与 reviewed_examples.json]"
    longest = max((len(match.group()) for match in re.finditer(r"`+", shortened)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{shortened}\n{fence}"


def example_markdown(examples: list[dict[str, Any]]) -> str:
    """Build readable paired examples with explicit selection rules and observed failure flags."""
    sections = []
    for example in examples:
        records = {row["method"]: row for row in example["records"]}
        first = records["cot"]
        lines = [f"### 样本 ID {example['id']}", "", f"选例依据：{'；'.join(example['selection_reasons'])}。", "",
                 f"题目：{first['question']}", "", f"Gold：`{first['gold']}`。", "",
                 "| 方法 | 解析答案 | 正确 | 输出 token | 延迟（秒） | 截断 | 解析错误 |",
                 "| --- | --- | --- | ---: | ---: | --- | --- |"]
        for method in METHODS:
            row = records[method]
            lines.append(f"| {CHINESE[method]} | {row['parsed_answer']} | {row['correct']} | {row['output_tokens']} | "
                         f"{row['latency_seconds']:.4f} | {row['truncated']} | {row['parse_error']} |")
        for method in METHODS:
            row = records[method]
            if method in {"cot", "cod"} or not row["correct"] or row["truncated"] or row["parse_error"] is not None:
                lines.extend(["", f"{CHINESE[method]} 的模型输出摘录：", "", fenced_excerpt(row["output"])])
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def render_report(run: ValidatedRun, figures: list[str], examples: list[dict[str, Any]], rescored: dict[str, Any]) -> str:
    """Render the standalone Chinese report using only validated measured results."""
    groups = run.summary["methods"]
    config, runtime = run.manifest["config"], run.manifest["runtime"]
    source, paper = run.provenance["source"], run.provenance["paper"]
    versions = "、".join(f"{name}={version}" for name, version in runtime["versions"].items())
    paper_rows = "\n".join(f"| {CHINESE[method]} | {accuracy:.1f}% | {tokens:.1f} |" for method, (accuracy, tokens) in PAPER_TABLE_6.items())
    measured_rows = "\n".join(
        f"| {CHINESE[method]} | {groups[method]['N']} | {groups[method]['correct']} | {groups[method]['accuracy'] * 100:.1f}% | "
        f"{groups[method]['mean_output_tokens']:.2f} | {groups[method]['mean_latency']:.4f} | {groups[method]['truncated']} | {groups[method]['parse_errors']} |"
        for method in METHODS)
    paired = {sample_id: {row["method"]: row for row in run.rows if row["id"] == sample_id} for sample_id in run.summary["sample_ids"]}
    cot_only = sum(item["cot"]["correct"] and not item["cod"]["correct"] for item in paired.values())
    cod_only = sum(item["cod"]["correct"] and not item["cot"]["correct"] for item in paired.values())
    wrong = sum(not row["correct"] for row in run.rows)
    parse_errors = sum(row["parse_error"] is not None for row in run.rows)
    truncated = sum(row["truncated"] for row in run.rows)
    error_types = dict(Counter(row["parse_error"] for row in run.rows if row["parse_error"] is not None))
    fingerprints = "\n".join(f"- `{name}`：`{digest}`" for name, digest in run.hashes.items())
    sample_ids = ", ".join(str(sample_id) for sample_id in run.summary["sample_ids"])
    all_correct_status = "passed" if not wrong else "failed"
    parse_status = "passed" if not parse_errors else "failed"
    truncated_status = "passed" if not truncated else "failed"
    license_files = [item["path"] for item in source["files"] if Path(item["path"]).name.lower().startswith(("license", "copying"))]
    license_note = (f"冻结源码中的许可文件：{', '.join(license_files)}。" if license_files else
                    "冻结的官方仓库快照包含 26 个文件；其中未提供 LICENSE 或 COPYING 文件。源码使用权限应向作者确认，来源引用本身不构成许可授权。")
    return f"""# Chain of Draft：GSM8K 本地复现实验

报告生成时间：{datetime.now(timezone.utc).isoformat()}。本报告从完整正式运行的逐条结果重新计算指标；正式实验包含同一批 100 道题的四种策略，共 400 次生成。全部结论以本次真实记录为依据。

## 论文与研究动机

论文为 *Chain of Draft: Thinking Faster by Writing Less*（arXiv:2502.18600）。CoT（Chain of Thought，思维链）通过逐步推理帮助求解，但输出推理文本消耗 token 与生成时间。CoD（Chain of Draft，草稿链）要求每个思考步骤仅保留最少草稿，并以最多 5 个英文词为目标，通过更短的外显推理探索准确率与计算开销的关系。该限制由提示词表达；本实验保存实际模型输出，以观察模型的遵循程度。

原论文表 6 的 Qwen2.5-3B-Instruct 结果如下。此表严格引用原论文，作用是提供研究背景；本次 100 题本地结果独立列示。

| 原论文表 6：方法 | 准确率 | 平均输出 token |
| --- | ---: | ---: |
{paper_rows}

论文：[arXiv PDF]({paper['requested_url']})；本地副本：[reference/paper.pdf](reference/paper.pdf)。

## 方法、数据与实验范围

直接回答、CoT 和 CoD 使用官方仓库的完整 8-shot 示例；CoD 0-shot 使用同一 CoD 指令并将示例数设为零。官方 `system_prompt`、示例与问题按照 `utils.py` 的换行规则拼成一段文本，再作为单条 `user` 消息传入模型 chat template。CoT 和 CoD 提示模型在最终答案前输出 `####`；直接回答提示仅输出数值。

数据为 `openai/gsm8k` 的 `main/test`，完整本地数据集含 {run.data_manifest['row_count']} 题。数据修订号为 `{run.data_manifest['revision']}`，JSONL SHA-256 为 `{run.data_manifest['sha256']}`。原始 `row_idx` 保留为整数 `id`。随机种子 {config['seed']} 对完整索引洗牌后取前 {config['sample_size']} 题；每题内的四种方法顺序也通过同一确定性计划随机化。四组共享题目，组间比较采用同题配对。

正式表格仅统计 `runs/full/`。Pilot 配置为 {config['pilot_size']} 题，用于前置流程检查，其结果独立保存并保持在正式汇总之外。100 题覆盖完整测试集的 {config['sample_size'] / run.data_manifest['row_count'] * 100:.2f}%；准确率每变化 1 题即变化 1 个百分点，抽样不确定性限制了对完整基准和其他模型的推广。

## 实际运行环境与计量口径

| 项目 | 运行记录 |
| --- | --- |
| 模型 | `{runtime['model_id']}` |
| 模型 revision | `{runtime['model_revision']}` |
| 精度、设备 | `{runtime['dtype']}`、`{runtime['device']}` |
| GPU | {runtime['gpu_name']}；显存 {runtime['gpu_total_memory_bytes'] / 1024**3:.2f} GiB |
| Python | {runtime['python']} |
| 系统 | {runtime['platform']} |
| CUDA | {runtime.get('cuda_version')} |
| 上下文上限 | {runtime.get('context_limit')} token |
| 输出预算 | 每次最多 {config['max_new_tokens']} 个新 token |
| 生成方式 | batch=1、greedy、`do_sample=False`；temperature 参数省略 |
| 依赖版本 | {versions} |
| 预热 | 共 {len(run.manifest['warmups'])} 次记录，其中 {sum(item['status'] == 'completed' for item in run.manifest['warmups'])} 次完成；配置预算 {config['warmup_tokens']} token/次 |

运行记录的计时定义为：`{runtime['timing']}`。延迟从 CUDA 同步后的 `model.generate` 调用开始，到返回并再次 CUDA 同步结束，包含输入处理阶段和逐 token 解码，tokenizer 与文本 decode 位于计时区间之外。该本地生成计时与原论文远程 API 端到端响应延迟具有不同边界；GPU 型号、运行时负载及不同长度的 few-shot 输入也会影响结果。预热输出独立记录并从正式均值中排除。

输出 token 数直接取生成的新 token ID 数量，包含 EOS（End of Sequence，序列结束）等实际生成的特殊 token；报告展示的可读文本通过 `skip_special_tokens=True` 解码。输入保留完整提示词，执行前检查输入长度加完整输出预算是否处于模型上下文内。预算耗尽且结尾缺少 EOS 的记录标记为 `truncated=true`。

评分使用 `{run.summary['scoring_policy']['name']}`：选取最后一个 `####` 后的文本；选定文本须恰好包含一个带符号数值，并使用 Decimal（十进制定点数）比较。缺少分隔符时选取完整输出，以相同数值规则解析。Gold 的最终分隔区须为完整数值。官方代码使用第一个分隔区，并通过字符串、首个无符号数值和 float 比较判分；本实验对多值或歧义输出采用更严格的显式解析错误。因此，评分差异也是与原论文结果之间的偏差来源。

## 可复现步骤与输入审计

1. 执行 `python prepare_resources.py --verify-only`，核对官方源码、提示词、论文及完整数据的指纹。
2. 执行 `python -m unittest -v test_experiment.py`，检查提示词、判分、计量和持久化契约；测试运行证据由对应执行环境保留。
3. 在 CUDA 环境执行 `python experiment.py --config config.json --phase pilot --output runs/pilot`。
4. 执行 `python experiment.py --config config.json --phase full --output runs/full`，生成四组同题正式记录。
5. 执行 `python report_results.py`，要求正式 manifest 已完成、400 个样本/方法组合唯一、四组题目相同、原始结果评分一致，并将独立重算指标与 `summary.json` 核对后生成本报告。

正式样本 ID 按计划顺序为：`{sample_ids}`。

输入文件指纹：

{fingerprints}

## 运行调试与失败留档

首次 Kaggle 尝试使用现场预装的 Transformers 5.20.2。2026-11-06 14:44:38 UTC，pilot 在 `model_load` 阶段触发模型 revision 校验：期望 `{config['model_revision']}`，实际读取 `model.config._commit_hash=None`，抛出 `RuntimeError`。归档 manifest 的 `runtime=null`、`warmups=[]`、`completed=false` 表明失败发生在预热、生成和评分之前；该尝试保留为环境调试证据。

归档文件为 [failure.json](debug/transformers5-first-attempt/runs/pilot/failure.json) 与 [manifest.json](debug/transformers5-first-attempt/runs/pilot/manifest.json)。Notebook bootstrap 为后续运行固定安装 `transformers==4.57.1`、`accelerate==1.10.1` 与 `PyYAML==6.0.3`；准备流程要求安装后重启 kernel，以重新载入依赖。这里观察到的是来源 revision 元数据缺失；仅凭该异常无法推断模型权重内容错误。正式统计来源仍为经过完整运行校验的 `runs/full/`，首次尝试的环境异常与模型答题失败分别记录。

## 正式结果

| 本次方法 | N | 正确数 | 准确率 | 平均输出 token | 平均生成延迟（秒） | 截断数 | 解析错误数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{measured_rows}

{comparison_sentence(groups, 'cod', 'cot')}

零样本消融：{comparison_sentence(groups, 'cod_zero', 'cod')}

同题正确性比较中，CoT 正确而 CoD 错误有 {cot_only} 题，CoD 正确而 CoT 错误有 {cod_only} 题。均值由所有输出计算，错误、解析错误和截断记录均保留在对应组的分母与开销统计中。CoD 的收益与代价由上述实测变化共同决定。

![正式实验的准确率、输出 token 与生成延迟]({figures[0]})

导出图：[PNG]({figures[0]})、[SVG]({figures[1]})。图中四种方法共享 100 道题；绘图标签采用英文，便于跨环境导出。

四组的数值汇总另存为 [metrics.csv](runs/full/metrics.csv)，包含严格/官方正确数与准确率、平均 token、秒制延迟、截断、解析错误和判定变化数量；CSV 中准确率为 0–1 比例。

![输出 token 与严格、官方准确率的关系]({figures[2]})

散点图以平均生成 token 为横轴、准确率为纵轴。实心圆表示严格评分，空心菱形表示官方评分；同一方法的两种判定共享生成开销。导出：[PNG]({figures[2]})、[SVG]({figures[3]})。

{official_rescore_markdown(run, rescored)}

## 偏差、失败与固定规则样例

本次运行共 {wrong} 条答案被判为错误，包含 {parse_errors} 条解析错误；截断共 {truncated} 条。解析错误类型计数为 `{json.dumps(error_types, ensure_ascii=False)}`。这些记录直接来自完整 JSONL，失败输出和实际消耗均保留。

主要可比性限制来自 100 题抽样、单个 3B 模型、FP16 本地 greedy 生成、同步 GPU 计时、特殊 token 计数和严格判分。8-shot 与 0-shot 同时改变示例上下文及输入长度，因此 0-shot 消融反映完整提示策略的差异。各方法固定使用同一输出预算，截断可能影响最终答案完整性；报告同时列示预算耗尽数量以辅助解释。单次运行无法量化设备负载波动或跨硬件稳定性。

选例规则在结果解释前固定为：计划中的前三题；计划顺序中首个 CoT/CoD 正确性差异；首个任一方法错误；首个解析错误；首个截断。重复 ID 合并，同一题展示四组评分与计量，CoT/CoD 及发生错误的其他方法提供输出摘录。类别缺席时跳过该类别。摘录最多显示前 1,200 字符；完整原文保存在 [results.jsonl](runs/full/results.jsonl) 和 [reviewed_examples.json](runs/full/reviewed_examples.json)。选例覆盖固定位置与失败类别，不能代表整体错误分布。

{example_markdown(examples)}

## 验证状态

| 检查 | 状态 | 实际证据与边界 |
| --- | --- | --- |
| 官方资源、本地数据和提示词指纹 | passed | 报告生成时执行本地资源校验 |
| 正式运行完整性 | passed | 完成 manifest；400 个唯一样本/方法组合；四组各 100 道相同题目 |
| 原始输出评分与摘要重算 | passed | 重算每条解析结果、正确性、token、延迟、截断和解析错误，并核对 summary |
| 官方评分敏感性复算 | passed | AST 直接执行冻结官方函数；逐条复算并记录输出与原 Gold 指纹；共 {rescored['changed_total']} 条判定变化 |
| 逐题答案 | {all_correct_status} | {400 - wrong}/400 条正确；failed 表示已保存的模型答题失败 |
| 答案解析 | {parse_status} | {parse_errors} 条解析错误，全部计入错误与成本 |
| 输出预算完整性 | {truncated_status} | {truncated} 条达到长度上限，截断标记保留 |
| 全量 1,319 题推理、其他模型及跨硬件复跑 | not run | 正式范围为同一模型上的固定 100 题 |
| 原论文 API 延迟复现 | not run | 本次采用已声明的本地生成计时 |
| Kaggle 外部成绩或排名验证 | not run | 运行记录支持本地实验指标；公开评分证据独立取得 |

`passed` 指报告实际完成的输入检查或已完成运行记录；`failed` 指表中明确列出的答题、解析或预算失败；`not run` 指本次范围以外的验证。单元测试通过与否以对应终端或 Notebook 的真实输出为准，本报告通过实验产物完成审计。

## 源码、数据与许可引用

- 原论文：*Chain of Draft: Thinking Faster by Writing Less*，arXiv:2502.18600，{paper['requested_url']}。PDF SHA-256：`{paper['sha256']}`。
- 官方实现：{source['repository']}，commit `{source['commit']}`；[冻结源码](reference/upstream/)，源码 ZIP SHA-256：`{source['archive']['sha256']}`，源码树 SHA-256：`{source['tree_sha256']}`。
- 数据：{run.data_manifest['dataset']}，`main/test`，revision `{run.data_manifest['revision']}`；[数据清单](data/manifest.json) 记录逐页来源与 X-Revision。
- 源码许可：{license_note}
- GSM8K 数据卡：[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)。使用与再分发须同时核对该数据集与基础模型的当前许可条款。

本次实验入口为 [experiment.py](experiment.py)，唯一正式配置为 [config.json](config.json)，报告与图表生成入口为 [report_results.py](report_results.py)。完整逐次记录、运行环境与摘要分别保存在 `runs/full/results.jsonl`、`runs/full/manifest.json` 和 `runs/full/summary.json`。
"""


def main() -> None:
    """Validate real full-run files, then write report, fixed examples and exportable figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project root containing config.json and runs/full")
    args = parser.parse_args()
    run = validate_run(args.root.resolve())
    rescored = official_rescore(run)
    figures = make_figures(run, rescored)
    metrics_csv = write_metrics_csv(run, rescored)
    examples = select_examples(run)
    exp.write_json(run.root / "runs" / "full" / "reviewed_examples.json", {
        "selection_rule": "First three planned IDs; first CoT/CoD correctness disagreement; first error, parse error, and truncation; deduplicate in plan order.",
        "source_hashes": run.hashes,
        "examples": examples,
    })
    exp.write_json(run.root / "runs" / "full" / "official_rescore.json", rescored)
    report = render_report(run, figures, examples, rescored)
    (run.root / "report.md").write_bytes(report.encode("utf-8"))
    print(json.dumps({"status": "passed", "records": 400, "report": str(run.root / "report.md"),
                      "figures": figures, "selected_examples": len(examples),
                      "metrics_csv": metrics_csv, "official_changed": rescored["changed_total"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
