"""Run the pinned, batch-one GSM8K Chain-of-Draft reproduction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time
import traceback
from typing import Any


CONFIG_KEYS = {
    "model_id", "model_revision", "seed", "sample_size", "pilot_size",
    "max_new_tokens", "warmup_tokens", "few_shot_count", "dtype", "device", "methods",
}
METHODS = {"baseline", "cot", "cod", "cod_zero"}
NUMBER = re.compile(r"(?<![\w.,])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?![\w,]|\.\d)")
SCORING_POLICY = {
    "name": "strict_decimal_v1",
    "prediction": "Use the final #### section when present; accept exactly one numeric value in the selected text.",
    "gold": "Require a final #### section containing one complete numeric literal.",
    "official_difference": "Official GSM8K code uses the first #### section, removes comma/$/%, then tries string equality and the first unsigned numeric match with float equality. This runner uses the final section, signed Decimal values and explicit parse errors for ambiguous text.",
}
RECORD_KEYS = {
    "schema_version", "phase", "sample_index", "id", "method", "question", "gold",
    "prompt_sha256", "input_tokens", "output_tokens", "output_token_ids", "raw_output",
    "output", "parsed_answer", "parse_error", "correct", "latency_seconds", "end_reason",
    "truncated", "started_at", "completed_at",
}


@dataclass
class Runtime:
    """Hold the loaded model, tokenizer and observed execution environment."""

    torch: Any
    tokenizer: Any
    model: Any
    device: str
    context_limit: int
    metadata: dict[str, Any]


def utc_now() -> str:
    """Return an unambiguous UTC timestamp for persisted execution events."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally for reproducibility and resume validation."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Hash the exact UTF-8 prompt passed to the chat template."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON artifact after flushing its complete contents."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document and preserve its path in failure messages."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def load_config(path: Path) -> dict[str, Any]:
    """Validate the single formal experiment configuration."""
    config = read_json(path)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"Config must contain exactly {sorted(CONFIG_KEYS)}")
    for key in ("seed", "sample_size", "pilot_size", "max_new_tokens", "warmup_tokens", "few_shot_count"):
        minimum = 0 if key in {"seed", "few_shot_count"} else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"Config {key} must be an integer >= {minimum}")
    if not 1 <= config["pilot_size"] <= config["sample_size"] <= 1319:
        raise ValueError("Require 1 <= pilot_size <= sample_size <= 1319")
    if not isinstance(config["model_id"], str) or not config["model_id"].strip():
        raise ValueError("Config model_id must be a nonempty Hugging Face model identifier")
    if not isinstance(config["model_revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise ValueError("Config model_revision must pin a full lowercase commit SHA")
    if config["dtype"] != "float16" or config["device"] != "cuda:0":
        raise ValueError("This experiment requires dtype=float16 and device=cuda:0")
    if (not isinstance(config["methods"], list) or any(not isinstance(item, str) for item in config["methods"])
            or len(config["methods"]) != 4 or set(config["methods"]) != METHODS):
        raise ValueError(f"Config methods must list each of {sorted(METHODS)} exactly once")
    return config


def decimal_string(value: Decimal) -> str:
    """Represent a finite numeric answer independently of redundant zeroes."""
    if not value.is_finite():
        raise ValueError("Answer must be finite")
    if not value:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def parse_prediction(output: str) -> tuple[str | None, str | None]:
    """Extract exactly one signed Decimal from the final answer section or whole output."""
    selected = output.rsplit("####", 1)[-1] if "####" in output else output
    matches = NUMBER.findall(selected.replace("\N{MINUS SIGN}", "-"))
    if len(matches) != 1:
        return None, "no_number" if not matches else "multiple_numbers"
    try:
        return decimal_string(Decimal(matches[0].replace(",", ""))), None
    except (InvalidOperation, ValueError):
        return None, "invalid_number"


def parse_gold(answer: str) -> str:
    """Require the dataset gold answer to end in a complete numeric #### section."""
    if "####" not in answer:
        raise ValueError("Gold answer lacks ####")
    selected = answer.rsplit("####", 1)[-1].strip()
    if not NUMBER.fullmatch(selected):
        raise ValueError(f"Gold final section must be one numeric literal: {selected!r}")
    return decimal_string(Decimal(selected.replace(",", "")))


def load_data(path: Path) -> list[dict[str, Any]]:
    """Load and validate the complete 1,319-item GSM8K test split."""
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or set(record) != {"id", "question", "answer"}:
                    raise ValueError("Expected exactly id/question/answer")
                if type(record["id"]) is not int or record["id"] != line_number - 1:
                    raise ValueError(f"Expected original ordered integer id {line_number - 1}, got {record['id']!r}")
                if any(not isinstance(record[key], str) or not record[key].strip() for key in ("question", "answer")):
                    raise ValueError("Expected nonempty question and answer strings")
                parse_gold(record["answer"])
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid dataset {path}:{line_number}: {exc}") from exc
            records.append(record)
    if len(records) != 1319:
        raise ValueError(f"Expected complete GSM8K test split (1319 rows), got {len(records)}")
    return records


def load_prompts(root: Path, few_shot_count: int) -> dict[str, dict[str, Any]]:
    """Read the three official YAML prompt configurations and validate their contract."""
    import yaml

    prompts = {}
    for method in ("baseline", "cot", "cod"):
        path = root / "prompts" / f"gsm8k_{method}.yaml"
        with path.open(encoding="utf-8") as stream:
            prompt = yaml.safe_load(stream)
        if not isinstance(prompt, dict) or set(prompt) != {"system_prompt", "format", "fewshot"}:
            raise ValueError(f"Prompt {path} requires exactly system_prompt/format/fewshot")
        if any(not isinstance(prompt[key], str) for key in ("system_prompt", "format")):
            raise ValueError(f"Prompt {path} requires string system_prompt and format")
        if not isinstance(prompt["fewshot"], list) or len(prompt["fewshot"]) < few_shot_count:
            raise ValueError(f"Prompt {path} has fewer than {few_shot_count} examples")
        for example in prompt["fewshot"]:
            if (not isinstance(example, dict) or set(example) != {"question", "answer"}
                    or any(not isinstance(value, str) for value in example.values())):
                raise ValueError(f"Prompt {path} has an invalid fewshot example")
        compose_request(prompt, "contract check", few_shot_count)
        prompts[method] = prompt
    return prompts


def compose_request(prompt: dict[str, Any], question: str, few_shot_count: int) -> str:
    """Compose the official system/examples/question text for a single user message."""
    if few_shot_count > len(prompt["fewshot"]):
        raise ValueError("Requested fewshot count exceeds available official examples")
    request = prompt["system_prompt"] + "\n"
    if few_shot_count:
        examples = [prompt["format"].format(**item) for item in prompt["fewshot"][:few_shot_count]]
        request += "\n".join(examples) + "\n"
    return request + prompt["format"].format(question=question, answer="")


def make_plan(data: list[dict[str, Any]], config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    """Shuffle all test indices and precompute deterministic per-question method order."""
    if phase not in {"pilot", "full"}:
        raise ValueError("Phase must be pilot or full")
    rng = random.Random(config["seed"])
    indices = list(range(len(data)))
    rng.shuffle(indices)
    count = config["pilot_size"] if phase == "pilot" else config["sample_size"]
    plan = []
    for index in indices[:count]:
        methods = list(config["methods"])
        rng.shuffle(methods)
        plan.append({"id": data[index]["id"], "data_index": index, "methods": methods})
    return plan


def make_identity(root: Path, config_path: Path, config: dict[str, Any], phase: str,
                  plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Fingerprint the exact configuration, source, dataset, prompts and implementation."""
    source_files = {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
                    for path in sorted((root / "reference").rglob("*")) if path.is_file()}
    if not source_files:
        raise ValueError("reference/ must contain frozen source and provenance files")
    return {
        "schema_version": 1, "phase": phase, "config": config, "plan": plan,
        "sample_ids": [item["id"] for item in plan], "scoring_policy": SCORING_POLICY,
        "fingerprints": {
            "config": sha256_file(config_path), "data/test.jsonl": sha256_file(root / "data" / "test.jsonl"),
            "prompts": {f"gsm8k_{method}.yaml": sha256_file(root / "prompts" / f"gsm8k_{method}.yaml")
                        for method in ("baseline", "cot", "cod")},
            "implementation": sha256_file(Path(__file__).resolve()), "source_files": source_files,
        },
    }


def validate_context(input_tokens: int, max_new_tokens: int, context_limit: int) -> None:
    """Fail explicitly when complete input and requested output exceed model context."""
    if input_tokens < 1 or max_new_tokens < 1 or input_tokens + max_new_tokens > context_limit:
        raise ValueError(f"Context exceeded: input={input_tokens}, max_new_tokens={max_new_tokens}, limit={context_limit}")


@lru_cache(maxsize=1)
def get_runtime(model_id: str, revision: str, dtype: str, device: str) -> Runtime:
    """Load one pinned FP16 CUDA model and cache it for notebook pilot/full calls."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype != "float16" or device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("FP16 inference requires an available CUDA device at cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True).to(device)
    model.eval()
    actual_revision = getattr(model.config, "_commit_hash", None)
    if actual_revision != revision:
        raise RuntimeError(f"Loaded model revision differs: expected={revision}, actual={actual_revision}")
    parameter = next(model.parameters())
    if parameter.dtype != torch.float16 or str(parameter.device) != device:
        raise RuntimeError(f"Loaded model has dtype={parameter.dtype}, device={parameter.device}")
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if type(context_limit) is not int or context_limit < 1:
        raise RuntimeError("Model config lacks a valid max_position_embeddings")
    tokenizer_limit = tokenizer.model_max_length
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000_000:
        context_limit = min(context_limit, tokenizer_limit)
    versions = {}
    for package in ("torch", "transformers", "tokenizers", "accelerate", "safetensors", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        "model_id": model_id, "model_revision": actual_revision,
        "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
        "dtype": str(parameter.dtype), "device": str(parameter.device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cuda_version": torch.version.cuda, "context_limit": context_limit,
        "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
        "timing": "batch=1; CUDA synchronized immediately before and after model.generate; seconds; tokenizer/decode excluded",
        "generation": {"do_sample": False, "temperature_argument": "omitted", "input_truncation": False},
    }
    return Runtime(torch, tokenizer, model, device, context_limit, metadata)


def generate_output(runtime: Runtime, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    """Generate greedily from one user message with synchronized GPU timing."""
    torch, tokenizer, model = runtime.torch, runtime.tokenizer, runtime.model
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(runtime.device)
    input_tokens = int(input_ids.shape[-1])
    validate_context(input_tokens, max_new_tokens, runtime.context_limit)
    eos = model.generation_config.eos_token_id
    if eos is None:
        raise RuntimeError("Pinned model generation config lacks eos_token_id")
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    pad = tokenizer.pad_token_id
    if pad is None:
        raise RuntimeError("Pinned tokenizer lacks pad_token_id")
    with torch.inference_mode():
        torch.cuda.synchronize(runtime.device)
        started = time.perf_counter()
        generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                   do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                                   pad_token_id=pad, return_dict_in_generate=False)
        torch.cuda.synchronize(runtime.device)
        latency = time.perf_counter() - started
    token_ids = generated[0, input_tokens:].tolist()
    if not token_ids:
        raise RuntimeError("Model returned zero generated tokens")
    end_reason = "eos" if token_ids[-1] in eos_ids else "length"
    if end_reason == "length" and len(token_ids) != max_new_tokens:
        raise RuntimeError(f"Unexpected generation stop after {len(token_ids)} tokens")
    return {
        "input_tokens": input_tokens, "output_tokens": len(token_ids), "output_token_ids": token_ids,
        "raw_output": tokenizer.decode(token_ids, skip_special_tokens=False),
        "output": tokenizer.decode(token_ids, skip_special_tokens=True), "latency_seconds": latency,
        "end_reason": end_reason, "truncated": end_reason == "length",
    }


def build_jobs(data: list[dict[str, Any]], plan: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any], str]]:
    """Expand the immutable sample plan into its exact persisted result order."""
    return [(sample_index, data[item["data_index"]], method)
            for sample_index, item in enumerate(plan) for method in item["methods"]]


def job_prompt(prompts: dict[str, dict[str, Any]], method: str, question: str, count: int) -> str:
    """Apply official CoD zero-shot or the configured official few-shot prompt."""
    return compose_request(prompts["cod" if method == "cod_zero" else method], question,
                           0 if method == "cod_zero" else count)


def repair_result_tail(path: Path, retained_bytes: int | None, detail: str) -> None:
    """Back up exact original bytes and durably repair the final JSONL commit boundary."""
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    backup = path.with_name(f"{path.name}.recovery-{digest}.bak")
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError(f"Recovery backup differs from original bytes: {backup}")
    else:
        with backup.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
    action = "append_missing_lf" if retained_bytes is None else "truncate_uncommitted_tail"
    recovery = {"action": action, "detail": detail, "original_sha256": digest,
                "original_bytes": len(original), "backup": backup.name,
                "retained_bytes": len(original) if retained_bytes is None else retained_bytes,
                "started_at": utc_now(), "status": "backed_up"}
    recovery_path = backup.with_suffix(".json")
    write_json(recovery_path, recovery)
    with path.open("r+b") as stream:
        if retained_bytes is None:
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")
        else:
            stream.truncate(retained_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    recovery.update({"status": "completed", "completed_at": utc_now()})
    write_json(recovery_path, recovery)
    print(f"RECOVER results={path.name} action={action} backup={backup.name}", flush=True)


def read_results(path: Path, jobs: list[tuple[int, dict[str, Any], str]], phase: str,
                 prompts: dict[str, dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the ordered result prefix and recover a backed-up uncommitted final line."""
    if not path.exists():
        return []
    records = []
    repair = None
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            committed = line.endswith(b"\n")
            try:
                row = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if committed:
                    raise ValueError(f"Invalid results {path}:{line_number}: {exc}") from exc
                repair = (stream.tell() - len(line), f"Uncommitted final line {line_number}: {exc}")
                break
            try:
                if not isinstance(row, dict) or set(row) != RECORD_KEYS or row["schema_version"] != 1:
                    raise ValueError("Result schema mismatch")
                if len(records) >= len(jobs):
                    raise ValueError("Results contain extra or duplicate records")
                sample_index, question, method = jobs[len(records)]
                expected = (phase, sample_index, question["id"], method, question["question"], parse_gold(question["answer"]))
                actual = tuple(row[key] for key in ("phase", "sample_index", "id", "method", "question", "gold"))
                if actual != expected:
                    raise ValueError("Results violate phase, sample or method order")
                prompt = job_prompt(prompts, method, question["question"], config["few_shot_count"])
                if row["prompt_sha256"] != sha256_text(prompt):
                    raise ValueError("Result prompt hash mismatch")
                if any(not isinstance(row[key], str) for key in ("output", "raw_output", "started_at", "completed_at")):
                    raise ValueError("Result output and timestamps must be strings")
                parsed, error = parse_prediction(row["output"])
                if (row["parsed_answer"], row["parse_error"], row["correct"]) != (parsed, error, error is None and parsed == expected[-1]):
                    raise ValueError("Result scoring fields disagree with saved output")
                if type(row["correct"]) is not bool or type(row["truncated"]) is not bool:
                    raise ValueError("Result correctness and truncation flags must be booleans")
                if (type(row["input_tokens"]) is not int or row["input_tokens"] < 1
                        or type(row["output_tokens"]) is not int or not 1 <= row["output_tokens"] <= config["max_new_tokens"]):
                    raise ValueError("Result token counts are invalid")
                if (not isinstance(row["output_token_ids"], list) or len(row["output_token_ids"]) != row["output_tokens"]
                        or any(type(token) is not int or token < 0 for token in row["output_token_ids"])):
                    raise ValueError("Result token ids disagree with token count")
                if (type(row["latency_seconds"]) not in {int, float} or not math.isfinite(row["latency_seconds"])
                        or row["latency_seconds"] < 0):
                    raise ValueError("Result latency must be a finite nonnegative number")
                if row["end_reason"] not in {"eos", "length"} or row["truncated"] != (row["end_reason"] == "length"):
                    raise ValueError("Result stop/truncation fields disagree")
                if row["truncated"] and row["output_tokens"] != config["max_new_tokens"]:
                    raise ValueError("Length stop must use the full configured token budget")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid results {path}:{line_number}: {exc}") from exc
            records.append(row)
            if not committed:
                repair = (None, f"Valid final record at line {line_number} lacks its LF delimiter")
    if repair is not None:
        repair_result_tail(path, *repair)
    return records


def summarize(records: list[dict[str, Any]], methods: list[str], phase: str, sample_ids: list[int]) -> dict[str, Any]:
    """Compute each method's accuracy, actual token use, latency and failure counts."""
    groups = {}
    for method in methods:
        rows = [row for row in records if row["method"] == method]
        count = len(rows)
        correct = sum(row["correct"] for row in rows)
        groups[method] = {
            "N": count, "correct": correct, "accuracy": correct / count if count else None,
            "mean_output_tokens": sum(row["output_tokens"] for row in rows) / count if count else None,
            "mean_latency": sum(row["latency_seconds"] for row in rows) / count if count else None,
            "truncated": sum(row["truncated"] for row in rows),
            "parse_errors": sum(row["parse_error"] is not None for row in rows),
        }
    return {"phase": phase, "sample_ids": sample_ids, "total_records": len(records), "methods": groups,
            "scoring_policy": SCORING_POLICY, "latency_unit": "seconds"}


def run_experiment(config_path: str | Path, phase: str, output_dir: str | Path) -> dict[str, Any]:
    """Run or safely resume one frozen experiment phase and persist every generation."""
    config_path = Path(config_path).resolve()
    root = config_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.jsonl"
    context: dict[str, Any] = {"phase": phase, "stage": "input_validation"}
    try:
        config = load_config(config_path)
        import prepare_resources

        if Path(prepare_resources.ROOT).resolve() != root:
            raise ValueError("Config directory must match the official resource preparation root")
        context["stage"] = "resource_verification"
        verification = prepare_resources.verify_local_resources()
        context["stage"] = "input_validation"
        data = load_data(root / "data" / "test.jsonl")
        prompts = load_prompts(root, config["few_shot_count"])
        plan = make_plan(data, config, phase)
        identity = make_identity(root, config_path, config, phase, plan)
        identity["resource_verification"] = verification
        jobs = build_jobs(data, plan)
        context["stage"] = "resume_validation"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in identity.items()):
                raise ValueError("Resume manifest differs from current config, data, prompts, source, code or sample plan")
            if not isinstance(manifest.get("warmups"), list) or "runtime" not in manifest:
                raise ValueError("Resume manifest lacks runtime/warmup contract")
        else:
            if any(path.name != "failure.json" for path in output_dir.iterdir()):
                raise ValueError("Output directory must be empty when creating a new manifest")
            manifest = {**identity, "created_at": utc_now(), "runtime": None, "warmups": [], "completed": False}
            write_json(manifest_path, manifest)
        records = read_results(results_path, jobs, phase, prompts, config)
        if len(records) < len(jobs):
            context["stage"] = "model_load"
            runtime = get_runtime(config["model_id"], config["model_revision"], config["dtype"], config["device"])
            if manifest["runtime"] is not None and manifest["runtime"] != runtime.metadata:
                raise ValueError("Resume runtime differs from the recorded model, hardware or software versions")
            manifest["runtime"] = runtime.metadata
            context["stage"] = "warmup"
            sample_index, question, method = jobs[len(records)]
            prompt = job_prompt(prompts, method, question["question"], config["few_shot_count"])
            warmup = {"started_at": utc_now(), "before_record": len(records), "id": question["id"],
                      "method": method, "requested_tokens": config["warmup_tokens"], "status": "started"}
            manifest["warmups"].append(warmup)
            write_json(manifest_path, manifest)
            print(f"BEGIN phase={phase} warmup tokens={config['warmup_tokens']}", flush=True)
            generated = generate_output(runtime, prompt, config["warmup_tokens"])
            warmup.update({"status": "completed", "completed_at": utc_now(), **generated})
            write_json(manifest_path, manifest)
            print(f"DONE phase={phase} warmup tokens={generated['output_tokens']}", flush=True)
            with results_path.open("a", encoding="utf-8", newline="\n") as stream:
                for sample_index, question, method in jobs[len(records):]:
                    context = {"phase": phase, "stage": "generation", "sample_index": sample_index,
                               "id": question["id"], "method": method}
                    print(f"BEGIN phase={phase} sample={sample_index + 1}/{len(plan)} id={question['id']} method={method}", flush=True)
                    started_at = utc_now()
                    prompt = job_prompt(prompts, method, question["question"], config["few_shot_count"])
                    generated = generate_output(runtime, prompt, config["max_new_tokens"])
                    parsed, error = parse_prediction(generated["output"])
                    gold = parse_gold(question["answer"])
                    row = {"schema_version": 1, "phase": phase, "sample_index": sample_index, "id": question["id"],
                           "method": method, "question": question["question"], "gold": gold,
                           "prompt_sha256": sha256_text(prompt), "started_at": started_at,
                           "completed_at": utc_now(), **generated, "parsed_answer": parsed, "parse_error": error,
                           "correct": error is None and parsed == gold}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    records.append(row)
                    print(f"DONE phase={phase} sample={sample_index + 1}/{len(plan)} id={question['id']} method={method} correct={row['correct']} tokens={row['output_tokens']} seconds={row['latency_seconds']:.4f} end={row['end_reason']} parse_error={error}", flush=True)
        context["stage"] = "summary"
        summary = summarize(records, config["methods"], phase, identity["sample_ids"])
        write_json(output_dir / "summary.json", summary)
        manifest["completed"] = True
        manifest["completed_at"] = utc_now()
        write_json(manifest_path, manifest)
        return summary
    except Exception as exc:
        write_json(output_dir / "failure.json", {**context, "failed_at": utc_now(), "exception_type": type(exc).__name__,
                                                "message": str(exc), "traceback": traceback.format_exc()})
        raise


def main() -> None:
    """Parse the sole CLI configuration path, phase and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("pilot", "full"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = run_experiment(args.config, args.phase, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
