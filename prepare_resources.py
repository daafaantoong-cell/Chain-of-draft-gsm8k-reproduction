"""Download and verify the pinned public resources for the CoD experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parent
SOURCE_REPOSITORY = "https://github.com/sileix/chain-of-draft"
SOURCE_COMMIT = "a7dbf5dea808b1aa1e12f7a90ea573321581df78"
SOURCE_ARCHIVE = f"https://codeload.github.com/sileix/chain-of-draft/zip/{SOURCE_COMMIT}"
DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAPER_URL = "https://arxiv.org/pdf/2502.18600"
EXPECTED_COUNT = 1319
PROMPTS = ("baseline", "cot", "cod")


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 hex digest of the supplied bytes."""
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write a UTF-8 JSON artifact with stable indentation and LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def make_session() -> requests.Session:
    """Create an HTTP session with bounded retries for transient public downloads."""
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "CoD-Reproduction-resource-preparation/1.0"
    return session


def fetch(session: requests.Session, url: str, **kwargs: object) -> requests.Response:
    """Fetch a public resource and raise an HTTP error for an unsuccessful response."""
    response = session.get(url, timeout=(20, 120), **kwargs)
    response.raise_for_status()
    return response


def response_metadata(response: requests.Response) -> dict[str, object]:
    """Capture public response identity and cache validators for provenance."""
    return {
        "requested_url": response.request.url,
        "resolved_url": response.url,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "x_revision": response.headers.get("X-Revision"),
        "bytes": len(response.content),
        "sha256": sha256_bytes(response.content),
    }


def source_tree_manifest(directory: Path) -> tuple[list[dict[str, object]], str]:
    """Hash every official source file and the ordered path-and-digest manifest."""
    files = []
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            content = path.read_bytes()
            file_hash = sha256_bytes(content)
            files.append({"path": relative, "bytes": len(content), "sha256": file_hash})
            digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
    return files, digest.hexdigest()


def prepare_source(session: requests.Session) -> dict[str, object]:
    """Extract the pinned official archive and copy the three prompt files byte for byte."""
    response = fetch(session, SOURCE_ARCHIVE)
    directory = ROOT / "reference" / "upstream"
    prefix = f"chain-of-draft-{SOURCE_COMMIT}/"
    archive_files = {}
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise ValueError(f"Source archive CRC validation failed: {bad_file}")
        for member in archive.infolist():
            if member.is_dir():
                continue
            if not member.filename.startswith(prefix):
                raise ValueError(f"Unexpected archive root: {member.filename}")
            relative = PurePosixPath(member.filename.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            archive_files[relative.as_posix()] = archive.read(member)
    existing = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    extra = existing - archive_files.keys()
    if extra:
        raise ValueError(f"Unexpected files in official source directory: {sorted(extra)}")
    for relative, content in archive_files.items():
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    prompt_files = []
    for strategy in PROMPTS:
        name = f"gsm8k_{strategy}.yaml"
        original = directory / "configs" / name
        destination = ROOT / "prompts" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        content = destination.read_bytes()
        count = len(re.findall(r"^  - question:", content.decode("utf-8"), re.MULTILINE))
        if count != 8:
            raise ValueError(f"Expected eight official examples in {name}; found {count}")
        prompt_files.append({
            "path": destination.relative_to(ROOT).as_posix(),
            "upstream_path": f"configs/{name}",
            "sha256": sha256_bytes(content),
            "fewshot_count": count,
        })
    files, tree_hash = source_tree_manifest(directory)
    print(f"Official source: {SOURCE_COMMIT}, {len(files)} files", flush=True)
    return {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "pin_resolution": {
            "method": "git ls-remote https://github.com/sileix/chain-of-draft.git HEAD",
            "observed_head": SOURCE_COMMIT,
            "observed_date": "2026-11-06",
            "behavior": "The recorded HEAD is the fixed source commit for this experiment.",
        },
        "archive": response_metadata(response),
        "directory": "reference/upstream",
        "tree_sha256_algorithm": "SHA256 of sorted UTF-8 path + NUL + file SHA256 + LF records",
        "tree_sha256": tree_hash,
        "files": files,
        "prompts": prompt_files,
        "request_contract": {
            "sources": ["utils.py:24", "llm_client.py:12", "tasks/base.py:30"],
            "compose": [
                "request = config.system_prompt + '\\n'",
                "shot=None uses len(config.fewshot), which is eight for each GSM8K configuration",
                "fewshot = [config.format.format(question=ex.question, answer=ex.answer) for ex in config.fewshot[:shot]]",
                "request += '\\n'.join(fewshot) + '\\n' when shot != 0",
                "request += config.format.format(question=question, answer='')",
            ],
            "yaml_whitespace": "YAML literal-block terminal newlines participate in composition.",
            "messages": [{"role": "user", "content": "the complete composed request"}],
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    }


def validate_records(records: list[dict[str, object]]) -> None:
    """Validate the complete ordered GSM8K test set and each numeric final gold answer."""
    if len(records) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} records; found {len(records)}")
    for expected_id, record in enumerate(records):
        if set(record) != {"id", "question", "answer"}:
            raise ValueError(f"Unexpected record fields at row {expected_id}")
        if type(record["id"]) is not int or record["id"] != expected_id:
            raise ValueError(f"Expected original row_idx {expected_id}; found {record['id']!r}")
        question = record["question"]
        answer = record["answer"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question must be a nonempty string at row {expected_id}")
        if not isinstance(answer, str) or "####" not in answer:
            raise ValueError(f"Missing final-answer separator at row {expected_id}")
        gold = answer.rsplit("####", 1)[1].strip().replace(",", "")
        try:
            value = Decimal(gold)
        except InvalidOperation as error:
            raise ValueError(f"Invalid numeric gold answer at row {expected_id}") from error
        if not value.is_finite():
            raise ValueError(f"Expected finite numeric gold answer at row {expected_id}")


def prepare_data(session: requests.Session) -> dict[str, object]:
    """Download all verified rows from one dataset revision and write deterministic JSONL."""
    metadata_url = f"https://huggingface.co/api/datasets/{DATASET_ID}/revision/{DATASET_REVISION}"
    metadata_response = fetch(session, metadata_url)
    metadata = metadata_response.json()
    if metadata.get("sha") != DATASET_REVISION:
        raise ValueError("Dataset metadata revision differs from the fixed experiment revision")
    records = []
    pages = []
    for offset in range(0, EXPECTED_COUNT, 100):
        length = min(100, EXPECTED_COUNT - offset)
        response = fetch(session, ROWS_URL, params={
            "dataset": DATASET_ID,
            "config": "main",
            "split": "test",
            "offset": offset,
            "length": length,
        })
        revision = response.headers.get("X-Revision")
        if revision != DATASET_REVISION:
            raise ValueError(
                f"Rows revision {revision!r} differs from pinned {DATASET_REVISION}; "
                "the rows endpoint serves the current dataset revision"
            )
        payload = response.json()
        if payload.get("num_rows_total") != EXPECTED_COUNT or payload.get("partial") is not False:
            raise ValueError(f"Unexpected dataset size or partial response at offset {offset}")
        features = {feature["name"] for feature in payload["features"]}
        if features != {"question", "answer"}:
            raise ValueError(f"Unexpected dataset features: {sorted(features)}")
        rows = payload["rows"]
        if len(rows) != length:
            raise ValueError(f"Expected {length} rows at offset {offset}; found {len(rows)}")
        for index, item in enumerate(rows, start=offset):
            if item["row_idx"] != index or item["truncated_cells"]:
                raise ValueError(f"Unordered or truncated dataset row at {index}")
            row = item["row"]
            records.append({"id": item["row_idx"], "question": row["question"], "answer": row["answer"]})
        pages.append({"offset": offset, "count": len(rows), **response_metadata(response)})
        print(f"GSM8K rows: {len(records)}/{EXPECTED_COUNT}", flush=True)
    validate_records(records)
    content = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records).encode("utf-8")
    destination = ROOT / "data" / "test.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    manifest = {
        "dataset": DATASET_ID,
        "config": "main",
        "split": "test",
        "revision": DATASET_REVISION,
        "source": ROWS_URL,
        "revision_contract": "Every response X-Revision equals the pinned dataset revision; rows API serves the current revision.",
        "dataset_metadata": response_metadata(metadata_response),
        "row_count": len(records),
        "id_contract": "id is the original integer row_idx; ordered contiguous IDs 0 through 1318",
        "columns": ["id", "question", "answer"],
        "path": "data/test.jsonl",
        "encoding": "UTF-8",
        "line_endings": "LF",
        "bytes": len(content),
        "sha256": sha256_bytes(content),
        "validation": {"unique_ids": True, "original_order": True, "all_gold_separators": True, "all_gold_numeric": True},
        "pages": pages,
    }
    write_json(ROOT / "data" / "manifest.json", manifest)
    return manifest


def prepare_paper(session: requests.Session) -> dict[str, object]:
    """Download the official arXiv PDF and record its exact delivered bytes."""
    response = fetch(session, PAPER_URL)
    if not response.content.startswith(b"%PDF-"):
        raise ValueError("The arXiv paper response is not a PDF")
    destination = ROOT / "reference" / "paper.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    print(f"Paper PDF: {len(response.content)} bytes", flush=True)
    return {"arxiv_id": "2502.18600", "path": "reference/paper.pdf", **response_metadata(response)}


def verify_local_resources() -> dict[str, object]:
    """Check local data, official source, prompt copies, and paper against saved fingerprints."""
    provenance = json.loads((ROOT / "reference" / "provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    content = (ROOT / manifest["path"]).read_bytes()
    if sha256_bytes(content) != manifest["sha256"]:
        raise ValueError("GSM8K JSONL SHA-256 mismatch")
    records = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    validate_records(records)
    source = provenance["source"]
    files, tree_hash = source_tree_manifest(ROOT / source["directory"])
    if files != source["files"] or tree_hash != source["tree_sha256"]:
        raise ValueError("Official source tree fingerprint mismatch")
    for prompt in source["prompts"]:
        copied = (ROOT / prompt["path"]).read_bytes()
        original = (ROOT / source["directory"] / prompt["upstream_path"]).read_bytes()
        if copied != original or sha256_bytes(copied) != prompt["sha256"]:
            raise ValueError(f"Official prompt copy mismatch: {prompt['path']}")
    paper = provenance["paper"]
    if sha256_bytes((ROOT / paper["path"]).read_bytes()) != paper["sha256"]:
        raise ValueError("Paper PDF SHA-256 mismatch")
    result = {
        "status": "passed",
        "rows": len(records),
        "data_sha256": manifest["sha256"],
        "source_commit": source["commit"],
        "source_tree_sha256": tree_hash,
        "paper_sha256": paper["sha256"],
        "prompt_copies": "byte-identical",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    """Prepare all pinned resources or verify their saved local fingerprints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify the prepared local artifacts without network access")
    args = parser.parse_args()
    if args.verify_only:
        verify_local_resources()
        return
    with make_session() as session:
        source = prepare_source(session)
        data = prepare_data(session)
        paper = prepare_paper(session)
    write_json(ROOT / "reference" / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "data": {"manifest": "data/manifest.json", "sha256": data["sha256"], "row_count": data["row_count"], "revision": data["revision"]},
        "paper": paper,
    })
    verify_local_resources()


if __name__ == "__main__":
    main()
