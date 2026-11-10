"""Standard-library tests for scoring, prompt fidelity, generation and safe resume."""

from contextlib import nullcontext, redirect_stdout
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import experiment as exp


def config_fixture() -> dict:
    """Return a pinned configuration with a small synthetic test workload."""
    return {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "seed": 42, "sample_size": 3, "pilot_size": 2, "max_new_tokens": 32,
        "warmup_tokens": 2, "few_shot_count": 8, "dtype": "float16", "device": "cuda:0",
        "methods": ["baseline", "cot", "cod", "cod_zero"],
    }


def prompt_fixture(method: str = "cod") -> dict:
    """Build distinguishable official-shape examples for exact prompt assertions."""
    return {"system_prompt": f"{method} system", "format": "Q: {question}\nA: {answer}\n",
            "fewshot": [{"question": f"example {index}", "answer": str(index)} for index in range(8)]}


def generated_fixture(runtime: object, prompt: str, max_new_tokens: int) -> dict:
    """Return a synthetic generation with the real persisted result contract."""
    return {"input_tokens": 12, "output_tokens": 2, "output_token_ids": [7, 9],
            "raw_output": "#### 42<|im_end|>", "output": "#### 42", "latency_seconds": 0.25,
            "end_reason": "eos", "truncated": False}


class ScoringTests(unittest.TestCase):
    """Exercise signed Decimal scoring and ambiguity rejection independently."""

    def test_supported_numbers(self) -> None:
        """Accept signed, fractional and grouped numbers without rounding."""
        cases = {"#### -12": "-12", "#### 1,234.500": "1234.5", "#### .25": "0.25",
                 "The answer is 42.": "42", "#### −0.50": "-0.5", "#### $1,000": "1000",
                 "#### 123456789012345678901234567890.123": "123456789012345678901234567890.123"}
        for output, expected in cases.items():
            with self.subTest(output=output):
                self.assertEqual(exp.parse_prediction(output), (expected, None))

    def test_final_section_has_priority(self) -> None:
        """Select the final marker even when reasoning or earlier markers contain gold."""
        self.assertEqual(exp.parse_prediction("42 was considered. #### 42\nActually #### 41"), ("41", None))

    def test_ambiguous_or_missing_numbers(self) -> None:
        """Reject reasoning with multiple values and final sections lacking one number."""
        cases = {"42 is mentioned, but answer 41": "multiple_numbers", "#### 42 or 41": "multiple_numbers",
                 "No result": "no_number", "42 in reasoning #### unknown": "no_number", "#### 1,23": "no_number"}
        for output, expected in cases.items():
            with self.subTest(output=output):
                self.assertEqual(exp.parse_prediction(output), (None, expected))

    def test_strict_gold(self) -> None:
        """Require a marker and a complete final numeric gold literal."""
        self.assertEqual(exp.parse_gold("reason 4\n#### -1,200.50"), "-1200.5")
        for answer in ("42", "#### 42 dollars", "#### 42 or 41", "####", "#### 1,23"):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                exp.parse_gold(answer)

    def test_summary_keeps_wrong_answer_and_parse_error_separate(self) -> None:
        """Count an output mentioning gold as wrong when its final answer differs."""
        rows = []
        for output in ("gold 42; #### 41", "42 then 41", "#### 42"):
            parsed, error = exp.parse_prediction(output)
            rows.append({"method": "cod", "correct": error is None and parsed == "42", "parse_error": error,
                         "output_tokens": 10, "latency_seconds": 0.5, "truncated": False})
        summary = exp.summarize(rows, ["cod"], "pilot", [1, 2, 3])["methods"]["cod"]
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["parse_errors"], 1)
        self.assertEqual(summary["accuracy"], 1 / 3)


class PromptAndPlanTests(unittest.TestCase):
    """Verify exact official composition and the fixed shuffled sample prefix."""

    def test_official_composition(self) -> None:
        """Preserve the official single-text newline and example formatting."""
        actual = exp.compose_request(prompt_fixture(), "target", 2)
        self.assertEqual(actual, "cod system\nQ: example 0\nA: 0\n\nQ: example 1\nA: 1\n\nQ: target\nA: \n")

    def test_cod_zero_reuses_cod_without_examples(self) -> None:
        """Use the CoD instruction with zero examples for the zero-shot group."""
        prompts = {method: prompt_fixture(method) for method in ("baseline", "cot", "cod")}
        self.assertEqual(exp.job_prompt(prompts, "cod_zero", "target", 8), "cod system\nQ: target\nA: \n")

    @unittest.skipUnless((Path(__file__).parent / "reference/upstream/utils.py").exists(), "Frozen upstream utils needed")
    def test_frozen_upstream_function_matches_exactly(self) -> None:
        """Compare actual official composition for eight-shot and zero-shot text."""
        source = (Path(__file__).parent / "reference/upstream/utils.py").read_text(encoding="utf-8")
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "compose_request")
        for argument in function.args.args:
            argument.annotation = None
        function.returns = None
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "frozen_upstream", "exec"), namespace)
        prompt = prompt_fixture()
        official_config = SimpleNamespace(**{**prompt, "fewshot": [SimpleNamespace(**row) for row in prompt["fewshot"]]})
        for count in (8, 0):
            with self.subTest(count=count):
                self.assertEqual(exp.compose_request(prompt, "target", count),
                                 namespace["compose_request"](official_config, count, "target"))

    def test_shuffled_full_split_and_pilot_prefix(self) -> None:
        """Pin seed-42 sample identities and identical pilot/full method order."""
        data = [{"id": index} for index in range(1319)]
        config = {**config_fixture(), "sample_size": 100, "pilot_size": 12}
        full = exp.make_plan(data, config, "full")
        pilot = exp.make_plan(data, config, "pilot")
        self.assertEqual([item["id"] for item in pilot], [588, 1214, 57, 906, 1252, 618, 599, 289, 608, 400, 237, 616])
        self.assertEqual(pilot, full[:12])
        self.assertEqual(len({item["id"] for item in full}), 100)
        self.assertTrue(all(set(item["methods"]) == exp.METHODS for item in full))

    def test_context_boundary(self) -> None:
        """Allow the exact context limit and reject even one excess token."""
        exp.validate_context(10, 20, 30)
        with self.assertRaisesRegex(ValueError, "input=11"):
            exp.validate_context(11, 20, 30)


class GenerationTests(unittest.TestCase):
    """Verify generation calls through fake tensors without importing Torch."""

    def make_runtime(self, tokens: list[int]) -> exp.Runtime:
        """Create a mock tensor/model runtime with an observable GPU call order."""
        torch = mock.MagicMock()
        torch.inference_mode.side_effect = nullcontext
        tokenizer = mock.Mock()
        tensor = mock.Mock()
        tensor.shape = (1, 3)
        tensor.to.return_value = tensor
        tokenizer.apply_chat_template.return_value = tensor
        tokenizer.pad_token_id = 0
        tokenizer.decode.side_effect = ["#### 42<eos>", "#### 42"]
        model = mock.Mock()
        model.generation_config.eos_token_id = [9, 10]
        generated = mock.MagicMock()
        generated.__getitem__.return_value.tolist.return_value = tokens
        model.generate.return_value = generated
        return exp.Runtime(torch, tokenizer, model, "cuda:0", 100, {})

    def test_single_user_greedy_real_token_count_and_eos(self) -> None:
        """Use one chat message, synchronized timing and all actual output tokens."""
        runtime = self.make_runtime([7, 8, 9])
        events = mock.Mock()
        events.attach_mock(runtime.torch.cuda.synchronize, "sync")
        events.attach_mock(runtime.model.generate, "generate")
        with mock.patch("experiment.time.perf_counter", side_effect=[1.0, 1.25]):
            result = exp.generate_output(runtime, "exact prompt", 20)
        runtime.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "exact prompt"}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt")
        arguments = runtime.model.generate.call_args.kwargs
        self.assertFalse(arguments["do_sample"])
        self.assertNotIn("temperature", arguments)
        self.assertEqual([call[0] for call in events.mock_calls if call[0] in {"sync", "generate"}],
                         ["sync", "generate", "sync"])
        self.assertEqual(result["output_tokens"], 3)
        self.assertEqual(result["latency_seconds"], 0.25)
        self.assertEqual(result["end_reason"], "eos")
        self.assertEqual(result["raw_output"], "#### 42<eos>")

    def test_length_finish_and_unexpected_stop(self) -> None:
        """Distinguish a full output budget from an unrecognized short stop."""
        self.assertTrue(exp.generate_output(self.make_runtime([7, 8]), "prompt", 2)["truncated"])
        with self.assertRaisesRegex(RuntimeError, "Unexpected generation stop"):
            exp.generate_output(self.make_runtime([7]), "prompt", 2)

    def test_context_failure_precedes_generation(self) -> None:
        """Reject oversized untruncated input before invoking the model."""
        runtime = self.make_runtime([9])
        runtime.context_limit = 10
        with self.assertRaises(ValueError):
            exp.generate_output(runtime, "prompt", 8)
        runtime.model.generate.assert_not_called()


class PersistenceTests(unittest.TestCase):
    """Exercise full filesystem contracts, failure recovery and tamper rejection."""

    def setUp(self) -> None:
        """Prepare a complete synthetic dataset and JSON-compatible YAML fixtures."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("data", "prompts", "reference"):
            (self.root / name).mkdir()
        self.config = config_fixture()
        exp.write_json(self.root / "config.json", self.config)
        (self.root / "reference" / "upstream.txt").write_text("frozen official source", encoding="utf-8")
        for method in ("baseline", "cot", "cod"):
            exp.write_json(self.root / "prompts" / f"gsm8k_{method}.yaml", prompt_fixture(method))
        with (self.root / "data" / "test.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(1319):
                stream.write(json.dumps({"id": index, "question": f"Question {index}?", "answer": "calculation\n#### 42"}) + "\n")
        self.yaml_patch = mock.patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=json.load)})
        self.yaml_patch.start()
        self.addCleanup(self.yaml_patch.stop)
        self.verification_mock = mock.Mock(return_value={"status": "passed"})
        self.resources_patch = mock.patch.dict(sys.modules, {
            "prepare_resources": SimpleNamespace(ROOT=self.root, verify_local_resources=self.verification_mock),
        })
        self.resources_patch.start()
        self.addCleanup(self.resources_patch.stop)
        self.runtime_patch = mock.patch("experiment.get_runtime", return_value=SimpleNamespace(metadata={"gpu_name": "synthetic T4"}))
        self.runtime_mock = self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.generation_patch = mock.patch("experiment.generate_output", side_effect=generated_fixture)
        self.generation_mock = self.generation_patch.start()
        self.addCleanup(self.generation_patch.stop)
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.output = self.root / "runs" / "pilot"

    def run_pilot(self) -> dict:
        """Execute the standard pilot fixture through its public Python interface."""
        return exp.run_experiment(self.root / "config.json", "pilot", self.output)

    def read_rows(self) -> list[dict]:
        """Read persisted result lines for observable assertions and corruption tests."""
        return [json.loads(line) for line in (self.output / "results.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_complete_and_repeat_without_regeneration(self) -> None:
        """Persist all four groups per question and reuse a completed result set."""
        summary = self.run_pilot()
        self.assertEqual(summary["total_records"], 8)
        for group in summary["methods"].values():
            self.assertEqual(group["N"], 2)
            self.assertEqual(group["accuracy"], 1.0)
            self.assertEqual(group["mean_output_tokens"], 2)
        rows = self.read_rows()
        self.assertTrue(all(set(row) == exp.RECORD_KEYS for row in rows))
        manifest = exp.read_json(self.output / "manifest.json")
        self.assertEqual(manifest["warmups"][0]["status"], "completed")
        self.assertEqual(manifest["warmups"][0]["requested_tokens"], 2)
        self.assertTrue(manifest["completed"])
        self.generation_mock.reset_mock()
        self.runtime_mock.reset_mock()
        self.assertEqual(self.run_pilot(), summary)
        self.generation_mock.assert_not_called()
        self.runtime_mock.assert_not_called()

    def test_interruption_saves_context_and_resumes_exact_prefix(self) -> None:
        """Resume after a real injected generation failure with each pair saved once."""
        calls = 0

        def interrupted(runtime: object, prompt: str, max_new_tokens: int) -> dict:
            """Fail the third measured generation after two durable result lines."""
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("synthetic GPU failure")
            return generated_fixture(runtime, prompt, max_new_tokens)

        self.generation_mock.side_effect = interrupted
        with self.assertRaisesRegex(RuntimeError, "synthetic GPU failure"):
            self.run_pilot()
        before = self.read_rows()
        self.assertEqual(len(before), 2)
        failure = exp.read_json(self.output / "failure.json")
        self.assertEqual(failure["stage"], "generation")
        self.assertIn("method", failure)
        self.generation_mock.side_effect = generated_fixture
        self.generation_mock.reset_mock()
        self.run_pilot()
        self.assertEqual(self.read_rows()[:2], before)
        self.assertEqual(len(self.read_rows()), 8)
        self.assertEqual(self.generation_mock.call_count, 7)
        self.assertEqual(len(exp.read_json(self.output / "manifest.json")["warmups"]), 2)

    def test_duplicate_wrong_phase_and_missing_contract_fail(self) -> None:
        """Reject duplicate rows, phase reuse and missing result fields explicitly."""
        self.run_pilot()
        path = self.output / "results.jsonl"
        original = path.read_text(encoding="utf-8")
        path.write_text(original + original.splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra or duplicate"):
            self.run_pilot()
        path.write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            exp.run_experiment(self.root / "config.json", "full", self.output)
        rows = self.read_rows()
        del rows[0]["raw_output"]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            self.run_pilot()

    def test_modified_source_hash_fails(self) -> None:
        """Prevent results from mixing source versions even after completion."""
        self.run_pilot()
        (self.root / "reference" / "upstream.txt").write_text("changed source", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.run_pilot()

    def test_complete_tail_without_lf_recovers_before_append(self) -> None:
        """Keep a valid unterminated record and add its delimiter before resuming."""
        self.run_pilot()
        path = self.output / "results.jsonl"
        original_rows = self.read_rows()
        content = b"\n".join(path.read_bytes().splitlines()[:2])
        path.write_bytes(content)
        self.generation_mock.reset_mock()
        self.run_pilot()
        rows = self.read_rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[:2], original_rows[:2])
        self.assertEqual(self.generation_mock.call_count, 7)
        backups = list(self.output.glob("results.jsonl.recovery-*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), content)
        recovery = exp.read_json(backups[0].with_suffix(".json"))
        self.assertEqual(recovery["action"], "append_missing_lf")
        self.assertEqual(recovery["status"], "completed")
        self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_partial_final_json_preserves_backup_and_completed_rows(self) -> None:
        """Archive a partial write and regenerate only the uncommitted method onward."""
        self.run_pilot()
        path = self.output / "results.jsonl"
        original_rows = self.read_rows()
        committed = b"\n".join(path.read_bytes().splitlines()[:2]) + b"\n"
        original = committed + b'{"schema_version":1,"phase":"pil'
        path.write_bytes(original)
        self.generation_mock.reset_mock()
        self.run_pilot()
        rows = self.read_rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[:2], original_rows[:2])
        self.assertEqual(self.generation_mock.call_count, 7)
        backups = list(self.output.glob("results.jsonl.recovery-*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        recovery = exp.read_json(backups[0].with_suffix(".json"))
        self.assertEqual(recovery["action"], "truncate_uncommitted_tail")
        self.assertEqual(recovery["retained_bytes"], len(committed))
        self.assertEqual(recovery["status"], "completed")

    def test_middle_corruption_fails_with_original_bytes_preserved(self) -> None:
        """Reject committed interior corruption before any tail repair or model load."""
        self.run_pilot()
        path = self.output / "results.jsonl"
        lines = path.read_bytes().splitlines()
        lines[1] = b'{"broken":'
        original = b"\n".join(lines)
        path.write_bytes(original)
        self.runtime_mock.reset_mock()
        with self.assertRaisesRegex(ValueError, r"results.jsonl:2"):
            self.run_pilot()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.output.glob("results.jsonl.recovery-*")), [])
        self.runtime_mock.assert_not_called()

    def test_committed_corrupt_tail_fails(self) -> None:
        """Preserve a malformed LF-terminated final record as an explicit failure."""
        self.run_pilot()
        path = self.output / "results.jsonl"
        original = path.read_bytes() + b'{"broken":\n'
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, r"results.jsonl:9"):
            self.run_pilot()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.output.glob("results.jsonl.recovery-*")), [])

    def test_reordered_dataset_fails_before_identity_and_model(self) -> None:
        """Require original row identities even when provenance validation is mocked."""
        path = self.root / "data" / "test.jsonl"
        path.write_text("\n".join(reversed(path.read_text(encoding="utf-8").splitlines())) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected original ordered integer id 0"):
            self.run_pilot()
        self.verification_mock.assert_called_once_with()
        self.runtime_mock.assert_not_called()
        self.assertFalse((self.output / "manifest.json").exists())

    def test_official_prompt_verification_failure_precedes_identity_and_model(self) -> None:
        """Enforce the preparer's provenance rejection before accepting changed prompts."""
        path = self.root / "prompts" / "gsm8k_cod.yaml"
        trusted_hash = exp.sha256_file(path)
        changed = prompt_fixture()
        changed["system_prompt"] = "changed instruction"
        exp.write_json(path, changed)

        def verify_prepared_prompt() -> dict:
            """Simulate the preparer's saved prompt fingerprint boundary."""
            if exp.sha256_file(path) != trusted_hash:
                raise ValueError("Official prompt copy mismatch")
            return {"status": "passed"}

        self.verification_mock.side_effect = verify_prepared_prompt
        with self.assertRaisesRegex(ValueError, "Official prompt copy mismatch"):
            self.run_pilot()
        self.verification_mock.assert_called_once_with()
        self.runtime_mock.assert_not_called()
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertEqual(exp.read_json(self.output / "failure.json")["stage"], "resource_verification")

    def test_reordered_or_rescored_results_fail(self) -> None:
        """Verify both the execution order and scoring against saved full output."""
        self.run_pilot()
        rows = self.read_rows()
        path = self.output / "results.jsonl"
        reordered = [rows[1], rows[0], *rows[2:]]
        path.write_text("\n".join(json.dumps(row) for row in reordered) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "method order"):
            self.run_pilot()
        rows[0]["output"] = "gold 42 #### 41"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "scoring fields"):
            self.run_pilot()

    def test_config_rejects_unknown_and_duplicate_methods(self) -> None:
        """Require one formal configuration contract and exactly the four methods."""
        exp.write_json(self.root / "config.json", {**self.config, "temperature": 0})
        with self.assertRaisesRegex(ValueError, "exactly"):
            exp.load_config(self.root / "config.json")
        exp.write_json(self.root / "config.json", {**self.config, "methods": ["cot"] * 4})
        with self.assertRaisesRegex(ValueError, "exactly once"):
            exp.load_config(self.root / "config.json")

    def test_dataset_and_prompt_contracts(self) -> None:
        """Reject incomplete datasets and malformed prompt example structures."""
        (self.root / "data" / "test.jsonl").write_text('{"id": 0, "question": "q", "answer": "#### 42"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "1319 rows"):
            exp.load_data(self.root / "data" / "test.jsonl")
        prompt = prompt_fixture()
        prompt["fewshot"][0]["answer"] = 42
        exp.write_json(self.root / "prompts" / "gsm8k_cod.yaml", prompt)
        with self.assertRaisesRegex(ValueError, "invalid fewshot"):
            exp.load_prompts(self.root, 8)


if __name__ == "__main__":
    unittest.main()
