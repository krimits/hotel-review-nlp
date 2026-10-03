"""Compare generator prompts/models on frozen upstream results; no training or deployment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from reviewnlp.triage.generator_experiment import (
    CANDIDATES,
    DEFAULT_CANDIDATES,
    RecordingTransport,
    builder,
    cost_summary,
    digest_bytes,
    export_annotation,
    freeze_selection,
    prompt_fingerprint,
    read_dataset,
    run_candidate,
    structural_summary,
    validate_upstream,
)
from reviewnlp.triage.jev_client import JevClient, JevConfig, urllib_transport
from reviewnlp.triage.pipeline import TriagePipeline
from reviewnlp.triage.questions import QUESTIONS_SHA256, QUESTIONS_VERSION
from reviewnlp.triage.qwen_generator import MAX_NEW_TOKENS, ActionSignals, QwenActionGenerator
from reviewnlp.triage.routing import RoutingThresholds
from reviewnlp.triage.staged_generator import TwoStageGenerator, messages_measures

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "experiments" / "triage_generator"
HUB_REPO = "krimits/distilbert-hotel-reviews"
HUB_REVISION = "7306aebcaaebc00d579f5d0a91001ae376f18158"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upstream_config():
    return {"distilbert_repo": HUB_REPO, "distilbert_revision": HUB_REVISION, "route": "openrouter",
            "jev_model_requested": "jev-latest", "questions_version": QUESTIONS_VERSION,
            "questions_sha256": QUESTIONS_SHA256, "thresholds": RoutingThresholds().as_dict()}


def prepare_upstream(rows, dataset_sha, output):
    cache_path = output / "upstream.json"
    config = upstream_config()
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        validate_upstream(cache, rows, dataset_sha, config)
        return cache
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required; no requests sent")
    from huggingface_hub import snapshot_download
    from reviewnlp.serving.model_wrapper import ModelWrapper

    wrapper = ModelWrapper(model_type="encoder", model_path=snapshot_download(HUB_REPO, revision=HUB_REVISION))
    transport = RecordingTransport(urllib_transport)
    jev = JevClient(JevConfig(enabled=True, route="openrouter", api_key=key), transport=transport)
    pipeline = TriagePipeline(wrapper, jev, generator=None)
    cache = {"dataset_sha256": dataset_sha, "config": config, "records": []}
    for row in rows:
        start = len(transport.attempts)
        result = pipeline.run(row["text"])
        cache["records"].append({"id": row["id"], "sentiment": result.sentiment.model_dump(mode="json"),
                                 "complaints": result.complaints.model_dump(mode="json"),
                                 "routing": result.routing.model_dump(mode="json"),
                                 "timings": result.timings.model_dump(mode="json"),
                                 "api_attempts": transport.attempts[start:]})
        # Preserve incurred calls even if a later upstream validation fails.
        cache["api_cost"] = cost_summary(transport.attempts)
        write_json(cache_path, cache)
        print("Upstream:", len(cache["records"]), "/", len(rows), flush=True)
    validate_upstream(cache, rows, dataset_sha, config)
    return cache


def pinned_configs(selection, candidates=DEFAULT_CANDIDATES, reference=None):
    if selection:
        config = selection["config"]
        candidate = selection["candidate"]
        expected = {**CANDIDATES[candidate], "prompt_sha256": prompt_fingerprint(candidate)}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("frozen prompt/model configuration differs from this code")
        if selection["upstream_config"] != upstream_config():
            raise ValueError("frozen upstream config differs")
        return {candidate: config}
    if not candidates or len(set(candidates)) != len(candidates) or any(name not in CANDIDATES for name in candidates):
        raise ValueError("choose distinct known candidates")
    revisions = {}
    if reference:
        # Reuse the reference weights instead of resolving today's mutable model main.
        for name in candidates:
            model = CANDIDATES[name]["model"]
            matching = [config for config in reference["info"]["candidates"].values() if config["model"] == model]
            hashes = {config.get("revision") for config in matching}
            if len(hashes) != 1 or not re.fullmatch(r"[a-f0-9]{40}", next(iter(hashes), "") or ""):
                raise ValueError("reference does not identify one pinned revision for " + model)
            revisions[model] = hashes.pop()
            if name in reference["info"]["candidates"]:
                old = reference["info"]["candidates"][name]
                expected = {**CANDIDATES[name], "prompt_sha256": prompt_fingerprint(name)}
                if any(old.get(key) != value for key, value in expected.items()):
                    raise ValueError("reference baseline prompt/config differs: " + name)
    else:
        from huggingface_hub import HfApi

        for name in candidates:
            model = CANDIDATES[name]["model"]
            if model not in revisions:
                revisions[model] = HfApi().model_info(model).sha
    return {name: {**config, "revision": revisions[config["model"]], "prompt_sha256": prompt_fingerprint(name)}
            for name in candidates for config in [CANDIDATES[name]]}


def read_reference_run(path, rows, dataset_sha):
    """Verify a completed dev run before reusing any prediction or model revision."""
    info_bytes, cache_bytes = (path / "run.json").read_bytes(), (path / "upstream.json").read_bytes()
    info, cache = json.loads(info_bytes), json.loads(cache_bytes)
    if (info.get("split") != "dev" or info.get("execution_complete") is not True
            or info.get("dataset_sha256") != dataset_sha or info.get("reviews") != len(rows)):
        raise ValueError("reference must be a complete dev run on these same cases")
    if info.get("max_new_tokens") != MAX_NEW_TOKENS or info.get("do_sample") is not False:
        raise ValueError("reference decoding settings differ")
    expected = info.get("files", {}).get("upstream.json")
    if not expected or digest_bytes(cache_bytes) != expected:
        raise ValueError("reference upstream fingerprint differs")
    if info.get("upstream_config") != upstream_config():
        raise ValueError("reference upstream configuration differs")
    validate_upstream(cache, rows, dataset_sha, upstream_config())
    if info.get("jev_model_resolved") != cache["records"][0]["complaints"]["model"]:
        raise ValueError("reference responding Jev model differs")
    return {"info": info, "cache": cache, "cache_bytes": cache_bytes,
            "run_bytes": info_bytes, "run_sha256": digest_bytes(info_bytes), "upstream_sha256": expected}


def compare(output, split, selection=None, *, candidates=DEFAULT_CANDIDATES, reference_run=None):
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise ValueError("a CUDA GPU is required for this comparison")
    if split == "holdout" and not selection:
        raise ValueError("holdout requires a frozen selection")
    if reference_run is not None and (split != "dev" or selection is not None):
        raise ValueError("reference reuse is only for development")
    path = DATA / (split + ".json")
    dataset = read_dataset(path, split)
    dataset_sha = digest_bytes(path.read_bytes())
    reference = read_reference_run(reference_run, dataset["reviews"], dataset_sha) if reference_run is not None else None
    configs = pinned_configs(selection, candidates, reference)
    if selection:
        # Create this before any API call. Default reruns stop even after a partial final evaluation.
        marker = ROOT / "runs" / ("generator_holdout_" + digest_bytes(json.dumps(selection, sort_keys=True).encode()) + ".lock")
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(str(output) + "\n")
    if reference:
        cache = reference["cache"]
        (output / "upstream.json").write_bytes(reference["cache_bytes"])
        (output / "reference_run.json").write_bytes(reference["run_bytes"])
        print("Reusing reference upstream and pinned weights; no new Jev requests.", flush=True)
    else:
        cache = prepare_upstream(dataset["reviews"], dataset_sha, output)
    jev_model = cache["records"][0]["complaints"]["model"]
    if selection and selection.get("jev_model_resolved") != jev_model:
        raise ValueError("Jev changed after development selection; final generator evaluation was not run")
    records, warmups, bundle, active_model = [], {}, None, None
    try:
        for name, config in configs.items():
            if active_model != config["model"]:
                bundle = None
                gc.collect()
                torch.cuda.empty_cache()
            generator = QwenActionGenerator(config["model"], device="cuda", revision=config["revision"],
                                            message_builder=builder(name),
                                            prompt_version=config["prompt"] + ("" if config["sentiment_metadata"] else "-no-sentiment"))
            if name == "F":
                def action_factory(pending, config=config):
                    return QwenActionGenerator(config["model"], device="cuda", revision=config["revision"],
                                               message_builder=lambda review, signals: messages_measures(review, pending),
                                               prompt_version=config["prompt"] + ":measures")

                generator = TwoStageGenerator(generator, action_factory)
            if bundle is not None:
                generator._bundle = bundle  # experiment shares identical weights across prompt variants
            start = time.perf_counter()
            # Same independent warmup for every candidate; excluded from case latency.
            try:
                generator.generate("Thank you for the pleasant stay.", ActionSignals("positive", 0.99))
                torch.cuda.synchronize()
            except Exception as error:  # noqa: BLE001 - export a failed run before returning exit 1
                records.append({"candidate": name, "id": dataset["reviews"][0]["id"],
                                "production_routed": cache["records"][0]["routing"]["qwen_triggered"],
                                "upstream_sha256": None, "raw": None, "hit_token_budget": None,
                                "json_valid": False, "full_json_valid": False, "duplicate_json_keys": [],
                                "actions": [], "dropped": 0,
                                "seconds": round(time.perf_counter() - start, 6),
                                "error": "load_or_warmup:" + type(error).__name__})
                with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                del generator
                break
            warmups[name] = {"load_and_warmup_seconds": round(time.perf_counter() - start, 3)}
            new_records = run_candidate(name, dataset["reviews"], cache, generator)
            records.extend(new_records)
            bundle, active_model = generator._bundle, config["model"]
            del generator
            with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(name, json.dumps(structural_summary(new_records)), flush=True)
            if any(record["error"] for record in new_records):
                break
    finally:
        bundle = None
        gc.collect()
        torch.cuda.empty_cache()
    summaries = {name: structural_summary([row for row in records if row["candidate"] == name]) for name in configs}
    complete = all(count["reviews_attempted"] == len(dataset["reviews"]) and not count["generation_errors"]
                   for count in summaries.values())
    annotation_seed = 20261016 if "F" in configs else 20261003
    if records:
        export_annotation(output, dataset["reviews"], records, shuffle_seed=annotation_seed)
    files = {name: digest_bytes((output / name).read_bytes()) for name in
             ("upstream.json", "results.jsonl", "human_review.csv", "annotation_key.json", "reference_run.json")
             if (output / name).exists()}
    info = {"split": split, "dataset_sha256": dataset_sha, "reviews": len(dataset["reviews"]),
            "candidates": configs, "upstream_config": upstream_config(),
            "api_cost": cost_summary([]) if reference else cache["api_cost"],
            "upstream_reused": bool(reference),
            "jev_model_resolved": jev_model,
            "warmups": warmups, "summary": summaries, "execution_complete": complete,
            "quality_evaluated": False, "annotation_shuffle_seed": annotation_seed,
            "files": files, "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
            "generation_budget": {name: {"max_new_tokens_per_call": MAX_NEW_TOKENS,
                                          "max_calls_per_review": config.get("max_generation_calls", 1)}
                                  for name, config in configs.items()},
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "finished_utc": datetime.now(timezone.utc).isoformat()}
    if reference:
        info["reference_run"] = {"run_sha256": reference["run_sha256"],
                                 "upstream_sha256": reference["upstream_sha256"],
                                 "source_commit": reference["info"].get("source_commit"),
                                 "historical_api_cost": cache["api_cost"]}
    if selection:
        info["selection"] = selection
    write_json(output / "run.json", info)
    lines = ["# Generator comparison", "", "Synthetic cases; structural counts are not human usefulness.",
             "", "| Candidate | Attempted | Parser accepted JSON | Whole JSON, unique keys | Reviews with accepted actions | Budget hits | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, count in summaries.items():
        lines.append(f"| {name} | {count['reviews_attempted']} | {count['parser_json_valid']} | "
                     f"{count['full_json_valid']} | {count['reviews_with_accepted_actions']} | "
                     f"{count['token_budget_hits']} | {count['generation_errors']} |")
    lines.extend(["", "Human quality: pending. Fill human_review.csv using the protocol.",
                  "All cases are queried for generator diagnosis; production_routed preserves the actual routing.",
                  "Case latency excludes warmup; local GPU billing is unknown.",
                  "API cost: " + json.dumps(cache["api_cost"])])
    lines.append("Whole JSON requires one complete JSON object without duplicate keys; it is not semantic validation.")
    if "F" in configs:
        count = summaries["F"]
        lines.extend(["F is a workflow experiment: up to two 400-token calls, not the same inference budget as C.",
                      f"F issue calls: {count['issue_calls']}; measure calls: {count['measure_calls']}; "
                      f"workflow failures: {count['workflow_failures']}.",
                      "F raw is assembled from issue IDs; actual model text and timings are in stages[].",
                      "Duplicate JSON keys are rejected by the current parser; historical sheets are not reinterpreted."])
    if reference:
        lines.extend(["Reused upstream: no new Jev calls. The upstream API cost above belongs to the reference run.",
                      "Current API attempts: 0. Local GPU billing remains unknown."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "freeze", "holdout"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dev-run", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--candidate", choices=list(CANDIDATES))
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--reference-run", type=Path,
                        help="reuse a completed dev run's upstream and pinned weights; no new Jev requests")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--allow-external-api", action="store_true")
    args = parser.parse_args()
    if args.stage == "freeze":
        if not all((args.dev_run, args.ratings, args.candidate, args.selection)):
            parser.error("freeze needs --dev-run --ratings --candidate --selection")
        selection = freeze_selection(args.dev_run, args.ratings, args.candidate)
        # Do not overwrite a previously frozen configuration.
        with args.selection.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(selection, indent=2) + "\n")
        print("Selection frozen. Holdout has not been opened.")
        return 0
    if args.reference_run is not None and args.stage != "dev":
        parser.error("--reference-run is only valid for dev")
    if len(args.candidates) != len(set(args.candidates)):
        parser.error("--candidates must be distinct")
    if not args.allow_external_api and args.reference_run is None:
        parser.error("this comparison needs --allow-external-api for the synthetic upstream cases")
    if not args.output:
        parser.error("--output is required")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()) or output == (ROOT / "runs").resolve():
        parser.error("--output must be a new folder inside runs/")
    output.mkdir(parents=True, exist_ok=False)
    selection = None
    if args.stage == "holdout":
        if not args.selection:
            parser.error("holdout requires --selection from the freeze step")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
    return compare(output, args.stage, selection, candidates=args.candidates, reference_run=args.reference_run)


if __name__ == "__main__":
    raise SystemExit(main())
