"""Compare generator prompts/models on frozen upstream results; no training or deployment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from reviewnlp.triage.generator_experiment import (
    CANDIDATES,
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


def pinned_configs(selection):
    from huggingface_hub import HfApi

    if selection:
        config = selection["config"]
        candidate = selection["candidate"]
        expected = {**CANDIDATES[candidate], "prompt_sha256": prompt_fingerprint(candidate)}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("frozen prompt/model configuration differs from this code")
        if selection["upstream_config"] != upstream_config():
            raise ValueError("frozen upstream config differs")
        return {candidate: config}
    revisions = {}
    for candidate in CANDIDATES.values():
        model = candidate["model"]
        if model not in revisions:
            revisions[model] = HfApi().model_info(model).sha
    return {name: {**config, "revision": revisions[config["model"]], "prompt_sha256": prompt_fingerprint(name)}
            for name, config in CANDIDATES.items()}


def compare(output, split, selection=None):
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise ValueError("a CUDA GPU is required for this comparison")
    if split == "holdout" and not selection:
        raise ValueError("holdout requires a frozen selection")
    path = DATA / (split + ".json")
    dataset = read_dataset(path, split)
    dataset_sha = digest_bytes(path.read_bytes())
    configs = pinned_configs(selection)
    if selection:
        # Create this before any API call. Default reruns stop even after a partial final evaluation.
        marker = ROOT / "runs" / ("generator_holdout_" + digest_bytes(json.dumps(selection, sort_keys=True).encode()) + ".lock")
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(str(output) + "\n")
    cache = prepare_upstream(dataset["reviews"], dataset_sha, output)
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
                                "json_valid": False, "actions": [], "dropped": 0,
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
    if records:
        export_annotation(output, dataset["reviews"], records)
    files = {name: digest_bytes((output / name).read_bytes()) for name in
             ("upstream.json", "results.jsonl", "human_review.csv", "annotation_key.json") if (output / name).exists()}
    info = {"split": split, "dataset_sha256": dataset_sha, "reviews": len(dataset["reviews"]),
            "candidates": configs, "upstream_config": upstream_config(), "api_cost": cache["api_cost"],
            "warmups": warmups, "summary": summaries, "execution_complete": complete,
            "quality_evaluated": False, "files": files, "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "finished_utc": datetime.now(timezone.utc).isoformat()}
    if selection:
        info["selection"] = selection
    write_json(output / "run.json", info)
    lines = ["# Generator comparison", "", "Synthetic cases; structural counts are not human usefulness.",
             "", "| Candidate | Attempted | JSON valid | Reviews with accepted actions | Budget hits | Errors |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, count in summaries.items():
        lines.append(f"| {name} | {count['reviews_attempted']} | {count['parser_json_valid']} | "
                     f"{count['reviews_with_accepted_actions']} | {count['token_budget_hits']} | {count['generation_errors']} |")
    lines.extend(["", "Human quality: pending. Fill human_review.csv using the protocol.",
                  "All cases are queried for generator diagnosis; production_routed preserves the actual routing.",
                  "Case latency excludes warmup; local GPU billing is unknown.",
                  "API cost: " + json.dumps(cache["api_cost"])])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "freeze", "holdout"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dev-run", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--candidate", choices=list(CANDIDATES))
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
    if not args.allow_external_api:
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
    return compare(output, args.stage, selection)


if __name__ == "__main__":
    raise SystemExit(main())
