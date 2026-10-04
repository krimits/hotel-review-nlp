"""Eager model loading and explicit diagnostics across the ZeroGPU process boundary.

Import spaces in the app before importing this module. GPU workers return only plain data;
changes to a worker's issue list are not assumed to appear in the parent process.
"""

from __future__ import annotations

import copy
import os
import platform
from dataclasses import asdict
from importlib import metadata

from reviewnlp.triage.demo_service import (
    QWEN_REPO,
    QWEN_REVISION,
    DemoService,
    PinnedSentiment,
    clear_generation_diagnostics,
    make_generator,
    public_stage_reports,
)
from reviewnlp.triage.evidence_generator import VERSION
from reviewnlp.triage.qwen_generator import ActionSignals, GenerationResult, _load_qwen


def load_space_qwen():
    """ZeroGPU CUDA placement happens during app initialization, before requests."""
    if os.environ.get("SPACES_ZERO_GPU", "").lower() not in {"1", "true", "yes"}:
        return _load_qwen(QWEN_REPO, None, QWEN_REVISION)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_REPO, revision=QWEN_REVISION, dtype=torch.bfloat16).eval().to("cuda")
    return tokenizer, model


def run_worker(generator, review: str, signals: dict) -> dict:
    """Execute both Qwen calls inside one GPU allocation, then discard raw diagnostics."""
    failed = False
    try:
        try:
            generated = generator.generate(review, ActionSignals(**signals))
        except Exception:
            failed = True
            generated = GenerationResult("", False, QWEN_REPO, VERSION)
        return {"generated": asdict(generated), "generation_failed": failed,
                "issues": copy.deepcopy(generator.issues),
                "review_reasons": list(generator.review_reasons),
                "stage_reports": public_stage_reports(generator)}
    finally:
        clear_generation_diagnostics(generator)


class WorkerGenerator:
    """Copy returned assessments into the parent; never rely on mutations made in a fork."""

    model_name, prompt_version, requires_exact_quotes = QWEN_REPO, VERSION, True

    def __init__(self, worker):
        self.worker = worker
        self.issues, self.review_reasons, self.stage_reports = [], [], []

    def generate(self, review, signals):
        clear_generation_diagnostics(self)
        try:
            response = self.worker(review, asdict(signals))
        except Exception:
            self.stage_reports = [{"stage": "qwen_runtime", "status": "error",
                                   "error": "gpu_unavailable_or_timeout"}]
            raise RuntimeError("qwen_worker_unavailable") from None
        self.issues = response["issues"]
        self.review_reasons = response["review_reasons"]
        self.stage_reports = response["stage_reports"]
        if response["generation_failed"]:
            raise RuntimeError("qwen_generation_failed") from None
        return GenerationResult(**response["generated"])


class SpaceRuntime:
    """Load once; a startup failure stays visible and does not trigger repeated downloads."""

    def __init__(self, worker, *, wrapper=None, generator=None):
        self.wrapper = wrapper if wrapper is not None else PinnedSentiment()
        self.generator = generator if generator is not None else make_generator(loader=load_space_qwen)
        self.service = DemoService(self.wrapper, WorkerGenerator(worker))
        self.initialized, self.startup_failure = False, None

    def preload(self):
        if self.initialized:
            return
        self.initialized = True
        stage = "sentiment_load"
        try:
            self.wrapper._models()
            stage = "qwen_load"
            self.generator.extractor._models()
        except Exception:
            self.startup_failure = {"stage": stage, "status": "error", "error": "model_load_failed"}

    def run_qwen(self, review, signals):
        return run_worker(self.generator, review, signals)

    def information(self):
        bundle = getattr(self.generator, "_bundle", None)
        model = bundle[1] if bundle else None
        versions = {}
        for package in ("torch", "transformers", "huggingface-hub", "accelerate", "gradio", "pydantic", "spaces"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        return {"ready": self.initialized and self.startup_failure is None,
                "startup_failure": self.startup_failure,
                "qwen_device": str(getattr(model, "device", "not_loaded")),
                "qwen_precision": str(getattr(model, "dtype", "not_loaded")),
                "python": platform.python_version(), "versions": versions,
                "gpu_duration_seconds": 60, "gpu_billing": "unknown",
                "zero_gpu": os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "true", "yes"}}
