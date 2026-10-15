"""Notebook 27 runs the BiLSTM (v3) and DistilBERT-seed protocols; checked offline, nothing trained."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from reviewnlp.utils.seed import load_config

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "27_seeds_and_bilstm_colab.ipynb"


@pytest.fixture(scope="module")
def cells() -> dict[str, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


@pytest.fixture(scope="module")
def runs(cells, tmp_path_factory) -> dict:
    namespace = {"load_config": lambda path: load_config(str(ROOT / path))}
    exec(cells["parameters"], namespace)
    namespace["CONFIG_DIR"] = tmp_path_factory.mktemp("configs")
    exec(cells["plan"], namespace)
    return namespace


def test_the_notebook_pins_a_commit_that_holds_the_tooling(cells):
    commit = re.search(r'SOURCE_COMMIT = "([0-9a-f]{40})"', cells["parameters"])[1]
    for path in ("scripts/compare_bilstm_clean.py", "scripts/compare_seeds.py", "scripts/benchmark_frontier.py",
                 "configs/bilstm_v2_time.yaml", "docs/experiments/distilbert_seed_robustness/PROTOCOL.md"):
        assert subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], cwd=ROOT).returncode == 0, path
    assert '"--detach", SOURCE_COMMIT' in cells["checkout"] and "GIT_COMMIT != SOURCE_COMMIT" in cells["checkout"]


def test_the_runs_are_exactly_the_protocols_runs(runs):
    assert runs["EXTRA_SEEDS"] == (43, 44)
    assert set(runs["RUNS"]) == {
        ("time", "bilstm"), ("random", "bilstm"), ("hotel", "bilstm"),
        ("time", "bilstm_seed43"), ("time", "bilstm_seed44"),
        ("time", "distilbert_seed43"), ("time", "distilbert_seed44"),
        ("time", "distilbert_lora_scratch_seed43"), ("time", "distilbert_lora_scratch_seed44")}
    assert next(iter(runs["RUNS"])) == ("time", "bilstm")  # the primary comparison's run goes first


def test_seed_configs_change_only_the_seed_and_the_output(runs):
    for (_view, name), (_, config, out_dir, extra) in runs["RUNS"].items():
        derived = yaml.safe_load(Path(config).read_text(encoding="utf-8")) if Path(config).is_absolute() \
            else load_config(str(ROOT / config))
        base_name = "bilstm_v2_time" if "bilstm" in name else "distilbert_v2_time"
        if "seed" not in name:
            assert out_dir == derived["output"]["model_dir"] or out_dir.endswith("_lora_scratch")
            continue
        base = load_config(str(ROOT / "configs" / f"{base_name}.yaml"))
        seed = int(name[-2:])
        assert derived["seed"] == seed
        assert {**derived, "seed": base["seed"], "output": base["output"]} == base, name
        assert extra == (runs["LORA"] if "lora" in name else [])


def test_the_bundle_lands_where_the_comparison_scripts_read(runs, cells):
    import importlib.util

    spec = importlib.util.spec_from_file_location("compare_seeds", ROOT / "scripts" / "compare_seeds.py")
    seeds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seeds)
    expected = {("bilstm_v2", view, "bilstm") for view in ("random", "time", "hotel")}
    expected |= {("bilstm_v2", "time", f"bilstm_seed{seed}") for seed in (43, 44)}
    expected |= {("distilbert_v2", "time", name) for seed in (43, 44) for name in seeds.run_names(seed)}
    landed = {("bilstm_v2" if module == runs["BILSTM"] else "distilbert_v2", view, name)
              for (view, name), (module, _, _, _) in runs["RUNS"].items()}
    assert landed == expected
    assert 'bundle = "bilstm_v2" if module == BILSTM else "distilbert_v2"' in cells["collect"]
    assert "scripts/compare_bilstm_clean.py" in cells["collect"] and "scripts/compare_seeds.py" in cells["collect"]


def test_the_frontier_times_the_sessions_models_on_one_thread(cells):
    assert "RUNS['time', 'bilstm']" in cells["frontier"]
    assert "RUNS['time', 'distilbert_seed43']" in cells["frontier"]
    assert "RUNS['time', 'distilbert_lora_scratch_seed43']" in cells["frontier"]
    assert "scripts/benchmark_frontier.py" in cells["frontier"]
    assert "weights" not in cells["collect"].split("KEEP = ", 1)[1].split("]", 1)[0]
