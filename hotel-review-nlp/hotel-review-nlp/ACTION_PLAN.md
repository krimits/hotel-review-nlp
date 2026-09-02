# 4-week action plan

Each week ships something demoable. The `make` targets are the definition of done.

## Week 1 - data + classical baselines (CPU only)

- [ ] Create the GitHub repo, push this skeleton, enable the CI badge (`ci.yml` is included)
- [ ] Download the Booking.com 515K CSV (see `data/raw/README.md`), run `make data`
- [ ] Skim `notebooks/01_eda.ipynb` - confirm the label-source chart and length stats
- [ ] Run `make baselines` - two feature views × two models, metrics land in `runs/classical/`
- [ ] Fill the first two rows of the README results table

**Interview story ready by Friday:** "unambiguous labels from the schema, not score thresholds."

## Week 2 - PyTorch baseline + from-scratch LoRA

- [ ] `make bilstm` - verify early stopping fires and dev macro-F1 is reported
- [ ] Read the LoRA paper (§4 is enough) alongside `reviewnlp/lora/lora.py` line by line;
      be ready to explain `kaiming_uniform(a=√5)` and why `B=0` makes ΔW=0 at step 0
- [ ] `pytest tests/test_lora.py -v` locally; on a machine with peft installed, watch the
      equivalence test pass too
- [ ] `python -m reviewnlp.llm.train_distilbert_lora ...` - first GPU run of the repo
- [ ] Compare trainable-params and macro-F1 vs the full fine-tune

**Story:** "I implemented the paper from the equations, and proved it against PEFT with tests."

## Week 3 - the LLM fine-tune (Colab) + unified benchmark

- [ ] Open `notebooks/02_train_qlora_colab.ipynb` on a free-T4 Colab, run top to bottom
      (~30-45 min: setup + training)
- [ ] Sanity-check with the two example reviews, then download the adapter
- [ ] Run `make benchmark` with every model present - `runs/benchmark/results.json` +
      `confusion.png` + `latency.png` generated
- [ ] Read the McNemar section of DESIGN.md; paste the significance results into the README
- [ ] Run `notebooks/03_llm_annotation_demo.ipynb` - the unlabeled-CSV story

**Story:** "one frozen test set, five families, significance-tested differences."

## Week 4 - serving, quantization, polish

- [ ] `MODEL_TYPE=encoder MODEL_PATH=runs/distilbert uvicorn reviewnlp.serving.app:app`
- [ ] `bash scripts/demo_api.sh` - capture a terminal screenshot for the README
- [ ] `make quantbench` - INT8 before/after table into `runs/quantization/benchmark.json`
- [ ] `locust -f scripts/load_test.py --headless -u 20 -t 60s` - RPS + p95 into the README
- [ ] Docker: `make docker` and run the container with the stub model (CI-safe smoke test)
- [ ] Record a 3-4 min Loom walking through: README table → LoRA tests → Colab run → API demo
- [ ] Final polish: fill all result placeholders, clean runs/ from git, tag `v1.0.0`

**Deliverables:** repo link + Loom + a 6-line "what I'd do with more compute" paragraph
(DESIGN.md §8) for the application email.

## Standing rules during the 4 weeks

1. Every experiment goes through a config file - no magic numbers in cells.
2. Commit after every green test run; small commits tell the story better.
3. If a run fails on Colab, capture the cell output in the commit message - debugging
   evidence is portfolio gold.
