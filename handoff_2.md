# Handoff 2 — SSFL framework (state as of 2026-11-23)

Continues `handoff.md`. Repository: https://github.com/ozanyildirim22/FederatedFramework, branch `main`,
latest commit `0e4d001`. The working tree is clean and pushed. Experiments run on Google Colab (96 GB GPU)
through `colab/SSFL_Colab.ipynb`; local development uses a Windows `.venv`.

**Working rule:** commit and push only when the user explicitly says "commit and push".

---

## 1. What the framework contains now

| Area | Content |
|---|---|
| Algorithms | FedAvg, FedProx, FedAvg/FedProx + FixMatch, FedAvg/FedProx + UDA, FedMatch, FedTriNet, FedLabel (+ supervised, pseudolabel) |
| Datasets | CIFAR-10 (ResNet-34, 100 clients, C = 0.1), OrganAMNIST and PathMNIST (ResNet-18, 10 clients, C = 1.0), synthetic (tests) |
| Partitioning | Dirichlet α (thesis: 0.1 and 1.0) with a balance cap; labeled/unlabeled split per client at random (ADD) or stratified |
| Metrics | accuracy, loss, macro precision / recall / F1 (confusion matrix); relative ratio method / FedAvg (ADD Eq. 7) |
| GUI | "SSFL Studio" (Gradio 6): Experiment, Thesis grid, Results (leaderboard, run details, curves, thesis table + LaTeX), Guide |
| Run records | `config.yaml` (every parameter incl. defaults), `metrics.csv` (per round), `summary.json` (status, timing, environment, best results), `global_model.pt` |
| Speed | Fast GPU mode: `data.augment_device=gpu` + `strategy.precision=bf16`; automatic DataLoader workers (`data.num_workers=-1`) |

## 2. FedLabel implementation (checked against the ICCV 2023 paper, its appendix, and the ADD)

1. **Step 1:** w_L starts from W_t; τ SGD steps on labeled data (`sup_epochs`, or `sup_steps` for an exact τ).
2. **Step 2:** w_U starts from W_t; τ′ steps on unlabeled data (`unsup_epochs` / `unsup_steps`). Per sample, the
   global and local model are scored with h(·) = variance of the logits (`confidence`); the more confident one
   (ties → global) gives the hard pseudo-label if max softmax **> β** (strict); cross-entropy on the strong view.
3. **Step 3:** KL(s(w_U) ‖ s̄*) where the non-selected model predicts the same class, weighted
   λ_k = λ₀ · h(s̄*) / h(s*) (≤ λ₀).
4. The client uploads Δw_L + Δw_U; aggregation weight r_k = samples actually used (labeled + unlabeled kept).

Parameters (`ssfl/strategies/algorithms.py`): `beta=0.5`, `lambda0=1.0`, `lambda_u=1.0`, `sup_epochs`/`unsup_epochs`
(presets: 1), `sup_steps`/`unsup_steps`, `confidence` (variance | prob_variance | max_prob | entropy),
`selection` (adaptive | local | global), `kl_direction` (add | teacher), `weight_by_used`.
Old names `tau` → `beta` and `lambda_kl` → `lambda0` are still accepted.

## 3. Changes made in this conversation (latest first)

### After `0e4d001` (not committed yet)
- **CIFAR-10 local work:** `strategy.local_steps: 15` in the preset (every method; FedLabel τ = τ′ = 15). One
  epoch was only ~3 labeled + ~1 unlabeled step per round there. The grid skips finished runs by name, so the
  old CIFAR-10 runs need a new `output_dir` (or deleting) before rerunning.
- **Speed options:** `strategy.cudnn_benchmark` (on in the CIFAR-10 preset and in ⚡ Fast GPU mode) and
  `python -m ssfl -c CONFIG --bench N`, which times N rounds with / without channels_last and the cuDNN benchmark.
  On the RTX 3050 Laptop neither helps, and a framework step costs the same as a raw PyTorch step (~74 ms,
  ResNet-34, batch 32). The A100's ~45 ms for a batch-32 step looks CPU-launch-bound.
- **`strategy.compile`** (none | default | reduce-overhead | max-autotune-no-cudagraphs): `torch.compile` of the
  trained model only, via `Strategy.training_forward`; frozen copies and evaluation stay eager. Off by default;
  `--bench` times `default` and `reduce-overhead` too. Untested on a real GPU so far (no Triton on Windows); the
  tests run the wrapper path with the eager backend.
- **GUI ⏱️ Speed check button** (Experiment tab): `SpeedCheck` in `gui.py` runs `bench.bench_config` on the
  current form (FedLabel if selected, else the first algorithm) in a background thread; results in the
  "⏱️ Speed check" dashboard tab. Launch / Run grid refuse to start while it runs; Stop ends it after the
  current setting.

### Appendix alignment (commit `0e4d001`)
- **β default 0.5** (appendix Fig. 10: 0.3–0.5 best, 0.9 worst); previously 0.95. Presets, thesis grid and code changed.
- **3 seeds averaged:** grid key `seeds` (top level or per experiment). Thesis grid: `seeds: [42, 43, 44]`.
  The preset seed (42) keeps the old run names; other seeds add `_s43` / `_s44`. The grid runs all seed-42 runs
  first, then 43, then 44. Results thesis table and LaTeX show **mean ± std**.
- **Paper's test protocol (optional):** `data.holdout=resplit data.val_ratio=0.05 data.test_ratio=0.15` pools
  all splits (incl. MedMNIST's val split) into 80 / 5 / 15 train / val / test. `data.val_ratio > 0` with
  `holdout=official` holds out part of the training set. Runs then log `val_*` metrics; the summary and the
  Results tab report **test accuracy at the best validation round**. Default is unchanged (official test set, ADD).
- **Computation cost (paper Table 7):** every round logs `client_s` (local training per client) and `server_s`;
  summary has `mean_client_s`, `mean_server_s`; shown in Run details.
- **New ablations** (seed 42 only):
  - OrganAMNIST 5% labels, α 0.1: β 0.3 / 0.7 / 0.8 / 0.9 / 0.95, entropy confidence (plus the older
    λ₀ 0 / 0.5 / 2, τ 5 / 20 / 50, selection local / global, max-prob confidence).
  - OrganAMNIST 20% labels, α 0.1: τ 5 / 20 / 50 (appendix Fig. 12).
  - CIFAR-10 20% labels, α 0.1: λ₀ 0 / 2, β 0.3 / 0.9 / 0.95, entropy, no strong augmentation (β 0.5 and 0.95).
- Thesis grid now has **386 runs**: 146 with seed 42, then 120 each for seeds 43 and 44.
- GUI fixes: rounds slider up to 2,000 (presets use 1,000; it used to error); grid runs without a tag are no longer
  labelled like "FedAvg (a0p1)".

### Earlier in the conversation (commits up to `651e3dc`)
- No horizontal flips for OrganAMNIST (left/right organ classes); `data.hflip` = auto / on / off.
  **OrganAMNIST runs made before this fix used flips and should be rerun.**
- FedLabel aligned with the ADD and the paper; UDA and the paper's baselines added (FedProx+FixMatch/UDA via
  `strategy.prox_mu`); macro metrics; random labeled split; RandAugment N=1, M=10 without Cutout.
- ResNet-34 for CIFAR-10.
- Run records with full parameters and timing; Run details view; GUI download and chart fixes.

## 4. Hyperparameters: known vs. chosen

From the paper / appendix / ADD: models, client counts and participation, Dirichlet α, label ratios, random labeled
split, 80/5/15 split and 3 seeds (appendix), variance confidence, strict β threshold, λ_k formula, Δw_L + Δw_U,
r_k weighting, SGD, 1,000 rounds (ADD), RandAugment N=1 M=10 (ADD; the appendix text contradicts itself).

**Not given anywhere — the framework's own choices (state them in the thesis):** learning rate 0.03, SGD momentum 0.9
(Nesterov), weight decay 5e-4, labeled batch 32, unlabeled batch 7× (μ = 7), local work: CIFAR-10 15 SGD steps per round for every method
(`strategy.local_steps: 15`, FedLabel τ = τ′ = 15; one epoch was only ~3 + ~1 steps there), MedMNIST τ = τ′ = 1 local epoch, λ₀ = 1,
λ_u = 1, GroupNorm, gradient clipping 5.0, no pretraining, MedMNIST at 28×28, evaluation every 5 rounds,
baseline settings (FixMatch τ 0.95; UDA T 0.4 / mask 0.8; FedProx μ 0.01; FedMatch τ 0.85, λ_s 10, λ_iccs 0.01,
λ_L2 10, λ_L1 1e-4, 2 helpers; FedTriNet 200 warm-up rounds, α 0.93).

## 5. Known deviations and open points

- **Client participation:** the paper samples 30% of OrganAMNIST clients per round; the ADD and presets use 100%.
- **BloodMNIST** is in the paper but not in the ADD; not added.
- **Earlier FedLabel results used β 0.95.** The grid would skip them as finished (same names): rerun with a new
  `output_dir` or `--no-resume --only fedlabel`.
- **Switching `data.holdout`** changes every run without changing names: use a separate `output_dir`.
- FedLabel is about 10× slower per round than FedAvg (extra unlabeled pass through three models).
- ADD document issues to fix (from the earlier review): Eq. 5 notation, Eq. 6 written as a binary formula, the ratio
  baseline ambiguity, KL direction, odd Table 2 numbers, uncited references [4][9][10], the FedMatch 10% claim,
  missing Table 1.

## 6. How to run

```bash
python -m pytest -q tests                                            # 99 tests
python -m ssfl -c configs/quick_test.yaml                            # seconds, synthetic data
python -m ssfl -c configs/organamnist_fedlabel.yaml --set data.labeled_ratio=0.05 server.rounds=50
python -m ssfl --grid configs/grids/thesis.yaml --dry-run            # list the 386 runs
python -m ssfl --grid configs/grids/thesis.yaml --only organamnist --set output_dir=<runs dir>
python -m ssfl --gui                                                 # SSFL Studio on http://localhost:7860
```

On Colab: run the notebook from the top (it clones `main`). GUI cell:
`!python -m ssfl --gui --share --data-dir "{DATA_DIR}" --runs-dir "{RUNS_DIR}"`. With `USE_DRIVE = True` the grid
resumes after a timeout. On large GPUs set `FAST = True` (or tick ⚡ Fast GPU mode in the GUI).
The GUI header's build chip shows the running commit.

## 7. Key files

| File | Role |
|---|---|
| `ssfl/strategies/algorithms.py` | FixMatch, UDA, FedMatch, FedTriNet, FedLabel, confidence scores |
| `ssfl/strategies/base.py` | local loop, precision, FedProx term, `round_batches` |
| `ssfl/data/manager.py` | loading, holdout split (`make_holdout`), partitioning, loaders, auto workers |
| `ssfl/data/datasets.py`, `augment.py`, `gpu.py`, `partition.py` | loaders, CPU / GPU augmentation, splits |
| `ssfl/experiment.py` | federated loop, validation, timing, `summary.json` |
| `ssfl/grid.py` | grid expansion (labels × α × algorithms × seeds), resume |
| `ssfl/gui.py` | SSFL Studio; Results helpers `thesis_table`, `thesis_display`, `to_latex` |
| `configs/*.yaml`, `configs/grids/thesis.yaml` | presets and the thesis grid |
| `tests/test_framework.py`, `tests/test_variants.py` | 99 tests |
