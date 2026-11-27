# Handoff 3 — this conversation (2026-11-27)

Repository: https://github.com/ozanyildirim22/FederatedFramework, branch `main`. Pushed up to `cdf71d5`.
**Not committed yet:** parallel runs (`ssfl/parallel.py`, `tests/test_parallel.py`, edits in `experiment.py`,
`grid.py`, `__main__.py`, `gui.py`, `README.md`) and this file. All 121 tests pass.

**Working rules from the user**
- Commit and push only when the user says so.
- Use **Fast GPU mode** for every framework run (GUI checkbox; CLI: `data.augment_device=gpu
  strategy.precision=bf16 strategy.cudnn_benchmark=true`). On the RTX 3050 laptop it is not always faster.
- The user is a beginner: explain simply, with small tables.
- Experiments run on Colab (A100 80 GB) through the GUI's public `gradio.live` link; run folders are in
  `/content/drive/MyDrive/ssfl/runs`. The Colab GUI still runs build `f5b1f0e`: restart its cell (after a
  pull) to get pause/resume and parallel runs, but only when nothing is running there.

---

## 1. Code changes in this conversation

| Commit | Change |
|---|---|
| `bc4f37c` | FedLabel client replaced by the `test1` version (branch `test1` of `C:\Users\ozan\Desktop\Federated Learning`, `fedlabel/client.py`); framework preprocessing kept. `--clear-runs` CLI (`ssfl/cleanup.py`). |
| `ac2b18e` | GUI: 🗑️ Clear runs tab + button next to Refresh in Results. |
| `f5b1f0e` | Presets `configs/organamnist_fedlabel_test1.yaml`, `configs/cifar10_fedlabel_test1.yaml` (test1 settings: batch 20, μ = 1, lr 0.01, β 0.95, τ/τ′ 30/30 resp. 15/15, 800 rounds, BatchNorm). |
| `cdf71d5` | **Pause / resume**: `checkpoint.pt` + `checkpoint.json` in the run folder (global model, history, all RNG states, client sampling, aggregator + strategy state); `server.checkpoint_every` (default 25); GUI ⏸ Pause / ⏯ Resume (paused queue keeps waiting runs as `queued` folders); CLI Ctrl+C = pause, `--resume RUN_DIR`, `--list-resumable`; grid continues paused runs. Resumed = identical to an uninterrupted run (tested for FedLabel, FedMatch, FedTriNet, FixMatch). |
| uncommitted | **Parallel runs**: each run is its own process (`python -m ssfl --resume <folder>`), at most N at once; `PAUSE` / `STOP` control files in the run folder; GUI "Parallel runs" (Server & run; Thesis grid tab); CLI `--grid ... --parallel N`; children get `OMP_NUM_THREADS = cores / N`; resume order: started runs first, then queued runs by `queued_at`. |

FedLabel (test1 version) details: h(·) = variance | entropy | max_prob on clamped logits; global model is the
pseudo-labeller only if strictly more confident (ties → local); strict β; strong view of kept samples only; step
skipped if < 2 kept; KL(s̄ ‖ s(w_U)) with λ_k = λ₀·min(1, h(s̄)/h(s*)); merge: params Δw_L + Δw_U, float buffers
averaged, int buffers from W_t; r_k = labeled set size + kept unlabeled samples. (`kl_direction`,
`local_warmup_steps`, `temperature` are retired; `handoff_2.md` §2 is outdated on ties and KL direction.)

## 2. Experiment results (single seed each)

| Run | Setting | Result |
|---|---|---|
| Laptop, no federation | CIFAR-10, ResNet-34, the 10k labeled images, 15 epochs | 71.4 % → data / model / evaluation work |
| Laptop, FedAvg | CIFAR-10 test1 settings, α = 100 | 7 % → 26 % in 10 rounds → federated loop works |
| Laptop, FedAvg | CIFAR-10 test1 settings, α = 0.1, 60 rounds | 10–23 %; global model predicts mostly one class, a different one each round (collapse) |
| Colab | CIFAR-10 test1, **α = 1.0**, 100 rounds | FedLabel 35.6 %, FedAvg 33.7 % |
| Colab | CIFAR-10 test1, α = 0.1, **GroupNorm**, 100 rounds | FedLabel 10 % throughout, FedAvg 10–21 % → BatchNorm is not the cause |
| Colab `organamnist_test1_fast_*` | OrganAMNIST test1 settings, 800 rounds, fast mode | **FedLabel 74.7 % best (r650), 67.6 % final; FedAvg 68.6 % best, 55.0 % final.** Paper: FedLabel 69.43 ± 1.58, FedAvg 41.90; test1: 74.47 % |
| Colab `cifar10_a0p1_b090_fedlabel` | CIFAR-10 test1 settings, α = 0.1, **β = 0.90**, 800 rounds, fast mode, eval every 50 | 10 % until ~r100 → 32.9 % (r600) → 19.7 % (r700) → 40.1 % (r750) → **47.0 % (r800)**. Paper 46.05 ± 1.09; test1 29.87 % (β 0.90); paper FedAvg 28.83 % |
| Colab `cifar10_a0p1_b090_fedavg` | same, FedAvg | **was still running** at handoff (16.6 % r50, 10.7 % r100); read the result from the Results tab / Drive |

Conclusions so far: the framework reproduces test1 and the paper. At α = 0.1 the first 100–300 rounds are a
collapse (one class predicted); 100 rounds is far too short. Accuracy swings ±10–20 points between evaluations,
so a single best / final number is noisy: report averages (last evaluations, 3 seeds).

Other findings
- **test1 vs framework speed** (1.2 vs 7 s/round, OrganAMNIST on the A100): test1 pickles ready tensors and only
  RandAugments the few kept samples; the framework augments every labeled image and makes weak + strong views of
  every unlabeled image. Fast mode: ~4.6 s/round (OrganAMNIST), ~11 s/round FedLabel and ~3 s/round FedAvg (CIFAR-10).
- **test1's score** = average over 10 random clients' own test splits, once; recomputed for our models this is
  about the same as the global test accuracy (not the reason for any gap).
- **μ = 7 (224 unlabeled images per step)** applies to every semi-supervised algorithm in the framework, not
  only FixMatch; it is the framework's choice (from FixMatch).
- **Paper appendix:** 80/5/15 split, 3 seeds, RandAugment(1, 10), β 0.3–0.5 best and 0.9 worst; Fig. 6: CIFAR-10
  τ = 30 best (~38 %), τ = 15 ~30 %; OrganAMNIST τ ≥ 15 all ~70 %, τ = 5 worst. Framework: CIFAR-10 preset τ = 15
  steps; MedMNIST presets 1 epoch ≈ 6 labeled steps per client on OrganAMNIST (1–10 at α 0.1), ≈ 15 on
  PathMNIST. **User decided to keep the presets as they are.**
- Parallel runs on the laptop: no gain (one run already keeps the RTX 3050 busy). The A100 gain is **not measured yet**.
- Evaluation cost: under ~1 s on the A100 (estimate), so evaluating every 1–5 rounds is cheap there; on the laptop
  one evaluation costs about one round.

## 3. Next steps (agreed, not done)

1. Read the FedAvg result of `cifar10_a0p1_b090_fedavg` (Results tab).
2. **CIFAR-10 α = 0.1, β = 0.5 FedLabel**: CIFAR-10 (test1) scenario, β 0.5, 800 rounds, Fast GPU mode,
   **Evaluate every 1 round** (user wants paper-like curves). Compare with β 0.90 (47.0 %) and the paper (46.05 %).
3. **β chosen by validation**: grid with `data.holdout: resplit`, `data.val_ratio: 0.05`, `data.test_ratio: 0.15`,
   β ∈ {0.3, 0.5, 0.7, 0.9}, α = 0.1, OrganAMNIST first then CIFAR-10, fast mode, own `output_dir`; pick the β with
   the best validation accuracy and report the test accuracy at that round. Give the user a time estimate first.
4. On the A100: measure parallel runs (1 vs 2–3) before relying on `--parallel`.
5. Final thesis numbers: 3 seeds, mean ± std.

Declined for now: "Paper figure" export (seed-averaged curves in the paper's style). Open small issue: the GUI ETA
assumes queued runs are as slow as the current one (overestimates for FedLabel → FedAvg queues).

## 4. Useful commands

```bash
python -m pytest -q tests                                   # 121 tests
python -m ssfl --list-resumable --runs-dir <runs dir>
python -m ssfl --resume <run folder> [--set server.rounds=1000]
python -m ssfl --grid configs/grids/thesis.yaml --parallel 3 --set output_dir=<runs dir>
python -m ssfl --clear-runs --runs-dir <runs dir> --status unfinished --dry-run
```
