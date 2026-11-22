# SSFL — Semi-Supervised Federated Learning framework

A modular, plug-and-play PyTorch framework that follows the three-tier architecture
of the project (Data Manager → Client Side → Server Side). Every component is
selected **by name from a YAML file**, and new algorithms are added with a single
decorator — no framework code has to change.

```
ssfl/
├── registry.py          # name -> class registries (the plug-and-play mechanism)
├── config.py            # typed dataclass config, YAML loading, CLI overrides
├── data/                # ── Tier 1: Data Management Module
│   ├── datasets.py      #   CIFAR-10, OrganAMNIST, PathMNIST, synthetic
│   ├── partition.py     #   Partitioner: IID / Dirichlet(alpha) non-IID + labeled/unlabeled split
│   ├── augment.py       #   Augmentation Factory: weak (flip+crop) / strong (RandAugment+Cutout)
│   └── manager.py       #   DataManager: builds every client's labeled & unlabeled loaders
├── strategies/          # ── Tier 2: Client-Side Execution Module
│   ├── base.py          #   Training Protocol Interface (epochs, backprop, FedProx term)
│   ├── algorithms.py    #   Pseudo-Labeling Engines: supervised, pseudolabel, fixmatch, fedlabel
│   ├── baselines.py     #   fedavg, fedprox (labeled-only FL baselines)
│   ├── fedmatch.py      #   FedMatch (inter-client consistency + sigma/psi decomposition)
│   └── fedtrinet.py     #   FedTriNet (local / global / combined models + dynamic threshold)
├── losses.py            #   Composite Loss Calculator (CE + masked CE + KL consistency)
├── client.py            #   Client: load W_t -> local train -> return delta_w_k
├── server/              # ── Tier 3: Server-Side Aggregation Module
│   ├── server.py        #   client selection (fraction C), broadcast, aggregation, evaluation
│   └── aggregators.py   #   fedavg, fedavgm, fedadam, fedmedian
├── models.py            # simple_cnn, resnet18, resnet34 (BatchNorm or GroupNorm)
├── experiment.py        # wires everything + logging (metrics.csv, summary.json, model)
└── grid.py              # runs a whole grid of experiments from one YAML (resumable)
```

## Google Colab

Open `colab/SSFL_Colab.ipynb` in Colab
([direct link](https://colab.research.google.com/github/ozanyildirim22/FederatedFramework/blob/main/colab/SSFL_Colab.ipynb)),
select a GPU runtime and run the cells — it clones this repository automatically.

## Install

```bash
git clone https://github.com/ozanyildirim22/FederatedFramework.git
cd FederatedFramework
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .[dev,gui]
```

## Graphical interface — SSFL Studio

```bash
python -m ssfl --gui          # opens http://127.0.0.1:7860
```

* **🧪 Experiment** — click a **scenario** card (Quick demo, OrganAMNIST, PathMNIST, CIFAR-10),
  tick one or **several algorithms** (they run one after another on the same data split),
  set the labeled ratio / rounds / non-IID α, and press **🚀 Launch**. Each algorithm's
  parameters appear as a form with explanations; nothing has to be typed as JSON.
  The dashboard shows live KPI cards (test accuracy, best accuracy, pseudo-label mask
  and accuracy, loss, ETA) and charts; **🧩 Preview split** shows how the classes are
  spread across clients. Refreshing the page re-attaches to a running job.
* **🗺️ Thesis grid** — run `configs/grids/thesis.yaml` (or the seconds-long `demo.yaml`)
  with a live status table; finished runs are skipped.
* **🏆 Results** — leaderboard, curves of any metric, and the **thesis table**
  (best accuracy per algorithm × dataset × label ratio) with CSV and LaTeX download.
* **📘 Guide** — what every algorithm does and how to use the studio.
* In Colab, run the notebook's GUI cell:
  `!python -m ssfl --gui --share --data-dir "{DATA_DIR}" --runs-dir "{RUNS_DIR}"`.
  It starts the GUI as a fresh process (always the latest cloned code) and prints a public
  `gradio.live` link. The **build** chip in the header shows the git commit the GUI runs.

## Run

```bash
python -m ssfl --list                                   # show every registered component
python -m ssfl -c configs/quick_test.yaml               # 10-second sanity check, no download
python -m ssfl -c configs/cifar10_fedlabel.yaml         # cross-device, 100 clients, C = 0.1
python -m ssfl -c configs/organamnist_fedlabel.yaml     # cross-silo, 10 clients, C = 1.0
python -m ssfl -c configs/pathmnist_fedlabel.yaml
```

Change anything from the command line with `--set` (dotted keys):

```bash
python -m ssfl -c configs/cifar10_fedlabel.yaml --set data.labeled_ratio=0.5 strategy.name=fixmatch
python -m ssfl -c configs/organamnist_fedlabel.yaml --set data.labeled_ratio=0.2 data.alpha=0.1
python -m ssfl -c configs/cifar10_fedlabel.yaml --set strategy.params.selection=global   # ablation
python -m ssfl -c configs/cifar10_fedlabel.yaml --set strategy.prox_mu=0.01              # + FedProx
```

### Full thesis grid

`configs/grids/thesis.yaml` lists every run of the thesis: the 3 datasets × 2 label ratios ×
α ∈ {0.1, 1.0} × the FedLabel paper's comparison (FedAvg, FedProx, FedAvg+FixMatch, FedAvg+UDA,
FedProx+FixMatch, FedProx+UDA, FedTriNet, FedMatch, FedLabel), FedAvg / FedProx with 100% labels
(upper bounds) and FedLabel ablations on λ₀, τ (5 / 20 / 50 steps), β, model selection and h(·) (131 runs).

```bash
python -m ssfl --grid configs/grids/thesis.yaml --dry-run              # list the runs
python -m ssfl --grid configs/grids/thesis.yaml --only organamnist     # run a subset
python -m ssfl --grid configs/grids/thesis.yaml --set output_dir=/content/drive/MyDrive/ssfl/runs
```

A run whose `runs/<name>_*/summary.json` already exists is skipped, so an interrupted grid
continues where it stopped (`--no-resume` runs everything again).

Or from Python:

```python
from ssfl import Experiment
history = Experiment.from_yaml("configs/cifar10_fedlabel.yaml", ["server.rounds=50"]).run()
```

Every run writes `runs/<name>_<timestamp>/` with:

* `config.yaml`: **every** setting, including all algorithm and aggregator parameters
  (defaults written out), so this file alone reproduces the run;
* `metrics.csv`: per round loss, mask rate, pseudo-label accuracy, test accuracy and time;
* `summary.json`: status (`running` / `finished` / `stopped` / `error`), start time, setup /
  training / total time, seconds per round, best accuracy and its round, final metrics, DataLoader
  workers, precision, and the environment (GPU, PyTorch, Python, code commit). It is updated after
  every round, so stopped or interrupted runs keep their timing too;
* `global_model.pt`: the final global model.

The GUI's **🏆 Results → 🔍 Run details** shows all of this for one run, with its curves, download
buttons and **Load these settings into Experiment**.

## Built-in algorithms

| Client strategy (`strategy.name`) | Description |
|---|---|
| `fedavg` | FedAvg (McMahan et al., 2017) on each client's labeled data. With `data.labeled_ratio=1.0` → fully supervised upper bound. (`supervised` is the same algorithm.) |
| `fedprox` | FedProx (Li et al., 2020): FedAvg + proximal term `mu/2·‖w − W_t‖²` (`mu`, default 0.01). |
| `fedmatch` | FedMatch (Jeong et al., ICLR 2021): weights `sigma + psi`; `sigma` learns from labeled data, `psi` from unlabeled data. Pseudo-labels come from a vote of the local model and `num_helpers` helper models chosen by the server (nearest neighbours of a model embedding), enforced on the strong view, + inter-client KL consistency and L1/L2 terms on `psi`. Params: `tau`, `lambda_s`, `lambda_iccs`, `lambda_l1`, `lambda_l2`, `num_helpers`, `helper_every`, `decompose`. |
| `fedtrinet` | FedTriNet (Che et al., 2021): `pretrain_rounds` of FedAvg on labeled data, then three players — the client's previous local model, the global model and a combined model (global shallow layers + local deep layers) — average their softmax outputs to give pseudo-labels. These are kept above a dynamic threshold `theta(t)`: clients upload `max p_G`, the server averages it, and the threshold decays with the paper's schedule (`alpha`=0.93). Params: `pretrain_rounds`, `alpha`, `lambda_u`, `finetune_steps`, `global_layers`, `unsup_view`. |
| `fedlabel` | FedLabel (Cho et al., ICCV 2023) as in the ADD (Section 3.1, Algorithm 2). **Step 1:** τ epochs (`sup_epochs`) of supervised training from W_t give the local model w_L. **Step 2:** a new model w_U, again from W_t, trains τ′ epochs (`unsup_epochs`) on unlabeled data: per sample, the global and local model are scored with h(·) = variance of the logits (`confidence`); the more confident one gives the hard pseudo-label if its top probability ≥ β (`beta`), trained on the strong view. **Step 3:** where the other model predicts the same class, a KL term (`kl_direction`) weighted by λ_k = λ₀·h(s̄)/h(s*) (`lambda0`). The client uploads Δw_L + Δw_U, and the server weights it by r_k = samples used that round (paper's Alg. 1; `weight_by_used`). τ / τ′ can also be given in SGD steps (`sup_steps`, `unsup_steps`), as in the paper. `selection: local\|global` and `confidence` give the ablations. |
| `pseudolabel` | Hard pseudo-labels above threshold `tau` (Lee, 2013). |
| `fixmatch` | Weak-view pseudo-label enforced on strong view (FedAvg+FixMatch; with `strategy.prox_mu` FedProx+FixMatch). |
| `uda` | UDA (Xie et al., 2020): sharpened weak-view prediction (T = 0.4, confidence mask 0.8) is the KL target for the strong view (FedAvg+UDA; with `strategy.prox_mu` FedProx+UDA). |
| any + `strategy.prox_mu > 0` | Adds the FedProx proximal term to any strategy (e.g. FixMatch + FedProx). |

Each algorithm's docstring (`ssfl/strategies/*.py`) gives the equations and lists
every implementation choice the paper leaves open.

| Dataset (`data.dataset`) | Classes | Train / test | Setting in the presets |
|---|---|---|---|
| `cifar10` | 10 | 50,000 / 10,000 (32×32 RGB) | cross-device: 100 clients, C = 0.1, 20% / 50% labels, **ResNet-34** |
| `organamnist` | 11 | 34,561 / 17,778 (28×28 gray) | cross-silo: 10 clients, C = 1.0, 5% / 20% labels, ResNet-18 |
| `pathmnist` | 9 | 89,996 / 7,180 (28×28 RGB, val split unused) | cross-silo: 10 clients, C = 1.0, 5% / 20% labels, ResNet-18 |

All presets follow the ADD: Dirichlet α = 1.0 (the grid also runs 0.1), 1,000 rounds, evaluation every
5 rounds, uniform random labeled/unlabeled split per client, weak view = flip + crop, strong view =
weak + RandAugment (N = 1, M = 10). `data.cutout=true` adds Cutout, `data.split=stratified` keeps class
proportions. Horizontal flips are **off for OrganAMNIST**: its classes include left and right
femur, kidney and lung, and a flipped left kidney would keep the label "left" (`data.hflip=true|false`
overrides; the run summary records what was used). Every evaluation reports Top-1 accuracy and macro-averaged precision, recall and F1
(`test_precision`, `test_recall`, `test_f1`); the Results tab's thesis table also gives the relative
ratio method / FedAvg (ADD Eq. 7).

All three download automatically into `data.root` the first time they are used.

| Aggregator (`server.aggregator`) | Description |
|---|---|
| `fedavg` | Dataset-size weighted average of `delta_w_k` (`weight_by: num_samples|num_labeled|uniform`). |
| `fedavgm` | FedAvg + server momentum (`beta`). |
| `fedadam` | Adaptive server optimizer (`server_lr`, `beta1`, `beta2`, `tau`). |
| `fedmedian` | Coordinate-wise median (robust aggregation). |

## Communication round (what `Experiment.run_round` does)

1. **Client selection** — server samples `round(C · N)` clients uniformly at random.
2. **Broadcast** — current global weights `W_t` are sent to the selected clients.
3. **Local execution** — each client loads `W_t`, trains on its private labeled +
   unlabeled batches with the chosen strategy, and computes `delta_w_k = w_k − W_t`.
4. **Upload** — only `delta_w_k` (and integer buffers) are returned; never data or labels.
5. **Aggregation** — the aggregator combines the updates into `W_{t+1}`.

## Adding your own components

```python
from ssfl import STRATEGIES, AGGREGATORS, Strategy, Aggregator

@STRATEGIES.register("my_algo")
class MyAlgo(Strategy):
    needs_global_model = True        # get a frozen copy of W_t as self.global_model
    def compute_loss(self, model, labeled, unlabeled):
        x, y = labeled                       # weak-augmented labeled batch
        u_weak, u_strong, _ = unlabeled      # 3rd item is the hidden true label (diagnostics only)
        ...
        return self.loss(sup=..., unsup=...)  # CompositeLoss -> (total, logs)

@AGGREGATORS.register("my_rule")
class MyRule(Aggregator):
    def combine(self, key, deltas, weights):   # deltas: (num_clients, *param_shape)
        ...
```

Algorithms with a server-side part (like FedMatch's helper selection or FedTriNet's
threshold) override the server hooks of `Strategy`: `setup(experiment)`,
`prepare_model(model)` (wrap the architecture once), `on_server_round_start(server, selected, rnd)`
and `on_server_round_end(server, updates, global_state, rnd)`. The client-side hooks are
`on_round_start(model, client)` / `on_round_end(model, client)`.

Then use `strategy.name: my_algo` / `server.aggregator: my_rule` in YAML. The same works
for `DATASETS`, `PARTITIONERS` and `MODELS`. See `examples/custom_algorithm.py`.

## Performance notes

* One local step processes `batch_size × (1 + 2·unlabeled_batch_ratio)` images
  (labeled + weak + strong views). With the presets that is 480 images; ResNet-18
  needs ~1.6 s per step on an RTX 3050 Laptop GPU (≈4 min per cross-silo round).
* Faster iteration while developing: `--set model.name=simple_cnn` or
  `data.unlabeled_batch_ratio=3`, fewer `server.rounds`.
* `data.num_workers` (default `-1` = auto) runs augmentation in background processes.
  Auto uses 0 on Windows (starting a worker costs seconds there) and for small rounds such
  as the quick demo, and up to 4 on Colab / Linux with real datasets. Workers start once per
  client per round, so for tiny rounds they cost more than they save.
* `strategy.grad_clip` (default 5.0) and zero-initialised residual branches keep
  early training stable.

### Large GPUs (A100 / H100 / RTX PRO class)

On a large GPU the training step itself is fast, so the CPU becomes the bottleneck. It decodes
and augments ~480 images per step, about 0.35-0.5 s per step for one process. Two options move
that work to the GPU (**⚡ Fast GPU mode** in the GUI turns on both):

| Setting | What it does |
|---|---|
| `data.augment_device=gpu` | Keeps the dataset on the GPU (OrganAMNIST 27 MB, CIFAR-10 150 MB) and augments whole batches there: flip + reflect crop, RandAugment with torchvision's own ops and magnitudes, Cutout. About 40 ms per step instead of about 350 ms of CPU work on an RTX 3050 Laptop. Statistically the same views as the CPU pipeline, but not bit-identical: keep `cpu` to reproduce earlier runs. |
| `strategy.precision=bf16` | bfloat16 autocast. It has the float32 range, so no loss scaling is needed (fp16, the old `amp`, made training diverge). Needs hardware bf16 (Ampere or newer); elsewhere it falls back to fp32. About 20% faster on an RTX 3050 Laptop. |
| `strategy.channels_last=true` | NHWC memory layout for convolutions (optional). |

On small GPUs (T4, laptops) the GPU itself is the bottleneck. CPU augmentation of the next batch
already overlaps with training, so `augment_device=gpu` gains little there and can even be slower.

```bash
python -m ssfl -c configs/organamnist_fedlabel.yaml --set data.augment_device=gpu strategy.precision=bf16
```

## Tests

```bash
pytest -q
```
