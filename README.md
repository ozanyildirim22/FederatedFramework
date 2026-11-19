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
│   └── algorithms.py    #   Pseudo-Labeling Engines: supervised, pseudolabel, fixmatch, fedlabel
├── losses.py            #   Composite Loss Calculator (CE + masked CE + KL consistency)
├── client.py            #   Client: load W_t -> local train -> return delta_w_k
├── server/              # ── Tier 3: Server-Side Aggregation Module
│   ├── server.py        #   client selection (fraction C), broadcast, aggregation, evaluation
│   └── aggregators.py   #   fedavg, fedavgm, fedadam, fedmedian
├── models.py            # simple_cnn, resnet18 (BatchNorm or GroupNorm)
└── experiment.py        # wires everything + logging (metrics.csv, summary.json, model)
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

## Graphical interface

```bash
python -m ssfl --gui          # opens http://127.0.0.1:7860
```

* **Experiment tab** — start from a preset, change any setting (dropdowns list every
  registered dataset / strategy / aggregator, including your own plug-ins), click
  **Preview data partition** to see the non-IID class split per client, then
  **Run** to watch accuracy, loss and pseudo-label statistics update every round.
  **Stop** ends the run after the current round; **Save as preset** writes `configs/<name>.yaml`.
* **Results tab** — tick finished runs to overlay their curves and get a summary table.
* In Colab: `from ssfl.gui import launch; launch()` shows the GUI inside the notebook.

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

Or from Python:

```python
from ssfl import Experiment
history = Experiment.from_yaml("configs/cifar10_fedlabel.yaml", ["server.rounds=50"]).run()
```

Every run writes `runs/<name>_<timestamp>/` with `config.yaml`, `metrics.csv`
(per-round loss, mask rate, pseudo-label accuracy, test accuracy), `summary.json`
and `global_model.pt`.

## Built-in algorithms

| Client strategy (`strategy.name`) | Description |
|---|---|
| `supervised` | FedAvg on labeled data only (lower bound). With `data.labeled_ratio=1.0` → fully-supervised upper bound. |
| `pseudolabel` | Hard pseudo-labels above threshold `tau` (Lee, 2013). |
| `fixmatch` | Weak-view pseudo-label enforced on strong view (FedAvg + FixMatch). |
| `fedlabel` | Per-sample choice between local and frozen global model, by confidence score (`variance` / `max_prob` / `entropy`), threshold `tau`, + global-local KL consistency (`lambda_kl`). `selection: local|global` gives the ablations. |
| any + `strategy.prox_mu > 0` | Adds the FedProx proximal term to any strategy. |

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

Then use `strategy.name: my_algo` / `server.aggregator: my_rule` in YAML. The same works
for `DATASETS`, `PARTITIONERS` and `MODELS`. See `examples/custom_algorithm.py`.

## Performance notes

* One local step processes `batch_size × (1 + 2·unlabeled_batch_ratio)` images
  (labeled + weak + strong views). With the presets that is 480 images; ResNet-18
  needs ~1.6 s per step on an RTX 3050 Laptop GPU (≈4 min per cross-silo round).
* Faster iteration while developing: `--set model.name=simple_cnn` or
  `data.unlabeled_batch_ratio=3`, fewer `server.rounds`.
* `data.num_workers > 0` moves augmentation off the main process.
* `strategy.grad_clip` (default 5.0) and zero-initialised residual branches keep
  early training stable; `strategy.amp=true` enables mixed precision.

## Tests

```bash
pytest -q
```
