"""Plugging a new algorithm into SSFL without touching the framework.

Run:  python examples/custom_algorithm.py
"""
import torch
import torch.nn.functional as F

from ssfl import AGGREGATORS, STRATEGIES, Aggregator, Experiment, Strategy, kl_consistency


# 1) A new pseudo-labeling strategy: soft pseudo-labels from the frozen global model
#    (a "global teacher" distillation baseline). Only compute_loss is needed.
@STRATEGIES.register("global_teacher")
class GlobalTeacher(Strategy):
    needs_global_model = True          # framework gives us self.global_model (frozen W_t)

    def __init__(self, cfg, num_classes, lambda_u: float = 1.0, temperature: float = 2.0, **params):
        params.setdefault("loss_weights", {"sup": 1.0, "distill": lambda_u})
        super().__init__(cfg, num_classes, **params)
        self.t = temperature

    def compute_loss(self, model, labeled, unlabeled):
        x, y = labeled
        u_weak, u_strong, _ = unlabeled          # 3rd item = true label, diagnostics only
        with torch.no_grad():
            teacher = self.global_model(u_weak)
        return self.loss(sup=F.cross_entropy(model(x), y),
                         distill=kl_consistency(model(u_strong), teacher, self.t))


# 2) A new aggregation rule: trimmed mean.
@AGGREGATORS.register("trimmed_mean")
class TrimmedMean(Aggregator):
    def __init__(self, trim: float = 0.1, **kw):
        super().__init__(**kw)
        self.trim = trim

    def combine(self, key, deltas, weights):
        k = int(self.trim * deltas.shape[0])
        s = deltas.sort(dim=0).values
        return s[k:deltas.shape[0] - k].mean(0) if deltas.shape[0] > 2 * k else s.mean(0)


if __name__ == "__main__":
    Experiment.from_yaml("configs/quick_test.yaml",
                         ["strategy.name=global_teacher", "strategy.params={}",
                          "server.aggregator=trimmed_mean"]).run()
