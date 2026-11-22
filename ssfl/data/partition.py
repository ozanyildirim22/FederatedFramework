"""Partitioners: split a global dataset into client subsets.

A partitioner is any callable ``(labels, num_clients, rng, **kw) -> list[np.ndarray]``
returning one index array per client. Register new ones with
``@PARTITIONERS.register("name")``.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..registry import PARTITIONERS


@PARTITIONERS.register("iid")
def iid_partition(labels: np.ndarray, num_clients: int, rng: np.random.Generator, **_) -> List[np.ndarray]:
    idx = rng.permutation(len(labels))
    return [np.sort(part) for part in np.array_split(idx, num_clients)]


@PARTITIONERS.register("dirichlet")
def dirichlet_partition(labels: np.ndarray, num_clients: int, rng: np.random.Generator,
                        alpha: float = 0.5, min_samples_per_client: int = 10,
                        max_retries: int = 100, **_) -> List[np.ndarray]:
    """Label-skew non-IID split.

    For every class ``c`` a vector ``p_c ~ Dir(alpha * 1_K)`` is drawn and the
    samples of class ``c`` are distributed across the ``K`` clients in those
    proportions. Small ``alpha`` (e.g. 0.1) -> highly skewed; large ``alpha``
    (e.g. 100) -> close to IID. Resamples until every client has at least
    ``min_samples_per_client`` samples.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n = len(labels)
    min_req = min(min_samples_per_client, n // num_clients)
    for _ in range(max_retries):
        buckets: List[list] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = rng.permutation(np.where(labels == c)[0])
            p = rng.dirichlet(np.full(num_clients, alpha))
            # Balance trick (Wang et al., 2020): stop giving to clients that are already full.
            sizes = np.array([len(b) for b in buckets])
            p = p * (sizes < n / num_clients)
            p = p / p.sum() if p.sum() > 0 else np.full(num_clients, 1.0 / num_clients)
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for b, part in zip(buckets, np.split(idx_c, cuts)):
                b.extend(part.tolist())
        if min(len(b) for b in buckets) >= min_req:
            return [np.sort(np.array(b, dtype=np.int64)) for b in buckets]
    raise RuntimeError(
        f"Could not build a Dirichlet partition with >= {min_req} samples per client "
        f"after {max_retries} tries; increase alpha or reduce num_clients.")


def split_labeled_unlabeled(indices: np.ndarray, labels: np.ndarray, labeled_ratio: float,
                            rng: np.random.Generator, mode: str = "stratified"):
    """Per-client split into labeled / unlabeled index sets.

    * ``random``     - a uniform random ``labeled_ratio`` of the client's samples (the ADD);
                       a class can, by chance, end up with no labels on a client.
    * ``stratified`` - the same fraction of every class.
    """
    if mode == "random":
        perm = rng.permutation(indices)
        k = int(round(labeled_ratio * len(perm)))
        if k == 0 and len(perm) > 0 and labeled_ratio > 0:            # guarantee >= 1 label
            k = 1
        return np.sort(perm[:k]), np.sort(perm[k:])
    if mode != "stratified":
        raise ValueError(f"Unknown split '{mode}' (random | stratified)")
    labeled, unlabeled = [], []
    for c in np.unique(labels[indices]):
        idx_c = rng.permutation(indices[labels[indices] == c])
        k = int(round(labeled_ratio * len(idx_c)))
        labeled.append(idx_c[:k])
        unlabeled.append(idx_c[k:])
    labeled = np.concatenate(labeled) if labeled else np.array([], dtype=np.int64)
    unlabeled = np.concatenate(unlabeled) if unlabeled else np.array([], dtype=np.int64)
    if len(labeled) == 0 and len(indices) > 0 and labeled_ratio > 0:  # guarantee >= 1 label
        labeled, unlabeled = unlabeled[:1], unlabeled[1:]
    return np.sort(labeled), np.sort(unlabeled)
