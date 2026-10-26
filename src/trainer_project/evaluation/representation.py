"""Held-out representation evaluation on explicit labeled reference data."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def feature_cache_key(
    *,
    checkpoint_hash: str,
    manifests: Any,
    class_maps: Any,
    preprocessing: Any,
    feature_selection: Any,
    protocol: Any,
) -> str:
    """A cache cannot be reused after a model, dataset, mapping or protocol change."""
    if not checkpoint_hash:
        raise ValueError("Feature cache requires a concrete checkpoint hash")
    payload = {
        "checkpoint_hash": checkpoint_hash,
        "manifests": manifests,
        "class_maps": class_maps,
        "preprocessing": preprocessing,
        "feature_selection": feature_selection,
        "protocol": protocol,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@contextlib.contextmanager
def preserve_rng():
    python = random.getstate()
    numpy = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python)
            np.random.set_state(numpy)


def _array(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _protocol(
    reference_features: Any,
    reference_labels: Any,
    query_features: Any,
    reference_ids: Sequence[Any] | None,
    query_ids: Sequence[Any] | None,
    task: str,
    query_role: str,
    allow_overlap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    if task not in {"ssl", "byol", "mae", "mim"}:
        raise ValueError("kNN and linear probe are supported only for SSL tasks")
    if query_role != "val":
        raise ValueError("Representation quality metrics must use validation query rows")
    ref, labels, query = (
        _array(reference_features),
        _array(reference_labels).astype(np.int64),
        _array(query_features),
    )
    if ref.ndim != 2 or query.ndim != 2 or ref.shape[1] != query.shape[1]:
        raise ValueError("Reference and query features must have compatible [N,D] shapes")
    if labels.shape != (len(ref),):
        raise ValueError("Reference labels must have shape [N]")
    if reference_ids is None or query_ids is None:
        raise ValueError("Explicit reference_ids and query_ids are required for leakage checks")
    ref_ids, qry_ids = list(reference_ids), list(query_ids)
    if len(ref_ids) != len(ref) or len(qry_ids) != len(query):
        raise ValueError("Feature identity counts do not match rows")
    valid = labels >= 0
    ref, labels = ref[valid], labels[valid]
    ref_ids = [identity for identity, known in zip(ref_ids, valid) if known]
    if not len(ref):
        raise ValueError("Labeled reference bank is empty; validation labels are never used as a fallback")
    if not np.isfinite(ref).all() or not np.isfinite(query).all():
        raise ValueError("Features contain NaN or infinity")
    if not allow_overlap and set(ref_ids) & set(qry_ids):
        raise ValueError("Reference and validation query identities overlap")
    return ref, labels, query, ref_ids, qry_ids


class KnnEvaluator:
    def __init__(
        self,
        k: int = 20,
        temperature: float = 0.07,
        distance: str = "cosine",
        normalize: bool = True,
        voting: str = "temperature",
        protocol: str = "reference_query",
        num_classes: int | None = None,
        **_: Any,
    ):
        if k < 1 or temperature <= 0:
            raise ValueError("k and temperature must be positive")
        if protocol not in {"reference_query", "val_leave_one_out"}:
            raise ValueError("Unknown kNN protocol")
        if distance not in {"cosine", "euclidean"} or voting not in {"temperature", "uniform"}:
            raise ValueError("Unsupported distance or voting rule")
        self.k, self.temperature, self.distance, self.normalize = k, temperature, distance, normalize
        self.voting, self.protocol, self.num_classes = voting, protocol, num_classes

    def predict(
        self,
        reference_features: Any,
        reference_labels: Any,
        query_features: Any,
        *,
        reference_ids: Sequence[Any] | None = None,
        query_ids: Sequence[Any] | None = None,
        reference_groups: Sequence[Any] | None = None,
        query_groups: Sequence[Any] | None = None,
        query_role: str = "val",
        task: str = "ssl",
    ) -> torch.Tensor:
        ref, labels, query, ref_ids, qry_ids = _protocol(
            reference_features,
            reference_labels,
            query_features,
            reference_ids,
            query_ids,
            task,
            query_role,
            self.protocol == "val_leave_one_out",
        )
        if self.protocol == "val_leave_one_out" and set(qry_ids) != set(ref_ids):
            # A labeled subset can form the bank when validation has missing labels.
            if not set(ref_ids).issubset(set(qry_ids)):
                raise ValueError("Leave-one-out reference must come from the same validation set")
        if (reference_groups is None) != (query_groups is None):
            raise ValueError("Both reference_groups and query_groups must be specified")
        if reference_groups is not None:
            if len(reference_groups) != len(_array(reference_labels)) or len(query_groups) != len(query):
                raise ValueError("Group identity counts mismatch")
            valid_labels = _array(reference_labels).astype(np.int64) >= 0
            reference_groups = [group for group, valid in zip(reference_groups, valid_labels) if valid]
            if self.protocol == "reference_query" and set(reference_groups) & set(query_groups):
                raise ValueError("Reference and validation query groups overlap")
        if self.normalize or self.distance == "cosine":
            ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
            query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        similarities = (
            query @ ref.T
            if self.distance == "cosine"
            else -np.sum((query[:, None] - ref[None]) ** 2, axis=-1)
        )
        count = self.num_classes or int(labels.max() + 1)
        if labels.max() >= count:
            raise ValueError("Reference label outside declared classes")
        probabilities = np.zeros((len(query), count), dtype=np.float64)
        for index, identity in enumerate(qry_ids):
            allowed = np.ones(len(ref), dtype=bool)
            if self.protocol == "val_leave_one_out":
                allowed &= np.array([other != identity for other in ref_ids])
                if reference_groups is not None:
                    allowed &= np.array([group != query_groups[index] for group in reference_groups])
            indices = np.flatnonzero(allowed)
            if not len(indices):
                raise ValueError(f"No eligible neighbors for validation identity {identity}")
            indices = indices[np.argsort(-similarities[index, indices], kind="stable")[: self.k]]
            if self.voting == "uniform":
                weights = np.ones(len(indices))
            else:
                scores = similarities[index, indices] / self.temperature
                weights = np.exp(scores - scores.max())
            probabilities[index] = np.bincount(labels[indices], weights=weights, minlength=count)
            probabilities[index] /= probabilities[index].sum()
        return torch.tensor(probabilities, dtype=torch.float32)

    __call__ = predict


class LinearProbeEvaluator:
    def __init__(
        self,
        num_classes: int | None = None,
        epochs: int = 100,
        lr: float = 0.1,
        weight_decay: float = 0.0,
        seed: int = 0,
        normalize: bool = True,
        optimizer: str = "sgd",
        batch_size: int = 256,
        refit_each_snapshot: bool = True,
        **_: Any,
    ):
        if epochs < 1 or lr <= 0 or batch_size < 1:
            raise ValueError("Probe epochs, learning rate and batch size must be positive")
        if not refit_each_snapshot:
            raise ValueError("Held-out snapshot probe requires a freshly initialized head")
        if optimizer.lower() not in {"sgd", "adam", "adamw"}:
            raise ValueError("Unsupported probe optimizer")
        self.num_classes, self.epochs, self.lr, self.weight_decay = num_classes, epochs, lr, weight_decay
        self.seed, self.normalize, self.optimizer, self.batch_size = (
            seed,
            normalize,
            optimizer.lower(),
            batch_size,
        )
        self.head_state: dict[str, torch.Tensor] | None = None

    def predict(
        self,
        reference_features: Any,
        reference_labels: Any,
        query_features: Any,
        *,
        reference_ids: Sequence[Any] | None = None,
        query_ids: Sequence[Any] | None = None,
        reference_groups: Sequence[Any] | None = None,
        query_groups: Sequence[Any] | None = None,
        query_role: str = "val",
        task: str = "ssl",
    ) -> torch.Tensor:
        ref, labels, query, _, _ = _protocol(
            reference_features,
            reference_labels,
            query_features,
            reference_ids,
            query_ids,
            task,
            query_role,
            False,
        )
        if (reference_groups is None) != (query_groups is None):
            raise ValueError("Both reference and query group IDs must be given")
        if reference_groups is not None and set(reference_groups) & set(query_groups):
            raise ValueError("Reference and validation query groups overlap")
        count = self.num_classes or int(labels.max() + 1)
        if labels.max() >= count:
            raise ValueError("Reference label outside declared classes")
        # The head is independent of the encoder, global optimizer and all training RNG.
        with preserve_rng(), torch.inference_mode(False), torch.enable_grad():
            torch.manual_seed(self.seed)
            x = torch.tensor(ref, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            q = torch.tensor(query, dtype=torch.float32)
            if self.normalize:
                x, q = F.normalize(x, dim=1), F.normalize(q, dim=1)
            head = torch.nn.Linear(x.shape[1], count)
            cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[
                self.optimizer
            ]
            optimizer = cls(head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            generator = torch.Generator().manual_seed(self.seed)
            for _ in range(self.epochs):
                order = torch.randperm(len(x), generator=generator)
                for indices in order.split(self.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head(x[indices]), y[indices])
                    loss.backward()
                    optimizer.step()
            self.head_state = copy.deepcopy(head.state_dict())
            with torch.no_grad():
                probabilities = head(q).softmax(dim=-1).detach()
        return probabilities

    __call__ = predict
