"""Behaviour cloning in plain numpy.

A two-hidden-layer MLP with a tanh output, observation normalisation and Adam,
fitted by mean squared error to the expert's actions. The hyperparameters follow
the original repository (256 units, 60 epochs, batch 256, learning rate 1e-3).

It is numpy rather than torch because the trained policy has to run inside the
OmniSim controller, whose interpreter carries numpy and not torch, and because
one implementation used for both simulators removes the learner as a variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NORM_CLIP = 10.0


@dataclass
class MLPPolicy:
    mean: np.ndarray
    std: np.ndarray
    weights: list = field(default_factory=list)
    biases: list = field(default_factory=list)

    @classmethod
    def init(cls, obs_dim: int, act_dim: int, hidden: int, rng: np.random.Generator):
        sizes = [obs_dim, hidden, hidden, act_dim]
        weights, biases = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(rng.uniform(-bound, bound, size=(fan_in, fan_out)))
            biases.append(np.zeros(fan_out))
        return cls(np.zeros(obs_dim), np.ones(obs_dim), weights, biases)

    def _forward(self, obs: np.ndarray):
        # Clipped as in the original repository: an input the demonstrations
        # barely vary (the box's sideways velocity) otherwise turns a small
        # disturbance into a huge activation and saturates every action.
        x = np.clip((obs - self.mean) / self.std, -NORM_CLIP, NORM_CLIP)
        acts = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(x @ w + b, 0.0)
            acts.append(x)
        out = np.tanh(x @ self.weights[-1] + self.biases[-1])
        return out, acts

    def act(self, obs: np.ndarray) -> np.ndarray:
        out, _ = self._forward(np.asarray(obs, dtype=np.float64)[None, :])
        return out[0].astype(np.float32)

    def save(self, path: str) -> None:
        arrays = {"mean": self.mean, "std": self.std}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "MLPPolicy":
        data = np.load(path, allow_pickle=False)
        n = sum(1 for k in data.files if k.startswith("w"))
        return cls(data["mean"], data["std"],
                   [data[f"w{i}"] for i in range(n)], [data[f"b{i}"] for i in range(n)])


def fit(obs: np.ndarray, act: np.ndarray, rng: np.random.Generator, hidden: int = 256,
        epochs: int = 60, batch_size: int = 256, lr: float = 1e-3,
        val_fraction: float = 0.1):
    """Fit a policy to (obs, act) pairs. Returns the policy and the loss curves."""
    obs = np.asarray(obs, dtype=np.float64)
    act = np.asarray(act, dtype=np.float64)
    n = len(obs)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    policy = MLPPolicy.init(obs.shape[1], act.shape[1], hidden, rng)
    policy.mean = obs[train_idx].mean(axis=0)
    policy.std = np.sqrt(obs[train_idx].var(axis=0) + 1e-8)

    params = policy.weights + policy.biases
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    train_curve, val_curve = [], []
    layers = len(policy.weights)

    for _ in range(epochs):
        order = train_idx[rng.permutation(len(train_idx))]
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            out, acts = policy._forward(obs[idx])
            err = out - act[idx]
            total += float(np.mean(err ** 2))
            batches += 1
            # Backpropagate the mean squared error through tanh and the ReLUs.
            grad = 2.0 * err / err.size * (1.0 - out ** 2)
            grads_w, grads_b = [None] * layers, [None] * layers
            for layer in range(layers - 1, -1, -1):
                grads_w[layer] = acts[layer].T @ grad
                grads_b[layer] = grad.sum(axis=0)
                if layer:
                    grad = (grad @ policy.weights[layer].T) * (acts[layer] > 0.0)
            step += 1
            for p, g, mi, vi in zip(params, grads_w + grads_b, m, v):
                mi *= beta1
                mi += (1.0 - beta1) * g
                vi *= beta2
                vi += (1.0 - beta2) * g * g
                m_hat = mi / (1.0 - beta1 ** step)
                v_hat = vi / (1.0 - beta2 ** step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        train_curve.append(total / max(1, batches))
        out, _ = policy._forward(obs[val_idx])
        val_curve.append(float(np.mean((out - act[val_idx]) ** 2)))
    return policy, train_curve, val_curve
