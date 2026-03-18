"""
SPAM (Spike-Aware Adam with Momentum Reset) Optimizer.

Adapted from: https://github.com/TianjinYellow/SPAM-Optimizer
Paper: "SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training"

Key features over standard Adam:
  1. Momentum reset every DeltaT steps — prevents stale momentum accumulation.
  2. Spike-aware gradient clipping — clips gradient spikes that exceed
     `threshold * EMA(grad^2)`, replacing them with a safe magnitude.
  3. Cosine warmup after each momentum reset — smoothly ramps the update
     scale back to 1.0 over `warmup_steps` steps.
"""

import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Cosine decay helper (used for post-reset warmup)
# ---------------------------------------------------------------------------

class CosineDecay:
    """Cosine schedule that decays `death_rate` from its initial value to `eta_min`."""

    def __init__(self, death_rate: float, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]),
            lr=death_rate,
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max + 1, eta_min, last_epoch,
        )
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        self.cosine_stepper.step(current_step)

    def get_dr(self, current_step: int) -> float:
        if current_step >= self.T_max:
            return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# SPAM optimizer
# ---------------------------------------------------------------------------

class SPAM(Optimizer):
    """
    SPAM — Spike-Aware Adam with Momentum Reset.

    Drop-in replacement for AdamW with two extra mechanisms that stabilise
    training when gradient spikes occur:

    * **Momentum reset** every ``DeltaT`` optimiser steps — first / second
      moment buffers are zeroed and a cosine warm-up ramps the effective
      update back to full strength over ``warmup_steps`` steps.
    * **Spike-aware clipping** — if ``grad_i^2 > threshold * v_i`` the
      gradient element is clamped to ``sign(grad_i) * sqrt(threshold * v_i)``,
      preventing a single spike from corrupting the momentum estimates.

    Args:
        params: Iterable of parameters or param-group dicts.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for running averages (default: (0.9, 0.999)).
        eps: Denominator term for numerical stability (default: 1e-6).
        weight_decay: Decoupled weight decay (default: 0.0).
        correct_bias: Apply Adam bias correction (default: True).
        warmup_steps: Cosine warm-up length after each momentum reset
            (default: 150).
        threshold: Spike detection multiplier on EMA(grad^2). Set to 0
            to disable spike clipping (default: 5000).
        DeltaT: Period (in steps) between momentum resets. Set to 0 to
            disable resets (default: 500).
        grad_accu_steps: Number of initial steps after a reset during
            which spike clipping is disabled so the EMA can warm up
            (default: 20).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        warmup_steps: int = 150,
        threshold: int = 5000,
        DeltaT: int = 500,
        grad_accu_steps: int = 20,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

        self.state["total_step"] = 0
        self.state["current_step"] = warmup_steps + 1  # start past warmup
        self.DeltaT = DeltaT
        self.warmup_steps = warmup_steps
        self.warmup = CosineDecay(0.99, warmup_steps)
        self.thres = threshold
        self.grad_accu_steps = grad_accu_steps

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scale_factor = 1.0 - self.warmup.get_dr(self.state["current_step"])

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SPAM does not support sparse gradients.")

                state = self.state[p]

                # Lazy state init
                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                # Momentum reset
                if self.DeltaT > 0 and (self.state["total_step"] + 1) % self.DeltaT == 0:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Spike-aware gradient clipping
                if self.thres > 0:
                    step_global = self.state["total_step"] + 1
                    if step_global >= self.grad_accu_steps:
                        mask = (grad ** 2) > (self.thres * exp_avg_sq)
                        should_clip = True
                        if self.DeltaT > 0:
                            should_clip = (step_global % self.DeltaT) >= self.grad_accu_steps
                        if should_clip:
                            safe_val = torch.sqrt(exp_avg_sq[mask] * self.thres)
                            grad[mask] = grad[mask].sign() * safe_val

                # EMA updates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bc1 = 1.0 - beta1 ** state["step"]
                    bc2 = 1.0 - beta2 ** state["step"]
                    step_size *= math.sqrt(bc2) / bc1

                norm_grad = exp_avg / denom
                p.add_(norm_grad, alpha=-step_size * scale_factor)

                # Decoupled weight decay
                if group["weight_decay"] > 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

        # Bookkeeping
        self.state["total_step"] += 1
        self.state["current_step"] += 1

        # Schedule next momentum reset
        if self.DeltaT > 0 and (self.state["total_step"] + 1) % self.DeltaT == 0:
            self.state["current_step"] = 0
            self.warmup = CosineDecay(0.99, self.warmup_steps)

        return loss
