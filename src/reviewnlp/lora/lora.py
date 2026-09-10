"""LoRA implemented from scratch in pure PyTorch - research-to-code demo.

Reference: "LoRA: Low-Rank Adaptation of Large Language Models",
Hu et al., 2021 (arXiv:2106.09685).

The core idea of the paper in one equation:

    h = W0 x + DeltaW x = W0 x + (alpha / r) * B A x

where ``W0`` is the frozen pretrained weight matrix (d x k), and the update
``DeltaW`` is factorized into two trainable matrices ``A`` (r x k) and
``B`` (d x r) with rank ``r << min(d, k)``.

Implementation choices, matched to the official code:

- ``A`` is initialised with ``kaiming_uniform_(a=sqrt(5))`` and ``B`` with
  zeros, so ``DeltaW = 0`` at step 0 and fine-tuning starts exactly at the
  pretrained function (the paper describes a Gaussian init for ``A``; with
  ``B = 0`` any init of ``A`` yields the same starting point).
- the ``alpha / r`` scaling is applied on the adapter path, mirroring
  ``lora_alpha / lora_dropout`` semantics of the reference implementation.
- dropout is applied on the *adapter* input path only (as in the official
  code), not on the frozen branch.
- ``merge()`` folds ``DeltaW`` into the base weight for inference-time
  parity with the unmerged model, bit-for-bit up to float error.

This module has **zero dependencies beyond torch** - the point of the file
is to demonstrate that the author understands the method, not the library.
Unit tests in ``tests/test_lora.py`` verify every property above, plus a
1:1 numerical comparison against Hugging Face ``peft`` when it is installed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` adding a trainable low-rank update.

    The wrapped linear layer is frozen; only ``lora_A`` / ``lora_B`` are
    trained. With ``r`` the rank and ``alpha`` the scaling constant, the
    effective update is ``scaling * B(A(x))`` with ``scaling = alpha / r``.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank must be positive, got {r}")
        in_features, out_features = base.in_features, base.out_features

        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.merged = False

        self.reset_lora_parameters()
        # LoRA weights must follow the base layer's dtype (4-bit/16-bit base).
        self.lora_A.to(base.weight.dtype)
        self.lora_B.to(base.weight.dtype)

    def reset_lora_parameters(self) -> None:
        # Official LoRA init: A ~ kaiming_uniform(a=sqrt(5)), B = 0.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)
        frozen = self.base(x)
        # DeltaW x = B(A(x)) * scaling   (dropout on the adapter input only)
        adapter = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return frozen + adapter.to(frozen.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold DeltaW into the base weight: W <- W + scaling * B @ A."""
        if self.merged:
            return
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        dtype = self.base.weight.dtype
        self.base.weight.add_(delta.to(dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Inverse of ``merge``: W <- W - scaling * B @ A."""
        if not self.merged:
            return
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        dtype = self.base.weight.dtype
        self.base.weight.sub_(delta.to(dtype))
        self.merged = False

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}, merged={self.merged}"


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Replace matching ``nn.Linear`` modules with ``LoRALinear`` in-place.

    ``target_modules`` entries are matched against the *last component* of
    each module name (e.g. ``"q_proj"`` hits every attention projection),
    or the full qualified name if it contains dots. This mirrors how
    ``peft`` resolves ``target_modules``. Returns the replaced names.
    """
    patterns = [re.compile(rf"(^|\.)({re.escape(tm)}|{tm})$") for tm in target_modules]
    replaced: list[str] = []

    # Snapshot first: replacing leaves does not add children, so iterating
    # over the frozen list is safe and O(n).
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(p.search(name) for p in patterns):
            continue
        parent = _parent_module(model, name)
        child_name = name.rsplit(".", 1)[-1]
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(name)

    if not replaced:
        raise ValueError("no modules matched target_modules - check the names")
    return replaced


def mark_only_lora_trainable(
    model: nn.Module,
    modules_to_save: Iterable[str] = (),
) -> None:
    """Freeze the base while keeping LoRA and selected task heads trainable.

    A pretrained encoder's classification head is newly initialized for this
    task. Freezing it would make the LoRA comparison invalid, so callers may
    explicitly retain modules such as ``pre_classifier`` and ``classifier``.
    """
    selected = tuple(modules_to_save)
    for name, parameter in model.named_parameters():
        keep = "lora_" in name or any(_belongs_to_module(name, module) for module in selected)
        parameter.requires_grad_(keep)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Adapter-only state dict (small checkpoint: MBs instead of GBs)."""
    return {name: p.detach().cpu() for name, p in model.named_parameters() if "lora_" in name}


def merge_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def merge_and_unload_lora(model: nn.Module) -> list[str]:
    """Merge every adapter and restore ordinary ``nn.Linear`` modules.

    The returned model can be saved and reloaded by vanilla Hugging Face
    ``AutoModel`` classes because no custom wrapper keys remain in its state
    dictionary.
    """
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        module.merge()
        parent = _parent_module(model, name)
        child_name = name.rsplit(".", 1)[-1]
        setattr(parent, child_name, module.base)
        replaced.append(name)
    return replaced


def _parent_module(root: nn.Module, qualified_name: str) -> nn.Module:
    parent_name, _, _ = qualified_name.rpartition(".")
    module = root
    for part in parent_name.split("."):
        if not part:
            continue
        module = getattr(module, part)
    return module


def _belongs_to_module(parameter_name: str, module_name: str) -> bool:
    return parameter_name.startswith(f"{module_name}.") or f".{module_name}." in (
        f".{parameter_name}."
    )
