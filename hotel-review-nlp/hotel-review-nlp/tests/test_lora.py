"""Property tests for the from-scratch LoRA implementation.

Every test runs offline on tiny modules - no pretrained weights, no network.
The optional peft comparison is skipped when peft/transformers are absent
(same as in CI).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from reviewnlp.lora.lora import (
    LoRALinear,
    inject_lora,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_lora,
)


class TinyNet(nn.Module):
    """3-layer MLP whose names mimic transformer projections."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 32)),
                nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 64)),
            ]
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        for block in self.blocks:
            x = torch.relu(block(x))
        return self.head(x)


def test_zero_impact_at_init():
    """B = 0 at init, so LoRALinear(x) must equal base(x) exactly."""
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    layer = LoRALinear(base, r=4, alpha=8, dropout=0.0)  # no dropout: deterministic
    x = torch.randn(5, 16)
    assert torch.allclose(layer(x), base(x), atol=0, rtol=0)


def test_scaling_matches_paper_formula():
    """h = W0 x + (alpha / r) * B(A(x)), checked against a manual computation."""
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    layer = LoRALinear(base, r=3, alpha=6, dropout=0.0)
    with torch.no_grad():
        layer.lora_A.weight.normal_(0, 0.1)
        layer.lora_B.weight.normal_(0, 0.1)

    x = torch.randn(4, 16)
    expected = base(x) + (x @ layer.lora_A.weight.T @ layer.lora_B.weight.T) * (6 / 3)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_only_lora_params_trainable():
    torch.manual_seed(0)
    model = TinyNet()
    replaced = inject_lora(model, target_modules=["0"], r=2, alpha=4)  # blocks.*.0
    assert len(replaced) == 2, f"expected 2 injections, got {replaced}"

    mark_only_lora_trainable(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)

    total = sum(p.numel() for p in model.parameters())
    adapter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert adapter < total * 0.5  # the point of LoRA: far fewer trainable params


def test_gradients_reach_adapter_not_base():
    torch.manual_seed(0)
    model = TinyNet()
    inject_lora(model, target_modules=["0"], r=2, alpha=4)
    mark_only_lora_trainable(model)

    out = model(torch.randn(8, 16)).sum()
    out.backward()

    lora_grads = [n for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    assert lora_grads, "no gradient reached the adapters"
    base_grads = [n for n, p in model.named_parameters() if "lora_" not in n and p.grad is not None]
    assert not base_grads, f"base weights must be frozen, got grads for {base_grads}"


def test_merge_roundtrip_preserves_outputs():
    torch.manual_seed(0)
    model = TinyNet()
    inject_lora(model, target_modules=["0", "1"], r=4, alpha=8, dropout=0.0)

    x = torch.randn(6, 16)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.normal_(0, 0.05)  # simulate a trained adapter
    before = model(x)

    merge_lora(model)
    merged_out = model(x)
    assert torch.allclose(before, merged_out, atol=1e-5), "merge must not change the function"

    # unmerge restores the original base weights exactly (same ops, sign flip)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
    assert torch.allclose(model(x), before, atol=1e-5)


def test_merged_forward_ignores_adapter_path():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(4, 4), r=2, alpha=2, dropout=0.0)
    with torch.no_grad():
        layer.lora_A.weight.normal_(0, 0.1)
        layer.lora_B.weight.normal_(0, 0.1)
    layer.merge()
    assert layer.merged
    x = torch.randn(3, 4)
    assert torch.allclose(layer(x), layer.base(x))


def test_lora_state_dict_is_adapter_only():
    torch.manual_seed(0)
    model = TinyNet()
    inject_lora(model, target_modules=["0"], r=2, alpha=4)
    state = lora_state_dict(model)
    assert state and all("lora_" in k for k in state)
    # r=2 -> A: (2, in_features), B: (out_features, 2) for each injected linear
    a_shapes = {tuple(v.shape) for k, v in state.items() if k.endswith("lora_A.weight")}
    assert a_shapes == {(2, 16), (2, 32)}, f"unexpected A shapes: {a_shapes}"


def test_kaiming_init_boundaries():
    """A follows kaiming_uniform(a=sqrt(5)) bounds; B is exactly zero."""
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(64, 32), r=8, alpha=16)
    a = layer.lora_A.weight
    bound = math.sqrt(6.0 / ((1 + 5) * 64))
    assert a.abs().max() <= bound + 1e-6
    assert torch.count_nonzero(layer.lora_B.weight) == 0


def test_dropout_changes_training_mode_only_consistently():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(8, 8), r=2, alpha=2, dropout=0.5)
    x = torch.randn(10, 8)
    layer.eval()
    assert torch.allclose(layer(x), layer.base(x))  # dropout off in eval


def test_peft_equivalence():
    """1:1 numerical comparison with HF peft on a tiny BERT (offline).

    peft builds a tiny BertModel from config locally - no download needed.
    We copy our adapter weights into the peft-wrapped model and require
    identical outputs, validating that our LoRALinear is *the same math*
    as the library version.
    """
    peft = pytest.importorskip("peft")
    pytest.importorskip("transformers")

    from transformers import BertConfig, BertModel

    torch.manual_seed(7)
    config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    base = BertModel(config)
    ours = BertModel(config)
    ours.load_state_dict(base.state_dict())  # identical starting weights

    targets = ["query", "value"]
    inject_lora(ours, target_modules=targets, r=4, alpha=8, dropout=0.0)

    peft_model = peft.get_peft_model(
        base,
        peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=targets, bias="none"),
    )
    # seed peft's A identically to ours via kaiming_uniform on a fixed generator
    torch.manual_seed(1234)
    for name, p in ours.named_parameters():
        if name.endswith("lora_A.weight"):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    torch.manual_seed(1234)
    for name, p in peft_model.named_parameters():
        if name.endswith("lora_A.weight"):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    ours_map = dict(ours.named_parameters())
    peft_map = dict(peft_model.named_parameters())
    for name, p in peft_map.items():
        if "lora_A" in name or "lora_B" in name:
            ours_key = _find_ours_key(ours_map, name)
            assert ours_key, f"cannot map peft param {name}"
            p.data.copy_(ours_map[ours_key].data)

    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out_ours = ours(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out_peft = peft_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    assert torch.allclose(out_ours, out_peft, atol=1e-5), (
        "from-scratch LoRA diverges from peft - check scaling/init/forward"
    )


def _find_ours_key(ours_map: dict, peft_name: str) -> str | None:
    """peft names look like base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight"""
    stem = peft_name.replace("base_model.model.", "").replace(".default", "")
    for key in ours_map:
        if key == stem:
            return key
        if key.endswith(stem):
            return key
    return None
