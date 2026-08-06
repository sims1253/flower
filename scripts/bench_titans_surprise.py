"""Comprehensive _surprise_update benchmark — measures the actual win across
the regimes that matter:

  (a) eval/forward-only: the autograd path builds an inner graph and destroys
      it every step (create_graph=False). This is the regime the spec targets
      and where the analytical win should be largest.
  (b) training mode: create_graph=True keeps a graph regardless (for outer CE
      backprop), so the win is smaller but the analytical path still avoids the
      inner-graph construction overhead.

Reports the per-layer per-step saving so it can be sized for a full model.
"""
from __future__ import annotations

import torch
import torch.utils.benchmark as benchmark

from flower.config import ModelConfig
from flower.models.titans_mac import TitansMACBlock


def cfg(analytical: bool) -> ModelConfig:
    return ModelConfig(
        variant="titans_mac",
        vocab_size=128,
        d_model=768,
        num_heads=8,
        num_layers=1,
        ffn_dim=2048,
        max_seq_len=64,
        local_window=16,
        memory_slots=16,
        titans_analytical_surprise=analytical,
    )


def bench(block, x, memory):
    t = benchmark.Timer(
        stmt="block._surprise_update(memory, x)",
        globals={"block": block, "x": x, "memory": memory},
        num_threads=1,
    ).timeit(100)
    return t


def main() -> None:
    device = torch.device("cuda")
    B, T, D = 8, 64, 768
    x = torch.randn(B, T, D, device=device)
    memory = torch.randn(B, 16, D, device=device) * 0.1

    # --- eval mode: forward-only, autograd builds+destroys inner graph ---
    block_auto_eval = TitansMACBlock(cfg(False)).to(device).eval()
    block_ana_eval = TitansMACBlock(cfg(True)).to(device).eval()
    block_ana_eval.load_state_dict(block_auto_eval.state_dict())

    with torch.no_grad():
        t_auto_eval = bench(block_auto_eval, x, memory)
        t_ana_eval = bench(block_ana_eval, x, memory)

    # --- training mode: create_graph=True on the autograd side ---
    block_auto_train = TitansMACBlock(cfg(False)).to(device).train()
    block_ana_train = TitansMACBlock(cfg(True)).to(device).train()
    block_ana_train.load_state_dict(block_auto_train.state_dict())

    t_auto_train = bench(block_auto_train, x, memory)
    t_ana_train = bench(block_ana_train, x, memory)

    print(f"\n_surprise_update (B=8, S=16, D=768), 100 calls, RTX 5090:\n")
    for mode, ta, tna in [
        ("eval (no_grad, create_graph=False)", t_auto_eval, t_ana_eval),
        ("train (create_graph=True)", t_auto_train, t_ana_train),
    ]:
        sp = ta.median / tna.median
        saved = (ta.median - tna.median) * 1e6
        print(f"  {mode}:")
        print(f"    autograd:   {ta.median*1e6:8.1f} us")
        print(f"    analytical: {tna.median*1e6:8.1f} us")
        print(f"    speedup:    {sp:5.2f}x   (saves {saved:.1f} us / layer / step)\n")


if __name__ == "__main__":
    main()
