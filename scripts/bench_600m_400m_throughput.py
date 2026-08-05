#!/usr/bin/env python3
"""Throughput A/B: 400M vs 600M @ seq=32K, the production path (compile+flex+
Muon+ffn-checkpoint). Reports steady-state tokens/sec — the number train.py
emits — plus peak VRAM. Decides which to run.

Single config arg: '400' or '600'. Run as separate processes with sleeps between
(GPU allocator is flaky across consecutive heavy runs).
"""
import sys, torch, time, gc
from flower.config import ModelConfig, TrainingConfig
from flower.models import build_model
from flower.models.base import prebuild_attention_masks
from flower.optim import build_optimizer

which = sys.argv[1]  # '400' or '600'
arch = {'400': (1024, 24, 16, 4096, 0.0018), '600': (1280, 28, 20, 5120, 0.0015)}[which]
d, L, H, ffn, muon_lr = arch
seq, batch, accum = 32768, 1, 8

torch.manual_seed(0)
cfg = ModelConfig(variant='vanilla_local', vocab_size=16384, d_model=d, num_heads=H,
    num_layers=L, ffn_dim=ffn, max_seq_len=seq, local_window=2048,
    norm_type='rmsnorm', ffn_activation='swiglu', ffn_param_match=True,
    qk_norm=True, use_bias=False, init_scheme='scaled', flex_attention=True,
    activation_checkpoint='ffn')
model = build_model(cfg).to('cuda').to(torch.bfloat16)
np = sum(p.numel() for p in model.parameters())/1e6
print(f"{which}M: {np:.0f}M params  d{d}/L{L}/H{H}  seq={seq} b={batch} accum={accum}", flush=True)
prebuild_attention_masks(model, seq, torch.device('cuda'))
opt = build_optimizer(model, TrainingConfig(optimizer='muon', muon_lr=muon_lr, gradient_accumulation_steps=accum))
optims = opt if isinstance(opt, list) else [opt]
model = torch.compile(model, mode='default', dynamic=False)
model.train()

# synthetic tokens (kernel time is shape/dtype-dependent, not content)
ids = torch.randint(0, 16384, (batch, seq), device='cuda')
labels = ids.clone()

# warmup: one full accumulated step (compiles + materializes optimizer state)
for _ in range(accum):
    for o in optims: o.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(ids, labels=labels)['loss']
    (loss/accum).backward()
for o in optims: o.step()
torch.cuda.synchronize()
print(f"warmup done, loss={float(loss):.4f}, alloc={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# timed: 3 accumulated steps, measure end-to-end (the train.py metric)
torch.cuda.reset_peak_memory_stats()
N_STEPS = 3
start = time.perf_counter()
for _ in range(N_STEPS):
    for o in optims: o.zero_grad(set_to_none=True)
    for _ in range(accum):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(ids, labels=labels)['loss']
        (loss/accum).backward()
    for o in optims: o.step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
# train.py convention: tokens = every micro-step's input_ids.numel() accumulated.
tokens = N_STEPS * accum * batch * seq
peak = torch.cuda.max_memory_allocated()/1e9
tok_s = tokens / elapsed
print(f"RESULT {which}M: {tok_s:.0f} tok/s  ({elapsed/N_STEPS:.1f}s/step, {accum} micro-steps)  peak={peak:.2f}GB  loss={float(loss):.4f}", flush=True)
