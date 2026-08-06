# Titans surprise without autograd: a closed-form gradient for memory writes

*2026-08-05*

The [Titans](https://arxiv.org/abs/2501.00663) family of memory-augmented
transformers has one defining trick: the "surprise" signal that gates each
memory write is the *gradient* of an inner associative-retrieval loss with
respect to the current memory state. Tokens that surprise the memory (i.e.,
that the memory can't yet reconstruct) get written harder. It's an elegant
idea — and in every implementation we've seen, including the original paper,
that gradient is computed with `torch.autograd.grad`.

That means every single forward step does this dance: detach the memory,
create a fresh `requires_grad` leaf, build an inner autograd graph for the
retrieval loss, differentiate it once, and tear the graph down again. In
training the graph is even *retained* so the outer cross-entropy can backprop
through the surprise. Everyone — us included — has treated this per-step graph
construction as an intrinsic cost of the method.

It isn't. The inner loss is a softmax-weighted retrieval, and its gradient
with respect to the memory slots has a closed form: a handful of batched
einsums. No inner graph is ever built. And because every operation in that
closed form is a standard differentiable PyTorch op, the *outer* gradient —
cross-entropy flowing back through the key/value projections, the learned
write-rate, and across layers — is fully preserved. We remove a whole category
of overhead without touching the learning dynamics.

This post walks through the result, the one subtle trap in the derivation, and
what we measured.

## What the surprise is, and why it's expensive

Concretely, the Titans inner loss asks: *given the current memory, can I
retrieve this token's value by attending over the memory with this token's
key?* Per batch element, with memory slots `M_s` (`s = 0..S-1`), a projected
key `k`, and a projected target value `v`:

```
scores_s = <M_s, k> / sqrt(D)        # attention score per slot
w_s      = softmax(scores)_s         # attention weight per slot
p        = sum_s w_s * M_s           # predicted value
loss     = ||p - v||^2 / (B * D)     # MSE, mean over batch and feature
```

The surprise signal is the *negative* gradient of this loss w.r.t. each slot,
`-d(loss)/d(M_s)` — the direction that reduces retrieval error. Memory is then
updated as `M_{t+1} = (1 - alpha) * M_t + alpha * write_scale * surprise`,
with `alpha` a learned per-slot rate.

The expensive part is getting `d(loss)/d(M_s)`. The standard way is
`torch.autograd.grad(loss, memory)`, which constructs an autograd graph for the
softmax + weighted-sum + MSE, walks it backward once, and frees it. At every
layer, every step. That graph build/teardown is pure overhead — the same
overhead that makes `torch.autograd.functional.jacobian` slow next to a
hand-derived gradient.

## The closed form

`M_s` enters the loss through two paths, and the gradient is their sum:

1. **Direct.** `M_s` appears in the predicted value `p = sum_s w_s * M_s`.
   Differentiating gives `(2/(B*D)) * w_s * a`, where `a = p - v` is the
   residual.

2. **Through softmax.** `M_s` also appears in the score `s_s = <M_s, k>/sqrt(D)`,
   which feeds *every* softmax weight. The softmax Jacobian couples them:
   `d(p)/d(s_s) = w_s * (M_s - p)`, and chaining through gives a term
   `(2*w_s/(B*D*sqrt(D))) * <a, M_s - p> * k`.

Adding them, the full gradient per slot is:

```
d(loss)/d(M_s) = (2/(B*D)) * w_s * [ a + (<a, M_s - p> / sqrt(D)) * k ]
```

and the surprise is its negation. In code that's three einsums and some
broadcasting — no graph:

```python
B, S, D = long_mem.shape
scores  = torch.einsum("bsd,bd->bs", long_mem, key) / sqrt(D)
w       = torch.softmax(scores, dim=-1)                       # (B, S)
p       = torch.einsum("bs,bsd->bd", w, long_mem)             # (B, D)
a       = p - value                                           # (B, D)
dot     = torch.einsum("bd,bsd->bs", a, long_mem - p[:, None])# (B, S) = <a, M_s - p>
factor  = 2.0 / (B * D)                                       # MSE mean reduction
grad    = factor * w[..., None] * (
              a[:, None] + (dot / sqrt(D))[..., None] * key[:, None]
          )                                                   # (B, S, D)
surprise = -grad
```

That's it. The whole inner-graph machinery collapses to O(S·D) work — a few
batched dot products.

## The trap: don't drop the constant

Here's the subtle part, and it's where a naive derivation goes wrong.

The factor `2/(B*D)` comes from `F.mse_loss` with default mean reduction,
which divides by *both* the batch and feature dimensions — not just `D`. It's
tempting to look at the downstream update `alpha * write_scale * surprise`,
notice that `alpha` and `write_scale` are learned, and declare the constant
"absorbed" — drop it, ship the simpler `w_s * [...]` form.

Don't. Two things break:

- **Numerical equivalence.** The whole point of the closed form is that it
  matches `torch.autograd.grad` to floating-point precision. Drop the
  `2/(B*D)` and you're off by exactly that factor — the gate fails, and the
  Titans write rule silently changes. We pinned this with a test: the closed
  form agrees with autograd at ~1e-9 in fp32 and ~1e-4 in bf16. With the
  constant dropped, it wouldn't.
- **Checkpoint compatibility.** `alpha` is initialised to `sigmoid(-2.0) ≈
  0.12` and `write_scale` to `1.0`. Those aren't free parameters waiting to
  absorb an arbitrary rescaling — they're learned starting from specific
  values that make the write rate well-behaved at init. Change the upstream
  constant and either old checkpoints stop reproducing or the effective
  learning dynamics shift.

So the constant stays. It's one line, but it's the difference between a drop-in
optimisation and a silent regression.

## What we measured

We benchmarked the isolated `_surprise_update` call (the surprise computation
plus the memory write, no attention or FFN) at `B=8, S=16, D=768`, 100 calls,
on an RTX 5090:

| mode | autograd | analytical | speedup | saves (per layer, per step) |
|---|---|---|---|---|
| eval (`no_grad`, `create_graph=False`) | 5249 µs | 3397 µs | **1.55×** | 1.85 ms |
| training (`create_graph=True`) | 4178 µs | 3407 µs | **1.23×** | 0.77 ms |

We'll be honest about what this is. The speedup is real but modest in absolute
terms, because the surprise computation is a small fraction of a full
transformer block — attention and the FFN dominate. For a 28-layer Titans
model that's roughly 21 ms saved per training step and 52 ms per eval step.
Useful, not transformative.

What *is* transformative is the shape of the result: a whole class of overhead
(graph construction for a per-step inner gradient) is just gone. The
implementation is shorter than the autograd version, has no `requires_grad`
leaf, no `enable_grad` context, no `create_graph`/`retain_graph` flags to get
right. The gradient falls out of the forward pass for free.

## Why this matters beyond the speedup

Titans is one variant, and surprise-gated memory is one mechanism. But the
pattern is general: any "test-time gradient" memory rule — where a write is
gated by the gradient of some cheap inner objective — is a candidate for the
same treatment. If the inner objective is simple enough (and most
retrieval-style losses are), its gradient has a closed form, the inner graph
can be eliminated, and the outer training gradient stays intact because the
closed form is just differentiable tensor ops.

The original Titans paper and its follow-ups treat the autograd call as the
cost of doing business. The contribution here is showing it doesn't have to
be: the surprise of a softmax-retrieval MSE has an exact, cheap analytical
form, and the method stays fully end-to-end differentiable with it. That's a
clean, self-contained result — and if convergence runs at training scale
confirm the loss curve is unchanged (which the 1e-9 fp32 equivalence strongly
implies), it's a citable optimisation for any Titans-style memory module.

---

*The full derivation, the corrected constant, the equivalence table, and the
benchmark script live in the repository: `NEXT_IDEAS.md` §7, with the
implementation in `flower/models/titans_mac.py` and tests in
`tests/test_titans_surprise.py`.*
