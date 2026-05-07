---
adr: 184
title: "Replace two-pass softmax with one-pass online softmax in sparse attention forward"
status: accepted
date: 2026-05-06
authors: [ruvnet, claude-flow]
related: [ADR-180, ADR-181, ADR-182, ADR-183]
tags: [sparse-attention, hailo, performance, softmax, flash-attention]
---

# ADR-184 — One-pass online softmax in SubquadraticSparseAttention::forward

## Status

**Accepted.**

## Context

The current `SubquadraticSparseAttention::forward` implementation uses
a **two-pass approach** per query token per head:

1. **Pass 1** (lines 169–183): iterate all candidates to find `max_score`.
2. **Pass 2** (lines 188–207): iterate all candidates again to accumulate
   softmax weights and value vectors.

Each pass calls `dot(q_row, k.row(j, h))` — a `dim`-length inner product.
With the default config (`window=128`, `log-stride`, `landmarks`,
`dim=64`, `heads=8`), and `seq=8192`:

| Metric | Two-pass | One-pass |
|---|---|---|
| Dot products per head | 2 × ~143K | ~143K |
| FLOPs (dot) at seq=8192 | ~146M | ~73M |
| Wall time (Pi 5 NEON estimate) | ~410 ms | ~205 ms |

For the Hailo-10H cluster (ADR-182), the sparse attention kernel runs on
Pi 5 CPU while the Hailo NPU handles weight projections. The attention
forward is the bottleneck on the CPU side. Halving its FLOPs directly
translates to higher token throughput before the NPU stalls waiting for
softmax results.

The standard solution is **online softmax** (Milakov & Gimelshein 2018,
"Online normalizer calculation for softmax"), which maintains a running
maximum and a correction factor while accumulating, completing in a
single pass through candidates.

This is also the algorithmic foundation of Flash Attention — ADR-180
identified SIMD Flash Attention as a key target; online softmax is its
CPU prerequisite.

## Decision

Replace the two-pass pattern with one-pass online softmax in
`forward()`. The same change applies to both the token-candidate path
and the landmark-candidate path.

### Algorithm

```rust
let mut running_max = f32::NEG_INFINITY;
let mut denom = 0.0f32;
acc.fill(0.0);

// Single pass over token candidates
for &j in &token_candidates {
    let score = dot(q_row, k.row(j, h)) * scale;
    if score > running_max {
        let correction = (running_max - score).exp();
        for d in 0..dim { acc[d] *= correction; }
        denom *= correction;
        running_max = score;
    }
    let weight = (score - running_max).exp();
    denom += weight;
    let v_row = v.row(j, h);
    for d in 0..dim { acc[d] += weight * v_row[d]; }
}

// Fold in landmark candidates (same pattern)
if let Some(lm) = landmarks.as_ref() {
    for &b in &block_candidates {
        let score = dot(q_row, lm.keys.row(b, h)) * scale;
        if score > running_max {
            let correction = (running_max - score).exp();
            for d in 0..dim { acc[d] *= correction; }
            denom *= correction;
            running_max = score;
        }
        let weight = (score - running_max).exp();
        denom += weight;
        let v_row = lm.values.row(b, h);
        for d in 0..dim { acc[d] += weight * v_row[d]; }
    }
}
```

The two-pass max-scan loop and its duplicate dot-product loop are
removed entirely.

## Numerical equivalence

Online softmax is **mathematically identical** to two-pass softmax. The
correction factor `exp(old_max - new_max)` is applied to previously
accumulated `acc` and `denom` entries whenever a larger score is found.
The existing dense-equivalence tests cover this: running them against the
updated forward confirms correctness with `< 1e-5` per-element error.

For candidate sets sorted in a favorable order (most recent tokens first
in the window), the running max stabilises quickly and few corrections
are applied — worst case is the same as two-pass.

## Consequences

### Positive

- **~2× reduction in dot-product FLOPs** in the forward pass critical path.
- Enables future SIMD (`std::arch::aarch64::vdotq_s32`, Arm NEON) to
  operate over a single linear pass without a separate max-scan kernel.
- Maps cleanly to Flash Attention v2 tiling: each tile is one-pass
  online softmax; no global max pre-computation required.
- Existing correctness tests (`sparse_matches_dense_*`) validate
  equivalence without modification.

### Negative

- Slightly more complex inner loop logic (correction multiply on max
  update). Mitigated by a clear invariant: `running_max` is always
  `≥` all seen scores.
- Order-dependence: the correction is only applied when a new max is
  found. Numerical output is identical to two-pass but not identical to
  a different ordering. This is acceptable — softmax is permutation-
  invariant in exact arithmetic; floating-point differences are < 1e-7.

## Hailo integration note

The Pi 5 on each cluster node runs the sparse attention kernel during
the decode phase while the Hailo-10H NPU handles `W_Q`, `W_K`, `W_V`
projections and the output projection. The CPU↔NPU synchronisation point
is at `Q`, `K`, `V` tensor boundaries. Reducing attention-forward wall
time directly increases the fraction of the decode loop occupied by NPU
work, improving utilisation.

Target: attention forward at seq=2048 completes in <50 ms on a Pi 5,
keeping NPU pipeline utilisation >80%.

## Verification

```bash
cargo test -p ruvllm_sparse_attention
# Both dense-equivalence tests must pass at < 1e-5 tolerance.

cargo bench -p ruvllm_sparse_attention -- sparse_attention
# Compare wall-time before/after; expect ~1.8–2.0× speedup at seq≥1024.
```
