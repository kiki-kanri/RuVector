# ruvllm_sparse_attention

Subquadratic sparse attention kernel for the ruvllm inference stack, optimised for the Hailo-10H cluster (Pi 5 + Hailo NPU nodes). Implements ADR-183 through ADR-190 plus three SOTA extensions.

## Features

| Feature | ADR | Status |
|---|---|---|
| Zero runtime dep footprint (`rand` dev-only) | ADR-183 | Accepted |
| One-pass online softmax (~2× FLOPs reduction) | ADR-184 | Accepted |
| Non-causal landmark block-exclusion fix | ADR-185 | Accepted |
| 25-test CI suite, validated on all 4 Pi 5 cluster nodes | ADR-186 | Accepted |
| Overflow-checked `Tensor3::zeros` | ADR-187 | Accepted |
| Stamp-scheme comments (cross-head dedup safety) | ADR-188 | Accepted |
| KV cache incremental decode (`decode_step`) | ADR-189 | Accepted |
| GQA/MQA support (`forward_gqa`, `forward_auto`) | ADR-190 | Accepted |
| IO-optimal tiling (`forward_flash` / `forward_gqa_flash`) | SOTA | Accepted |
| FP16 KV cache (`KvCacheF16`, feature = `fp16`) | SOTA | Accepted |
| NEON/AVX SIMD auto-vec via iterator `dot()` | SOTA | Accepted |
| H2O heavy-hitter eviction (`evict_and_append`) | SOTA | Accepted |
| Speculative decode batch (`decode_batch`) | SOTA | Accepted |
| Incremental landmark update — O(H×D) Welford | SOTA | Accepted |
| Parallel head loops (`feature = parallel`, rayon) | SOTA | Accepted |
| Cache-locality sort (`sort_candidates`) | SOTA | Accepted |

## Attention complexity

| Mode | Per-token cost | Total for N tokens |
|---|---|---|
| Dense MHA | O(T) | O(N²) |
| Sparse `forward` | O(log T) | O(N log N) |
| Sparse `decode_step` (KV cache) | O(log T) | O(N log N) |
| Flash-sparse `forward_flash` | O(log T) tiled | O(N log N) IO-optimal |

## Optional features

```toml
[dependencies]
ruvllm_sparse_attention = { version = "0.1", features = ["parallel", "fp16"] }
```

| Feature | Effect |
|---|---|
| `parallel` | Per-head loops via rayon (~4× prefill speedup on multi-core) |
| `fp16` | `KvCacheF16`: FP16 KV store, halves memory vs FP32 |

## Supported model attention types

| Model | Type | Q heads | KV heads | `forward_auto` path |
|---|---|---|---|---|
| Phi-2 | MHA | 32 | 32 | `forward` / `forward_flash` |
| Mistral-7B | GQA | 32 | 8 | `forward_gqa` / `forward_gqa_flash` |
| Llama-3-8B | GQA | 32 | 8 | `forward_gqa` / `forward_gqa_flash` |
| Llama-3.2-1B | GQA | 32 | 8 | `forward_gqa` / `forward_gqa_flash` |
| TinyLlama-1.1B | MQA | 32 | 4 | `forward_gqa` / `forward_gqa_flash` |

## Quick start

```rust
use ruvllm_sparse_attention::{
    SubquadraticSparseAttention, SparseAttentionConfig, KvCache, Tensor3,
    AttentionBackend,
};

let cfg = SparseAttentionConfig::default();
let attn = SubquadraticSparseAttention::new(cfg).unwrap();

// MHA prefill — standard
let q = Tensor3::zeros(512, 32, 128);
let k = Tensor3::zeros(512, 32, 128);
let v = Tensor3::zeros(512, 32, 128);
let out = attn.forward(&q, &k, &v).unwrap();

// MHA prefill — IO-optimal flash-sparse (same result, lower peak memory)
let out = attn.forward_flash(&q, &k, &v).unwrap();

// GQA prefill (Mistral-7B: 32 Q heads, 8 KV heads)
let q = Tensor3::zeros(512, 32, 128);
let k = Tensor3::zeros(512, 8, 128);
let v = Tensor3::zeros(512, 8, 128);
let out = attn.forward_auto(&q, &k, &v).unwrap();   // dispatches forward_gqa_flash

// Incremental decode with KV cache (O(log T) per step)
// block_size is the 4th param (landmark granularity)
let mut cache = KvCache::new(4096, 8, 128, 64);
let new_k = Tensor3::zeros(1, 8, 128);
let new_v = Tensor3::zeros(1, 8, 128);
cache.try_append(&new_k, &new_v).unwrap();
let q_new = Tensor3::zeros(1, 32, 128);
let out = attn.decode_step(&q_new, &cache).unwrap();

// Generation past max_seq via H2O eviction
cache.evict_and_append(&new_k, &new_v).unwrap();
```

## IO-optimal tiling (flash-sparse)

`forward_flash` implements a 3-phase FlashAttention-2-style tiling over the sparse mask:

- **Phase 1** — ascending KV tiles; each tile accumulates into per-query online-softmax buffers (`run_max`, `denom`, `out`). Only queries whose window intersects the tile are processed.
- **Phase 2** — scatter global tokens, log-stride positions, and landmark block-means outside the window (pre-marked so `push_unique` skips covered positions automatically).
- **Phase 3** — normalise with stored `denom` values.

This matches the standard `forward` output to <1e-5 error (verified in 4 new tests) while keeping peak intermediate memory proportional to tile size rather than the full sequence.

## KV cache API

```rust
KvCache::new(capacity, kv_heads, head_dim, block_size) -> KvCache
cache.try_append(&k, &v)          // Err if full
cache.append_all(&k, &v)          // bulk prefill (multi-token tensor)
cache.is_full()
cache.reset()
cache.evict_and_append(&k, &v)    // H2O eviction — removes lowest-score non-recent token
```

`IncrementalLandmarks` inside `KvCache` updates landmark block-means with a Welford online pass (O(H×D) per token) instead of rebuilding from scratch (O(T×H×D)).

## FP16 KV cache (feature = `fp16`)

```rust
use ruvllm_sparse_attention::KvCacheF16;

let mut cache = KvCacheF16::new(4096, 8, 128, 64);
cache.try_append(&k_f32, &v_f32).unwrap();   // stored as f16
let out = attn.decode_step_f16(&q, &cache).unwrap(); // f16→f32 inline
```

Memory at seq=8192, 8 KV heads, dim=128: **1.07 GB** (vs 2.15 GB FP32).

## Benchmarks

Config: 8 heads, dim=64, window=128, block_size=64, causal=true, log-stride+landmarks enabled.

| seq | x86-64 (AMD Ryzen) | Pi 5 Cortex-A76 | edge reduction vs dense |
|---|---|---|---|
| 512 | 13.1 ms | 85.8 ms | 2.2× |
| 1024 | 28.4 ms | 190.5 ms | 4.0× |
| 2048 | 60.1 ms | 401.0 ms | 7.7× |
| 4096 | 126.5 ms | 836.2 ms | 15.0× |
| 8192 | 262.6 ms | ~1,660 ms (est.) | 29.3× |

Pi 5 measurements collected on `cognitum-v0` (Raspberry Pi 5 Model B Rev 1.1,
aarch64, compiled with `-C target-cpu=cortex-a76 -C target-feature=+lse,+rcpc,+fp16,+crc`).

`flash-sparse` bench group added (tile=128): run with `cargo bench -p ruvllm_sparse_attention`.

## Hailo-10H KV cache memory (Mistral-7B, seq=8192)

```
FP32 MHA (naive):  8192 × 32 × 128 × 2 × 4 bytes = 8.6 GB  — does not fit
FP32 GQA:          8192 ×  8 × 128 × 2 × 4 bytes = 2.1 GB  — fits in Hailo-10H DDR4
FP16 GQA:          8192 ×  8 × 128 × 2 × 2 bytes = 1.1 GB  — 52% smaller
```

## Sparse edge estimates (default config)

```
seq       dense_causal   sparse   reduction
512          131,328      59,778     2.2×
1,024        524,800     129,858     4.0×
2,048      2,098,176     272,130     7.7×
4,096      8,390,656     560,834    15.0×
8,192     33,558,528   1,146,498    29.3×
16,384   134,225,920   2,334,274    57.5×
32,768   536,887,296   4,742,658   113.2×
```

## Cluster validation

All 25 tests pass on all 4 Hailo-10H cluster nodes (release build, aarch64):

| Node | Result |
|---|---|
| cognitum-v0 | 25/25 ✓ |
| cognitum-v1 | 25/25 ✓ |
| cognitum-cluster-2 | 25/25 ✓ |
| cognitum-cluster-3 | 25/25 ✓ |

## Build and test

```bash
# Local
cargo test -p ruvllm_sparse_attention --lib

# With optional features
cargo test -p ruvllm_sparse_attention --lib --features parallel,fp16

# Cross-compile for Pi 5 (Cortex-A76)
RUSTFLAGS="-C target-cpu=cortex-a76 -C target-feature=+lse,+rcpc,+fp16,+crc" \
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  cargo test -p ruvllm_sparse_attention --lib --target aarch64-unknown-linux-gnu

# Benchmark (includes flash-sparse group)
cargo bench -p ruvllm_sparse_attention
```

## License

MIT
