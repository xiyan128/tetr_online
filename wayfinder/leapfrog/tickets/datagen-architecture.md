---
id: T09
title: Datagen + training architecture — Mac-small, cloud-scalable
labels: [wayfinder:prototype]
status: open
assignee:
blocked-by: [T02, T03]
---

## Question

Design the self-play data plant that hits the throughput floor locally and scales out provider-agnostic:

- **Actor design**: parallel games per process, batched leaf inference (per-sibling-group BLAS vs cross-game fusion server — the deferred-pending-profile question), backend abstraction (BLAS / ANE-CoreML / cloud GPU) behind one seam.
- **Scale-out contract**: containerized datagen worker + shard upload; local Mac = the same worker at n=1; no provider assumed. What does the training side need (PyTorch, single-GPU first, DDP-ready)?
- **Durability**: resumable datagen (durable = shards ∩ sidecar, byte-identical regeneration — proven pattern), atomic shard writes, ShardWriter resume numbering (a known rl-branch bug class not to re-inherit).
- **Seed discipline**: an explicit seed-region allocator/registry (today caller-owned `--seeds BASE` — a leak risk across datagen/duels/gates at fleet scale).
- What lands on master vs stays campaign-local.

Cheap spikes are welcome (this is a prototype ticket): e.g. a 100-line fusion-server-vs-local-batch profile before committing.

Output: architecture decision + build plan with the throughput model's floor as its acceptance test.
