---
id: T05
title: Round-0 net on the clean stack
labels: [wayfinder:task]
status: open
assignee:
blocked-by: []
---

## Question

Produce the starting two-board P+V net on master's `tetr-nn` stack (the rl-worktree round-0 weights predate the v2 rebuild and its BC corpus carries the F27 seat-alternation bias):

1. Regenerate the BC corpus with the seat-alternation fix (mixed-strength CC2 ladder, decision shards via store-what-you-serve).
2. Train the P+V net (python/tetrnn), export, verify PyTorch↔Rust goldens.
3. Record receipts (corpus stats, start-gates: std(z_hat), top-1, AUC) and commit weights or document the artifact location.

This is incumbent-zero and the prerequisite for measuring search gain. Whether round 1 actually warm-starts from it is the design freeze's call — permitted by the purity contract either way.
