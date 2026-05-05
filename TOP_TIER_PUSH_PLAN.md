# Top-Tier Push Plan

Primary thesis: KV cache defragmentation should be a first-class runtime concern because fragmentation can cap batch size and waste residency capacity.
Honest fallback: The runtime can continue to rely on standard PagedAttention block allocation and tolerate fragmentation overhead.
Next gate: Instrument fragmentation ratio, relocation bytes, and compaction payoff on serving-like workloads.
Missing evidence: Demonstrated batch-size or latency improvements from safe defragmentation under active load.

Immediate actions:
1. Measure fragmentation before designing compaction.
2. Identify a safe relocation boundary for live requests.
3. Keep policy work gated on one reproducible fragmentation pathology.