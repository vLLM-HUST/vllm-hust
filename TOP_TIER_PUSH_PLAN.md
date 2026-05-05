# Top-Tier Push Plan

Primary thesis: Prefix lookup state and metadata carriage deserve source-level redesign because reuse quality depends on more than a top-level cache hit bit.
Honest fallback: The runtime can keep using high-level prefix caching with standard lookup behavior and simple eviction.
Next gate: Quantify lookup latency, metadata propagation cost, and hit quality under shared-prefix workloads.
Missing evidence: Measured reuse-quality gains or lookup-cost reductions from deeper prefix-state management.

Immediate actions:
1. Instrument the lookup boundary rather than proposing policies first.
2. Compare hit quality against the current standard path.
3. Keep claims on metadata-carriage value until lookup-path evidence is solid.