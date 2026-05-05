# Top-Tier Push Plan

Primary thesis: Shape-class-aware runner plans can reduce graph churn and transition overhead in variable-shape inference.
Honest fallback: The runtime can always use generic model-runner paths with no shape-class specialization.
Next gate: Export stable shape classes, plan churn, and graph reuse value across representative serving traces.
Missing evidence: Measured prefill-to-decode or graph-reuse gains from shape-class-aware runner dispatch.

Immediate actions:
1. Classify runner paths by stable shape families.
2. Quantify how often plan reuse would have been possible.
3. Keep the artifact as a shape-pathology study until runner specialization is operational.