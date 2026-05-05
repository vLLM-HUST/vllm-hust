# Top-Tier Push Plan

Primary thesis: Step-level executor rebalancing can improve distributed serving when cluster resources or interconnect performance are uneven.
Honest fallback: The system can always fall back to static even partitioning across executors.
Next gate: Export per-step imbalance and migration-cost signals on representative distributed runs.
Missing evidence: Measured throughput or utilization gains from dynamic rebalancing once movement overhead is counted.

Immediate actions:
1. Instrument executor imbalance before proposing a rebalance policy.
2. Quantify the cost of moving work at step granularity.
3. Keep the paper path on imbalance evidence until a stable seam is identified.