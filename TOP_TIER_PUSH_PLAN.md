# Top-Tier Push Plan

Primary thesis: Decode budget arbitration should be explicit and source-level because greedy allocation leaves throughput and fairness on the table under contention.
Honest fallback: The runtime can continue greedy decode admission bounded only by hard memory or token limits.
Next gate: Export decode-budget contention, fairness, and throughput-loss metrics for contested serving runs.
Missing evidence: Throughput-versus-fairness Pareto curves that justify a real arbitration mechanism.

Immediate actions:
1. Instrument budget contention before adding any arbitration policy.
2. Separate hard resource limits from scheduler policy loss.
3. Keep this line on evidence collection until the tradeoff surface is visible.