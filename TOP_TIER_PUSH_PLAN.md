# Top-Tier Push Plan

Primary thesis: Request-queue promotion should use reuse value and service urgency rather than only arrival order.
Honest fallback: The runtime can keep FCFS or simple queue discipline with no promotion-aware semantics.
Next gate: Export promotion benefit, starvation risk, and queue-turnover metrics on representative workloads.
Missing evidence: p99 latency or throughput gains from promotion logic once fairness and starvation costs are counted.

Immediate actions:
1. Measure queue-pathology signals before designing promotion rules.
2. Tie queue decisions to one concrete reuse or urgency signal.
3. Keep this line as a queue-pathology study until promotion benefit is reproducible.