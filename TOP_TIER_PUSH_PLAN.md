# Top-Tier Push Plan

Primary thesis: Worker control and data execution should be decoupled so host-side coordination does not dominate low-latency inference steps.
Honest fallback: The runtime can continue using coupled control and execution loops with standard coordination overhead.
Next gate: Measure control-path latency, inter-step bubble time, and command batching efficiency across parallel configurations.
Missing evidence: End-to-end latency reductions from a real decoupled worker-control path.

Immediate actions:
1. Instrument worker control latency before changing the worker architecture.
2. Separate control overhead from model execution time in one reproducible benchmark.
3. Keep the line pathology-first until the control bottleneck is firmly measured.