# Top-Tier Push Plan

Primary thesis: Host-device state staging should be coordinated by runtime state and scheduler hints instead of as an ad hoc OOM escape hatch.
Honest fallback: The runtime can fall back to fail-on-OOM or manual offload decisions with no integrated staging control.
Next gate: Measure staging delay, overlap ratio, and refault cost on large-context or oversized-model paths.
Missing evidence: Real serving wins that remain positive after counting migration and refault overhead.

Immediate actions:
1. Instrument the critical-path blocking caused by current staging behavior.
2. Separate background staging opportunity from unavoidable synchronous movement.
3. Keep this line on staging pathology until a repeatable offload seam is isolated.