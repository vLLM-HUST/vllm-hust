# Request preemption policy extension

V1 can delegate the choice of a request to preempt without exposing mutable
scheduler objects. Configure `--preemption-policy package.module.PolicyClass`.
The class must implement:

```python
def select_victim(self, context: PreemptionContext) -> str | None:
    ...
```

`PreemptionContext` and its candidate records are frozen snapshots. The return
value must be one of the candidate request IDs; returning `None` delegates that
decision to the built-in FCFS or priority policy. A class may optionally define
`from_vllm_config(vllm_config)` for construction from engine configuration.

Construction and contract errors fail engine startup. A runtime exception or
invalid request ID permanently disables the external policy in that engine
process and restores the built-in policy, so repeated scheduling cannot keep
executing a faulty extension. The scheduler exports cumulative
`vllm:preemption_policy_events` and `vllm:preemption_policy_enabled` metrics.
These distinguish a configured policy from one that is still effective.

The API does not expose `Request`, request queues, the KV-cache manager, or
scheduler mutation methods. This keeps the policy surface versionable and lets
the scheduler retain ownership of queue ordering, budget rollback, KV release,
and preemption accounting.
