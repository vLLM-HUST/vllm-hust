ARG BASE_IMAGE=quay.io/ascend/vllm-ascend@sha256:f4c89c293e076453e9eef9edb5fb9669740dccbd3c48619a9f976d775fc29b81
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VLLM_HUST_REVISION=unknown

LABEL org.opencontainers.image.source="https://github.com/vLLM-HUST/vllm-hust" \
      org.opencontainers.image.revision="${VLLM_HUST_REVISION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.description="vLLM-HUST typed extension host overlay for Ascend arm64"

# Keep the carrier narrow: the official Ascend image owns the platform runtime.
# This layer replaces only the generic extension admission and materialization
# files maintained by the vLLM-HUST thin fork.
COPY vllm/plugins/ /vllm-workspace/vllm/vllm/plugins/
COPY vllm/envs.py /vllm-workspace/vllm/vllm/envs.py
COPY vllm/distributed/kv_transfer/kv_connector/factory.py \
     /vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py
COPY vllm/v1/core/sched/victim_selector.py \
     /vllm-workspace/vllm/vllm/v1/core/sched/victim_selector.py
COPY docker/patch_ascend_v023_scheduler.py /tmp/patch_ascend_v023_scheduler.py

RUN python3 /tmp/patch_ascend_v023_scheduler.py && \
    rm /tmp/patch_ascend_v023_scheduler.py && \
    test "$(uname -m)" = "aarch64" && \
    grep -q 'vllm.scheduler.policy.v1' \
      /vllm-workspace/vllm/vllm/plugins/contracts.py && \
    grep -q '_materialize_typed_victim_selector' \
      /vllm-workspace/vllm/vllm/v1/core/sched/victim_selector.py && \
    grep -q 'get_victim_selector' \
      /vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py && \
    grep -q 'self.victim_selector.pick_victim' \
      /vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py
