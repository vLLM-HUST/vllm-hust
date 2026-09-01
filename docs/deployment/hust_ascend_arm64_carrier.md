# vLLM-HUST Ascend arm64 carrier

The HUST carrier is a narrow overlay on the official vLLM Ascend image. The
official image continues to own CANN, torch-npu, compiled extensions, and the
platform runtime. The overlay installs only the generic extension admission,
typed scheduler-policy materialization, and scheduler invocation sites
maintained by this fork.

Build from a reviewed vLLM-HUST commit:

```bash
docker build \
  --file docker/hust-ascend-arm64.Dockerfile \
  --build-arg VLLM_HUST_REVISION="$(git rev-parse HEAD)" \
  --tag local/vllm-hust:$(git rev-parse --short HEAD)-arm64 \
  .
```

The default base is pinned by digest. Override `BASE_IMAGE` only after the same
contract, online-load, conflict, and rollback gates pass against the new base.

This carrier does not require a self-hosted GitHub Actions runner. Build and
hardware acceptance run on demand on an operator-controlled Ascend host.
