<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

### Workspace Ascend Plugin (vllm-ascend)

For this multi-root workspace, you can install local `vllm-ascend` as a
platform plugin for `vllm-hust` with:

```bash
cd /home/shuhao/vllm-hust
bash scripts/install_local_ascend_plugin.sh
```

If your `vllm-ascend` repo is in a different location:

```bash
bash scripts/install_local_ascend_plugin.sh /path/to/vllm-ascend
```

This script installs `vllm-ascend` in editable mode and verifies that entry
points under `vllm.platform_plugins` are discoverable.
It defaults to lightweight mode (`COMPILE_CUSTOM_KERNELS=0`, `--no-deps`) so
you can wire the plugin in workspace even when Ascend custom-op toolchain is
not fully configured.

### Avoid Mixed Ascend Runtime (Recommended)

To avoid mixing multiple CANN/Ascend toolkit trees in one shell session,
always source a single runtime first:

```bash
cd /home/shuhao/vllm-hust
source scripts/use_single_ascend_env.sh /usr/local/Ascend/ascend-toolkit.bak.8.1/latest
```

The script now also loads `/usr/local/Ascend/nnal/atb/set_env.sh` to ensure
ATB operator runtime variables are configured. If this file is missing, install
NNAL/ATB package first.

Then run the benchmark through the wrapper (it sources the same environment
script internally):

```bash
bash scripts/run_ascend_latency_bench.sh /usr/local/Ascend/ascend-toolkit.bak.8.1/latest
```

If you omit the path, scripts use a default toolkit root suitable for this
workspace.

### One-Click Ascend Bootstrap

To make local Ascend deployment closer to a one-command flow, use:

```bash
cd /home/shuhao/vllm-hust
bash scripts/bootstrap_ascend.sh Qwen/Qwen2.5-1.5B-Instruct
```

This wrapper will:

- call Ascend manager first (local repo if present, otherwise install from PyPI)
- auto-detect and source a single Ascend runtime
- run an Ascend environment health check
- install `vllm-hust` in editable mode
- install the local `vllm-ascend` plugin
- prefer a local Hugging Face cache snapshot when available
- launch the OpenAI-compatible server with the best compatible runtime mode

Manager integration defaults:

- manager repo path: `/home/shuhao/vllm-hust-dev-hub/ascend-runtime-manager`
- manager PyPI package: `hust-ascend-manager`
- disable manager: `HUST_DISABLE_ASCEND_MANAGER=1`
- manager strict mode: `HUST_MANAGER_STRICT=1`
- manager system install steps: `HUST_MANAGER_APPLY_SYSTEM=1`
- manager PyPI override: `HUST_ASCEND_MANAGER_PYPI_SPEC='hust-ascend-manager==0.1.0'`

If you need strict `npugraph_ex` validation, set `HUST_REQUIRE_NPUGRAPH=1`
before running the script.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
