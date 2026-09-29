# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Region-relative descriptor capture on the CPU offloading handler.

The layout capture runs at the copy site and must turn absolute copy pointers
into region-relative offsets before anything leaves the process. These tests
use CPU tensors, whose ``data_ptr`` values exercise the same arithmetic.
"""

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.observability import (
    MAX_DESCRIPTOR_REGIONS,
    KVRegionDescriptor,
)
from vllm.v1.kv_offload.cpu.gpu_worker import build_region_descriptors


def test_offsets_are_relative_to_each_region():
    src = torch.zeros(128, dtype=torch.int8)
    dst = torch.zeros(128, dtype=torch.int8)
    all_src = np.array([src.data_ptr() + 32, src.data_ptr() + 64], dtype=np.int64)
    all_dst = np.array([dst.data_ptr() + 0, dst.data_ptr() + 96], dtype=np.int64)
    all_sizes = np.array([32, 32], dtype=np.int64)

    descriptors, dropped = build_region_descriptors(
        [(0, 0, 2)], [src], [dst], all_src, all_dst, all_sizes
    )

    assert dropped == 0
    assert descriptors == (
        KVRegionDescriptor(
            src_region_id=0, dst_region_id=0, src_offset=32, dst_offset=0, size=32
        ),
        KVRegionDescriptor(
            src_region_id=0, dst_region_id=0, src_offset=64, dst_offset=96, size=32
        ),
    )
    # Only relative quantities survive; no address can be reconstructed.
    for descriptor in descriptors:
        assert 0 <= descriptor.src_offset < src.numel()
        assert 0 <= descriptor.dst_offset < dst.numel()


def test_layout_entries_keep_region_ids_and_op_order():
    src = [torch.zeros(128, dtype=torch.int8) for _ in range(2)]
    dst = [torch.zeros(128, dtype=torch.int8) for _ in range(2)]
    all_src = np.array([src[1].data_ptr() + 8, src[1].data_ptr() + 16], dtype=np.int64)
    all_dst = np.array([dst[1].data_ptr() + 8, dst[1].data_ptr() + 16], dtype=np.int64)
    all_sizes = np.array([8, 8], dtype=np.int64)

    descriptors, dropped = build_region_descriptors(
        [(1, 0, 2)], src, dst, all_src, all_dst, all_sizes
    )

    assert dropped == 0
    assert [descriptor.src_region_id for descriptor in descriptors] == [1, 1]
    assert [descriptor.dst_region_id for descriptor in descriptors] == [1, 1]
    assert [descriptor.src_offset for descriptor in descriptors] == [8, 16]


def test_inventory_is_bounded_and_counts_dropped_descriptors():
    num_ops = MAX_DESCRIPTOR_REGIONS + 8
    src = torch.zeros(num_ops, dtype=torch.int8)
    dst = torch.zeros(num_ops, dtype=torch.int8)
    all_src = np.array([src.data_ptr() + i for i in range(num_ops)], dtype=np.int64)
    all_dst = np.array([dst.data_ptr() + i for i in range(num_ops)], dtype=np.int64)
    all_sizes = np.ones(num_ops, dtype=np.int64)

    descriptors, dropped = build_region_descriptors(
        [(0, 0, num_ops)], [src], [dst], all_src, all_dst, all_sizes
    )

    assert len(descriptors) == MAX_DESCRIPTOR_REGIONS
    assert dropped == 8


def test_no_copy_ops_produces_no_descriptors():
    empty = np.array([], dtype=np.int64)

    descriptors, dropped = build_region_descriptors([], [], [], empty, empty, empty)

    assert descriptors == ()
    assert dropped == 0
