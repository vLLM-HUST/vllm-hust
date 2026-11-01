"""Regression checks for the runtime branch's cache-block count contract."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator


class HybridCacheBlockCountTest(unittest.TestCase):
    def coordinator(self, managers):
        obj = object.__new__(HybridKVCacheCoordinator)
        obj.single_type_managers = managers
        obj.scheduler_block_size = 16
        obj.retention_interval = 64
        return obj

    def manager(self, count, *, eagle=False, block_size=8):
        return SimpleNamespace(
            use_eagle=eagle, block_size=block_size,
            cache_blocks=Mock(return_value=count),
        )

    def test_count_uses_maximum_and_preserves_alignment(self):
        managers = [self.manager(2), self.manager(5)]
        request = object()
        result = self.coordinator(managers).cache_blocks(request, 35)
        self.assertEqual(result, 5)
        for manager in managers:
            manager.cache_blocks.assert_called_once_with(
                request, 32, retention_interval=64,
            )

    def test_eagle_lookahead_is_clamped_without_changing_count(self):
        manager = self.manager(3, eagle=True)
        request = object()
        self.assertEqual(self.coordinator([manager]).cache_blocks(request, 35), 3)
        manager.cache_blocks.assert_called_once_with(
            request, 35, retention_interval=64,
        )

    def test_empty_and_no_new_blocks_return_zero(self):
        self.assertEqual(self.coordinator([]).cache_blocks(object(), 0), 0)
        self.assertEqual(
            self.coordinator([self.manager(0)]).cache_blocks(object(), 0), 0,
        )


if __name__ == '__main__':
    unittest.main()
