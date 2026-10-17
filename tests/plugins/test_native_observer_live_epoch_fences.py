# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Actual Host cross-producer/pidfd contracts, not native qualification."""
import errno
import threading
from pathlib import Path

import pytest

from vllm.plugins import native_observer_rebind as host
from test_native_observer_live_rebind import commit, pair_router as pair_router


@pytest.mark.parametrize('code',[errno.EACCES,errno.EIO,errno.ENOSPC])
def test_filesystem_integrity_error_is_not_typed_dead_observer(monkeypatch,code):
    monkeypatch.setenv(host.DIRECTORY_ENV,'fixture-only-grant-root')
    monkeypatch.setattr(host,'collector_kernel_gone',lambda *a:True)
    def failed(*a,**k):
        raise OSError(code,'fixture filesystem failure')
    monkeypatch.setattr(host.native,'_submit_fact',failed)
    with pytest.raises(OSError) as captured:
        host.native.submit_fact(Path('/fixture/socket'),{},expected_collector={'pid':1},
            expected_socket_identity=(1,1),expected_collector_source_sha256='fixture',expected_archive_identity={})
    assert not isinstance(captured.value,host.native.CollectorKernelGone)


def test_api_old_worker_first_after_commit_cannot_enter_new_collector(pair_router):
    worker,publish,command,calls,_=pair_router
    # API has a real source mapping in old epoch. This worker has never seen it.
    api=host.Router.__new__(host.Router)
    api.lock=threading.RLock()
    api.mapping_lock=threading.RLock()
    api.policy={'producer':{'pid':55},'start_id':worker.policy['start_id'],
        'receipt_sha256':worker.policy['receipt_sha256'],'producer_group':worker.policy['producer_group']}
    api._current_policy=lambda:None  # source/kernel probes have separate negative gates
    api.api_mappings={'fixture-old-queued':{'runtime_request_id':'fixture-old-queued',
        'client_request_id':'fixture/request-00000','generation':0,'source_fact_sha256':'fixture-source'}}
    commit(worker,command)
    def actual_api_registry(rid):
        response=api._read_only(dict(schema=host.COMMAND,domain=host.DOMAIN,operation='resolve',
            start_id=api.policy['start_id'],receipt_sha256=api.policy['receipt_sha256'],
            producer=worker.policy['producer'],runtime_request_id=rid,challenge_nonce='c'*64),
            worker.policy['producer'])
        return response['mapping']
    worker._resolve_api_mapping=actual_api_registry
    assert 'fixture-old-queued' not in worker.requests
    result=publish['tensor']({'runtime_request_id':'fixture-old-queued'})
    assert host.is_unavailable(result) and calls==[]
    assert worker.requests['fixture-old-queued']==0


def test_missing_original_api_mapping_is_not_local_new_epoch_guess(pair_router):
    worker,publish,command,calls,_=pair_router
    commit(worker,command)
    def unavailable(rid):
        raise ValueError('actual source API mapping not available')
    worker._resolve_api_mapping=unavailable
    with pytest.raises(ValueError,match='actual source API mapping'):
        publish['tensor']({'runtime_request_id':'fixture-no-api-binding'})
    assert calls==[] and 'fixture-no-api-binding' not in worker.requests


def test_changed_old_admission_frontier_cannot_commit_prepared_generation(pair_router):
    worker,_,command,_,_=pair_router
    prepared=worker.apply(command)
    final={key:value for key,value in command.items() if key!='contexts'}
    final.update(operation='commit',prepared_sha256=host.native.sha(host.native.canonical(prepared)),
        old_client_request_ids=['fixture/request-00001'])
    with pytest.raises(ValueError,match='exact prepared'):
        worker.apply(final)
    assert worker.generation==0


def test_pidfd_unknown_is_not_a_death_witness(pair_router):
    worker,_,_,_,_=pair_router
    with pytest.raises(ValueError,match='pidfd unavailable'):
        worker._collector_gone({'pid':777})


def test_api_epoch_lookup_does_not_wait_for_collector_transport_commit_lock():
    api=host.Router.__new__(host.Router)
    api.lock=threading.RLock();api.mapping_lock=threading.RLock()
    worker={'pid':123};api_kernel={'pid':55}
    api.policy={'producer':api_kernel,'start_id':'observer-loss-r1-ecpa',
        'receipt_sha256':'CPU-only','producer_group':[{'producer':worker,'role':'worker'}]}
    api._current_policy=lambda:None
    api.api_mappings={'queued':{'runtime_request_id':'queued','generation':0}}
    completed=threading.Event();result=[]
    def lookup():
        result.append(api._read_only(dict(schema=host.COMMAND,domain=host.DOMAIN,
            operation='resolve',producer=worker,start_id=api.policy['start_id'],
            receipt_sha256='CPU-only',challenge_nonce='c'*64,runtime_request_id='queued'),worker))
        completed.set()
    with api.lock:
        thread=threading.Thread(target=lookup,daemon=True);thread.start()
        assert completed.wait(timeout=1), 'API mapping lookup deadlocked behind transport commit'
    thread.join(timeout=1)
    assert result[0]['mapping']['generation']==0


def test_narrow_connection_loss_requires_independent_captured_pidfd_death(monkeypatch):
    monkeypatch.setenv(host.DIRECTORY_ENV,'CPU-only-grant-root')
    observed=[]
    def reset(*args,**kwargs):
        raise ConnectionResetError('CPU-only peer reset')
    monkeypatch.setattr(host.native,'_submit_fact',reset)
    collector={'pid':123,'start_ticks':456,'session':123,'boot_id':'CPU-only',
        'start_identity':'pid:123:start_ticks:456'}
    def death(kernel):
        observed.append(kernel)
        return True  # Isolated unit oracle; never a real runtime proof.
    monkeypatch.setattr(host,'collector_kernel_gone',death)
    with pytest.raises(host.native.CollectorKernelGone) as failure:
        host.native.submit_fact(Path('/fixture/socket'),{},expected_collector=collector,
            expected_socket_identity=(1,1),expected_collector_source_sha256='fixture',expected_archive_identity={})
    assert observed==[collector] and failure.value.expected_collector==collector

