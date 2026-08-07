"""`on_epoch` — the callback that lets an application show progress.

Training is by far the longest phase of a run and it emitted nothing a caller could act on. The
prints inside the loop are for a human reading a notebook and only fire every tenth epoch, so an
application driving `train_with_metrics` had no way to tell a bar from a hang: cecelia's optical-flow
training task showed no progress at all for the whole run.

The callback is not just "a number went up" — it carries the epoch's losses, because a caller
showing progress generally wants to show what it is converging to as well.
"""
import numpy as np
import pytest

from coastal.train import train_with_metrics

RUN = dict(num_epochs=3, batch_size=2, num_workers=0, use_amp=False, device='cpu', embedding_dim=4)


def _data(T=4, H=16, W=16, n_temporal=3, n_variance=2, seed=0):
    rng = np.random.default_rng(seed)
    frames = rng.random((T, H, W)).astype(np.float32)
    temporal = [{f"m_{i}": rng.random((H, W)).astype(np.float32) for i in range(n_temporal)}
                for _ in range(T)]
    variance = [{f"softmax_ch_{i}": rng.random((H, W)).astype(np.float32)
                 for i in range(n_variance)} for _ in range(T)]
    return frames, temporal, variance


def test_it_fires_once_per_epoch_one_based():
    seen = []
    frames, temporal, variance = _data()
    train_with_metrics(frames, temporal, variance,
                       on_epoch=lambda e, n, losses: seen.append((e, n)), **RUN)
    # 1-based and complete: a caller rendering "epoch 0/3" or missing the last tick is the whole
    # class of off-by-one that makes a progress bar never reach the end.
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_it_carries_this_epoch_s_losses():
    seen = []
    frames, temporal, variance = _data()
    _, history = train_with_metrics(frames, temporal, variance,
                                    on_epoch=lambda e, n, losses: seen.append(losses), **RUN)
    assert [s['total'] for s in seen] == pytest.approx(history['total'])
    # a copy, not the live dict the loop reuses — otherwise every entry a caller kept would end up
    # holding the LAST epoch's numbers
    assert len({id(s) for s in seen}) == 3


def test_omitting_it_changes_nothing():
    frames, temporal, variance = _data()
    _, a = train_with_metrics(frames, temporal, variance, **RUN)
    _, b = train_with_metrics(frames, temporal, variance, on_epoch=lambda *_: None, **RUN)
    assert a['total'] == pytest.approx(b['total'])
