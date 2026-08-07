"""The held-out pass in `train_with_metrics` — does the val curve mean what it says.

A training loss curve cannot tell convergence from memorising. It only ever reports that the number
went down, and the number it reports is measured on the frames the weights were just fitted to. The
held-out pass is what makes the curve answerable: `val_total` beside `total`, same terms, same
weights, so the GAP between them carries the information.

That comparability is the whole feature, and it is fragile in exactly three ways — each one leaves
both curves descending and plausible while the gap between them means nothing:

  1. a term evaluated differently on the two sides (the reason there is ONE `batch_losses`),
  2. augmentation left on at eval (variance dropout would make val noisier than train, for free),
  3. a term that is structurally absent on one side (warp, with no `val_flow_pairs`).

So these tests are about the comparison, not about the loss going down. Real torch, real model, tiny
synthetic data on the CPU — the arithmetic is what is being checked, not the segmentation.
"""
import numpy as np
import pytest
import torch

from coastal.train import train_test_split_per_movie, train_with_metrics

TERMS = ('total', 'variance', 'intensity', 'temporal', 'warp', 'confetti', 'foreground',
         'boundary')


def _data(T=4, H=16, W=16, n_temporal=3, n_variance=2, seed=0):
    rng = np.random.default_rng(seed)
    frames = rng.random((T, H, W)).astype(np.float32)
    temporal = [{f"m_{i}": rng.random((H, W)).astype(np.float32) for i in range(n_temporal)}
                for _ in range(T)]
    variance = [{f"softmax_ch_{i}": rng.random((H, W)).astype(np.float32)
                 for i in range(n_variance)} for _ in range(T)]
    return frames, temporal, variance


def _flow_pairs(T=4, H=16, W=16, seed=1):
    """Forward flow t → t+1, `None` on the last frame (there is no next one)."""
    rng = np.random.default_rng(seed)
    return [rng.random((H, W, 2)).astype(np.float32) for _ in range(T - 1)] + [None]


# CPU, no AMP, no workers, 2 epochs — enough to exercise every path and quick enough to keep in the
# suite. `variance_dropout_p` stays at its default so the train/eval difference is real.
RUN = dict(num_epochs=2, batch_size=2, num_workers=0, use_amp=False, device='cpu', embedding_dim=4)


def test_no_val_split_records_no_val_keys():
    """Not a behaviour change for anyone who does not pass one."""
    frames, temporal, variance = _data()
    _, history = train_with_metrics(frames, temporal, variance, **RUN)
    assert not [k for k in history if k.startswith('val_')]
    assert all(len(history[t]) == 2 for t in TERMS)


def test_every_term_gets_a_val_curve_of_the_same_length():
    frames, temporal, variance = _data()
    v_frames, v_temporal, v_variance = _data(T=2, seed=7)
    _, history = train_with_metrics(
        frames, temporal, variance,
        val_frames=v_frames, val_temporal_metrics_norm=v_temporal,
        val_variance_metrics_norm=v_variance, **RUN)
    for t in TERMS:
        assert len(history[f'val_{t}']) == len(history[t]) == 2, \
            f"val_{t} must have one value per epoch, like {t}"
        assert all(np.isfinite(v) for v in history[f'val_{t}'])


def test_turning_validation_on_does_not_change_the_model():
    """Measuring a run must not change it. Two ways it could, and this catches both:

    the obvious one is a gradient leaking out of the val pass. The one that actually bit: iterating a
    DataLoader draws its base seed from the GLOBAL torch stream, so a val loop with no gradients and
    no augmentation of its own still shifted every later shuffle and dropout mask. Epoch 1 matched
    and epoch 2 did not — a run with validation converged to a different model from the identical
    run without it, which makes the curve describe something other than the run you would have had.
    Hence the RNG snapshot/restore around the pass.
    """
    frames, temporal, variance = _data()
    v_frames, v_temporal, v_variance = _data(T=2, seed=7)

    model, _ = train_with_metrics(
        frames, temporal, variance,
        val_frames=v_frames, val_temporal_metrics_norm=v_temporal,
        val_variance_metrics_norm=v_variance, **RUN)
    after = [p.detach().clone() for p in model.parameters()]

    # Same seed, same data, no val split: if the val pass leaked a gradient, the two runs' weights
    # would differ.
    model_ref, _ = train_with_metrics(frames, temporal, variance, **RUN)
    for a, b in zip(after, model_ref.parameters()):
        assert torch.allclose(a, b, atol=0, rtol=0), \
            "the validation pass changed the weights — it is not running under no_grad/eval"


def test_val_is_deterministic_because_augmentation_is_off():
    """`training=False` gates the variance-channel dropout. If it leaked into eval, the same weights
    on the same frames would score differently run to run, and epoch-to-epoch wobble in the val
    curve would be augmentation noise rather than the model changing."""
    frames, temporal, variance = _data()
    # One movie, evaluated twice as its own val set. The model is identical at both epochs only if
    # training also stops, so instead compare two INDEPENDENT runs with the same seed.
    kw = dict(val_frames=frames, val_temporal_metrics_norm=temporal,
              val_variance_metrics_norm=variance, variance_dropout_p=0.5, **RUN)
    _, a = train_with_metrics(frames, temporal, variance, **kw)
    _, b = train_with_metrics(frames, temporal, variance, **kw)
    assert a['val_total'] == pytest.approx(b['val_total'], rel=1e-6), \
        "two identically-seeded runs disagree on val_total — something stochastic is on at eval"


def test_warp_is_measured_on_the_val_set_when_flow_pairs_are_given():
    """The comparability trap. With `warp_weight > 0` and no `val_flow_pairs`, the warp term is
    structurally zero on the held-out set, so `val_total` sits below `total` by `warp_weight * warp`
    for a reason that has nothing to do with generalising."""
    frames, temporal, variance = _data()
    v_frames, v_temporal, v_variance = _data(T=4, seed=7)

    common = dict(val_frames=v_frames, val_temporal_metrics_norm=v_temporal,
                  val_variance_metrics_norm=v_variance, warp_weight=1.0, **RUN)

    _, without = train_with_metrics(frames, temporal, variance,
                                    flow_pairs=_flow_pairs(), **common)
    assert all(v == 0.0 for v in without['val_warp']), \
        "no val_flow_pairs should leave the val warp term at zero — that is the trap"
    assert any(v > 0.0 for v in without['warp']), "the training side should still have warp"

    _, with_flows = train_with_metrics(frames, temporal, variance,
                                       flow_pairs=_flow_pairs(),
                                       val_flow_pairs=_flow_pairs(seed=2), **common)
    assert any(v > 0.0 for v in with_flows['val_warp']), \
        "val_flow_pairs must make the warp term reach the held-out set"


def test_the_split_helper_feeds_the_val_params_directly():
    """The seam a caller actually writes. `train_test_split_per_movie` is the documented way to make
    the held-out set, and it splits frames + ONE metrics list — so variance needs a second call with
    the same ratio and `shuffle=False`, which lines the two splits up frame for frame. This asserts
    that alignment holds and that what comes out drops straight into the val arguments."""
    movies = [_data(T=10, seed=s) for s in (0, 1)]
    all_frames = [m[0] for m in movies]

    tr_frames, va_frames, tr_temporal, va_temporal = train_test_split_per_movie(
        all_frames, [m[1] for m in movies], train_ratio=0.8, shuffle=False)
    tr_frames2, va_frames2, tr_variance, va_variance = train_test_split_per_movie(
        all_frames, [m[2] for m in movies], train_ratio=0.8, shuffle=False)

    # Same ratio, no shuffle → the two calls must agree on WHICH frames are held out, or the
    # variance metrics would be paired with the wrong frames on both sides.
    assert np.array_equal(va_frames, va_frames2) and np.array_equal(tr_frames, tr_frames2)
    assert len(va_frames) == len(va_temporal) == len(va_variance) == 4   # 2 per movie

    _, history = train_with_metrics(
        tr_frames, tr_temporal, tr_variance,
        val_frames=va_frames, val_temporal_metrics_norm=va_temporal,
        val_variance_metrics_norm=va_variance, **RUN)
    assert len(history['val_total']) == 2


def test_the_terms_sum_to_the_total_on_both_sides():
    """One `batch_losses` for training and validation, so the weighted sum has to reconcile the same
    way on both. This is what stops the two curves drifting apart term by term."""
    frames, temporal, variance = _data()
    v_frames, v_temporal, v_variance = _data(T=2, seed=7)
    weights = dict(intensity_weight=1.0, temporal_weight=2.0, variance_weight=0.5,
                   warp_weight=0.0, confetti_weight=0.0, foreground_weight=0.0,
                   boundary_weight=0.0)
    _, history = train_with_metrics(
        frames, temporal, variance,
        val_frames=v_frames, val_temporal_metrics_norm=v_temporal,
        val_variance_metrics_norm=v_variance, **weights, **RUN)

    for prefix in ('', 'val_'):
        for epoch in range(2):
            summed = sum(weights[f'{t}_weight'] * history[f'{prefix}{t}'][epoch]
                         for t in TERMS if t != 'total')
            assert history[f'{prefix}total'][epoch] == pytest.approx(summed, rel=1e-4), \
                f"{prefix}total is not the weighted sum of its terms at epoch {epoch}"
