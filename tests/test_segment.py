"""Construction smoke test for the two-pass segmentation inference.

Pins the fix for the constructor crash where `TwoPassSegmentationInference` forwarded a
`prob_merge_weight=` kwarg that `LearnedAffinityInference` does not accept (it is `prob_weight`,
the region-growing relaxation). Before the fix, *constructing* the recommended inference path
raised TypeError; this test would have caught it. No model forward pass is exercised (model=None).
"""

import torch

from coastal.segment import TwoPassSegmentationInference, LearnedAffinityInference


def test_two_pass_inference_constructs():
    # A stand-in module: construction only stores model.to(device).eval(); no forward pass here.
    model = torch.nn.Identity()
    seg = TwoPassSegmentationInference(
        model=model,
        seed_size_large=32, affinity_threshold_large=0.2, embedding_blur_sigma_large=1.5,
        seed_size_small=8, affinity_threshold_small=0.8, embedding_blur_sigma_small=1.5,
        merge_affinity_threshold_large=0.90, merge_affinity_threshold_small=0.90,
        prob_weight_large=0.3, prob_weight_small=0.3,
        prob_threshold=0.3, min_component_size=10, device="cpu",
    )
    # Both passes are real LearnedAffinityInference instances with the relaxation wired through.
    assert isinstance(seg.pass1, LearnedAffinityInference)
    assert isinstance(seg.pass2, LearnedAffinityInference)
    assert seg.pass1.prob_weight == 0.3
    assert seg.pass2.prob_weight == 0.3
