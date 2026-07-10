"""Coastal: Cell segmentation via optical flow metrics and learned embeddings."""

__version__ = "0.1.0"

from coastal.model import UNetWithEmbeddings
from coastal.loss import IntensityLoss, TemporalMetricsLoss, VarianceMetricsLoss, WarpConsistencyLoss
from coastal.train import (
    train_with_metrics,
    save_model,
    load_model,
    train_test_split,
    train_test_split_per_movie,
    prepare_data_for_unet_batch,
    prepare_data_for_unet_batch_4d,
    extract_sequences_from_volume,
    TemporalDatasetWithAugmentation,
)
from coastal.segment import LearnedAffinityInference, TwoPassSegmentationInference, Inference3D
from coastal.utils import match_masks_3d, intersection_over_union, filter_small_cells
from coastal.viz import visualize_frame_segmentation, plot_rgb_with_segmentation
from coastal.flow import prepare_data_for_unet, compute_variance_metrics, VarianceMetricsConfig, normalize_and_project, extract_dense_flow_pairs
from coastal.data import prepare_training_data, validate_training_data
from coastal.optimize import (
    optimize_segmentation_cma,
    score_segmentation,
    optimize_tracking_cma,
    score_tracking_scalar,
    TRACKING_PARAM_BOUNDS,
)
from coastal.morphology import (
    labels_to_polygons,
    fit_boundary_hmm,
    assign_boundary_states,
    assign_boundary_hmm_features,
    extract_boundary_features,
    extract_shape_features,
    extract_cell_morphology,
    median_filter_states,
    enforce_min_run_length,
    SHAPE_FEATURE_NAMES,
    hmm_feature_dim,
)
from coastal.track import (
    Track,
    compute_3d_centroids,
    extract_cell_colors,
    extract_cell_intensities,
)
from coastal.abm import (
    BreadcrumbField,
    MotilityState,
    CellAgent,
    ABMTracker,
    track_abm,
    compute_cell_flows,
    compute_cell_flow_features,
    smooth_cell_flows,
    blend_flows,
    track_sequence,
    extract_cell_embeddings,
    stitch_tracklets,
    score_tracking,
)

__all__ = [
    "UNetWithEmbeddings",
    "IntensityLoss",
    "TemporalMetricsLoss",
    "VarianceMetricsLoss",
    "WarpConsistencyLoss",
    "compute_variance_metrics",
    "normalize_and_project",
    "extract_dense_flow_pairs",
    "train_with_metrics",
    "save_model",
    "load_model",
    "train_test_split",
    "train_test_split_per_movie",
    "prepare_data_for_unet_batch",
    "prepare_data_for_unet_batch_4d",
    "extract_sequences_from_volume",
    "prepare_data_for_unet",
    "TemporalDatasetWithAugmentation",
    "LearnedAffinityInference",
    "TwoPassSegmentationInference",
    "Inference3D",
    "match_masks_3d",
    "intersection_over_union",
    "filter_small_cells",
    "visualize_frame_segmentation",
    "plot_rgb_with_segmentation",
    "prepare_training_data",
    "validate_training_data",
    "optimize_segmentation_cma",
    "score_segmentation",
    "optimize_tracking_cma",
    "score_tracking_scalar",
    "TRACKING_PARAM_BOUNDS",
    # morphology
    "labels_to_polygons",
    "fit_boundary_hmm",
    "assign_boundary_states",
    "assign_boundary_hmm_features",
    "extract_boundary_features",
    "extract_shape_features",
    "extract_cell_morphology",
    "median_filter_states",
    "enforce_min_run_length",
    "SHAPE_FEATURE_NAMES",
    "hmm_feature_dim",
    # tracking — data structures + feature extraction
    "Track",
    "compute_3d_centroids",
    "extract_cell_colors",
    "extract_cell_intensities",
    # ABM / tracking inference
    "BreadcrumbField",
    "MotilityState",
    "CellAgent",
    "ABMTracker",
    "track_abm",
    "compute_cell_flows",
    "compute_cell_flow_features",
    "smooth_cell_flows",
    "blend_flows",
    "track_sequence",
    "extract_cell_embeddings",
    "stitch_tracklets",
    "score_tracking",
]
