//! The six reasoning-budget dimensions: canonical order, scales, weights, and
//! scoring via six per-dimension LinearModels that share v2's feature vector.
//! Pure & deterministic. See SPEC-v3 §4.

use crate::learned::model::LinearModel;
use crate::model::DimensionScores;

/// Bump when the dimension label schema or order changes; `weights.rs` matches.
pub const BUDGET_SCHEMA_VERSION: u32 = 1;

/// Number of dimensions.
pub const N_DIMS: usize = 6;

/// Canonical dimension order. Load-bearing: heads, scales, and weights index by it.
pub const DIM_NAMES: [&str; N_DIMS] = [
    "reasoning_depth",
    "verification_difficulty",
    "constraint_density",
    "context_integration",
    "ambiguity",
    "error_cost",
];

/// Max integer each dimension's human rubric uses (SPEC-v3 §4.1).
pub const DIM_SCALES: [f64; N_DIMS] = [4.0, 4.0, 4.0, 4.0, 3.0, 4.0];

/// RBC weights (SPEC-v3 §4.1 / §6).
pub const DIM_WEIGHTS: [f64; N_DIMS] = [1.4, 1.1, 1.0, 1.0, 0.8, 1.2];

/// Theoretical maximum budget_score = Σ weight_i · scale_i = 25.2 (SPEC-v3 §4.1).
pub const MAX_BUDGET: f64 = 25.2;

/// Score all six dimensions for a query: each head outputs p∈(0,1) (logistic),
/// rescaled to its integer scale.
pub fn score_dims(models: &[LinearModel], query: &str) -> [f64; N_DIMS] {
    debug_assert_eq!(models.len(), N_DIMS, "expected six dimension heads");
    let mut out = [0.0; N_DIMS];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = models[i].difficulty(query).score * DIM_SCALES[i];
    }
    out
}

/// Weighted budget score from dimension values (SPEC-v3 §4.1 formula).
pub fn budget_score(dims: &[f64; N_DIMS]) -> f64 {
    (0..N_DIMS).map(|i| DIM_WEIGHTS[i] * dims[i]).sum()
}

/// Per-dimension contribution weight_i·dim_i (for reason_codes).
pub fn contributions(dims: &[f64; N_DIMS]) -> [f64; N_DIMS] {
    let mut c = [0.0; N_DIMS];
    for (i, slot) in c.iter_mut().enumerate() {
        *slot = DIM_WEIGHTS[i] * dims[i];
    }
    c
}

/// Convert the fixed array to the named output struct (canonical order).
pub fn to_scores(dims: &[f64; N_DIMS]) -> DimensionScores {
    DimensionScores {
        reasoning_depth: dims[0],
        verification_difficulty: dims[1],
        constraint_density: dims[2],
        context_integration: dims[3],
        ambiguity: dims[4],
        error_cost: dims[5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned::features::{feature_count, SCHEMA_VERSION};

    fn head_with(weight_idx: usize, w: f64) -> LinearModel {
        let n = feature_count();
        let mut weights = vec![0.0; n];
        weights[weight_idx] = w;
        LinearModel {
            schema_version: SCHEMA_VERSION,
            weights,
            bias: 0.0,
            means: vec![0.0; n],
            stds: vec![1.0; n],
        }
    }

    #[test]
    fn budget_score_matches_weighted_sum() {
        let dims = DIM_SCALES; // all-max dimensions
        assert!((budget_score(&dims) - MAX_BUDGET).abs() < 1e-9);
    }

    #[test]
    fn score_dims_stays_in_each_scale() {
        let models: Vec<LinearModel> = (0..N_DIMS).map(|_| head_with(0, 0.0)).collect();
        let d = score_dims(&models, "hello world");
        for i in 0..N_DIMS {
            assert!(d[i] >= 0.0 && d[i] <= DIM_SCALES[i], "dim {i} = {}", d[i]);
        }
    }

    #[test]
    fn positive_reasoning_head_raises_that_dimension() {
        // feature index 3 == reasoning_hits (see learned::features BASE_FEATURE_NAMES).
        let mut models: Vec<LinearModel> = (0..N_DIMS).map(|_| head_with(0, 0.0)).collect();
        models[0] = head_with(3, 5.0);
        let hard = score_dims(&models, "prove and derive step by step; analyze")[0];
        let easy = score_dims(&models, "hello")[0];
        assert!(hard > easy, "reasoning dim hard {hard} vs easy {easy}");
    }
}
