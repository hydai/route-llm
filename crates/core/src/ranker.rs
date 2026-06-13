use crate::difficulty::sigmoid;
use crate::model::{Difficulty, ModelProfile, RankedModel, RoutingPreferences};

const K: f64 = 8.0;

/// Rank candidate models for a query difficulty under the given preferences. See SPEC §7.
pub fn rank(
    difficulty: &Difficulty,
    profiles: &[ModelProfile],
    prefs: &RoutingPreferences,
) -> Vec<RankedModel> {
    let d = difficulty.score;
    let lambda = prefs.cost_bias;

    // (profile, adequacy, score)
    let mut scored: Vec<(ModelProfile, f64, f64)> = profiles
        .iter()
        .map(|m| {
            let adequacy = sigmoid(K * (m.quality - d));
            let score = adequacy - lambda * m.cost;
            (m.clone(), adequacy, score)
        })
        .collect();

    // score desc, then quality desc, then cost asc, then id asc (deterministic).
    scored.sort_by(|a, b| {
        use std::cmp::Ordering::Equal;
        b.2.partial_cmp(&a.2)
            .unwrap_or(Equal)
            .then(b.0.quality.partial_cmp(&a.0.quality).unwrap_or(Equal))
            .then(a.0.cost.partial_cmp(&b.0.cost).unwrap_or(Equal))
            .then(a.0.id.cmp(&b.0.id))
    });

    let max_quality = profiles.iter().map(|m| m.quality).fold(f64::MIN, f64::max);
    let cheapest_adequate_cost = scored
        .iter()
        .filter(|(_, adequacy, _)| *adequacy >= 0.5)
        .map(|(m, _, _)| m.cost)
        .fold(f64::MAX, f64::min);

    scored
        .into_iter()
        .map(|(m, adequacy, score)| {
            let reason = make_reason(d, &m, adequacy, max_quality, cheapest_adequate_cost);
            RankedModel {
                id: m.id,
                score,
                reason,
            }
        })
        .collect()
}

fn make_reason(
    d: f64,
    m: &ModelProfile,
    adequacy: f64,
    max_quality: f64,
    cheapest_adequate_cost: f64,
) -> String {
    if adequacy < 0.5 {
        format!("能力可能不足以可靠處理此難度 (difficulty {:.2})", d)
    } else if d >= 0.6 && (m.quality - max_quality).abs() < f64::EPSILON {
        "高難度，最強模型最可靠".into()
    } else if d < 0.4 && (m.cost - cheapest_adequate_cost).abs() < f64::EPSILON {
        "低難度，便宜且足夠".into()
    } else {
        "在品質與成本間取得平衡".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diff(score: f64) -> Difficulty {
        Difficulty {
            score,
            signals: vec![],
        }
    }
    fn m(id: &str, quality: f64, cost: f64) -> ModelProfile {
        ModelProfile {
            id: id.into(),
            quality,
            cost,
        }
    }
    fn prefs(cost_bias: f64) -> RoutingPreferences {
        RoutingPreferences { cost_bias }
    }

    #[test]
    fn easy_query_prefers_cheaper_adequate_model() {
        let out = rank(
            &diff(0.1),
            &[m("strong", 0.9, 0.9), m("cheap", 0.5, 0.1)],
            &prefs(0.5),
        );
        assert_eq!(out[0].id, "cheap");
    }

    #[test]
    fn hard_query_prefers_stronger_model() {
        let out = rank(
            &diff(0.9),
            &[m("strong", 0.9, 0.9), m("cheap", 0.5, 0.1)],
            &prefs(0.5),
        );
        assert_eq!(out[0].id, "strong");
    }

    #[test]
    fn cost_bias_changes_ordering() {
        let models = [m("strong", 0.9, 0.9), m("cheap", 0.6, 0.1)];
        let quality_first = rank(&diff(0.5), &models, &prefs(0.0));
        let cost_first = rank(&diff(0.5), &models, &prefs(1.0));
        assert_eq!(quality_first[0].id, "strong");
        assert_eq!(cost_first[0].id, "cheap");
    }

    #[test]
    fn ties_break_deterministically_by_id() {
        let out = rank(
            &diff(0.5),
            &[m("b", 0.7, 0.3), m("a", 0.7, 0.3)],
            &prefs(0.5),
        );
        assert_eq!(out[0].id, "a");
    }

    #[test]
    fn inadequate_model_gets_capability_reason() {
        let out = rank(&diff(0.95), &[m("weak", 0.3, 0.1)], &prefs(0.5));
        assert!(out[0].reason.contains("能力可能不足"));
    }
}
