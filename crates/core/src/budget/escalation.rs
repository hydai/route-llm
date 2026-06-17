//! Deterministic decision layer: risk floors, latest-info → tool, confidence,
//! two-estimator disagreement, and policy modes. Zero network. See SPEC-v3 §6.

use crate::budget::dims::{contributions, N_DIMS};
use crate::budget::level::{bounds, Level};
use crate::model::Policy;

/// Output of the decision layer.
pub struct Decision {
    pub level: Level,
    pub confidence: f64,
    pub needs_tool: bool,
    pub tool_type: Option<String>,
    pub requires_verifier: bool,
    pub fallback_policy: String,
    pub reason_codes: Vec<String>,
}

const HIGH_RISK: &[&str] = &[
    "legal",
    "lawsuit",
    "contract",
    "法律",
    "合約",
    "medical",
    "diagnosis",
    "health",
    "醫療",
    "病歷",
    "invest",
    "stock",
    "financial",
    "金融",
    "投資",
    "股票",
    "security",
    "vulnerability",
    "exploit",
    "資安",
    "漏洞",
    "production",
    "deploy",
    "生產環境",
    "部署",
    "pii",
    "personal data",
    "privacy",
    "個資",
    "隱私",
];

const LATEST_INFO: &[&str] = &[
    "today",
    "latest",
    "current",
    "right now",
    "this week",
    "今天",
    "最新",
    "現在",
    "目前",
    "exchange rate",
    "匯率",
    "stock price",
    "股價",
    "news",
    "新聞",
    "ceo of",
    "who is the current",
];

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Confidence = ½·boundary_margin + ½·estimator_agreement (SPEC-v3 §6.2).
pub fn confidence(score: f64, budget_level: Level, learned_level: Level) -> f64 {
    let (lo, hi) = bounds(budget_level);
    let half = ((hi - lo) / 2.0).max(1e-9);
    let margin = ((score - lo).min(hi - score) / half).clamp(0.0, 1.0);
    let agreement = 1.0 - (budget_level.index() as f64 - learned_level.index() as f64).abs() / 4.0;
    (0.5 * margin + 0.5 * agreement).clamp(0.0, 1.0)
}

/// Top-2 dimensions by contribution → reason codes (canonical order; SPEC-v3 §6/§7).
fn dimension_reason_codes(dims: &[f64; N_DIMS]) -> Vec<String> {
    const CODE: [&str; N_DIMS] = [
        "multi_step_reasoning",
        "needs_validation",
        "constraint_dense",
        "context_heavy",
        "ambiguous",
        "high_error_cost",
    ];
    let contrib = contributions(dims);
    let mut idx: Vec<usize> = (0..N_DIMS).collect();
    idx.sort_by(|&a, &b| {
        contrib[b]
            .partial_cmp(&contrib[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b)) // stable tie-break by index
    });
    idx.into_iter()
        .filter(|&i| contrib[i] > 0.0)
        .take(2)
        .map(|i| CODE[i].to_string())
        .collect()
}

/// Apply the full decision layer. `query_lower` is the lowercased query.
pub fn decide(
    policy: Policy,
    query_lower: &str,
    dims: &[f64; N_DIMS],
    score: f64,
    base_level: Level,
    learned_level: Level,
) -> Decision {
    let mut level = base_level;
    let mut reason_codes = dimension_reason_codes(dims);
    let conf = confidence(score, base_level, learned_level);

    // (2) high-risk floors
    let high_risk = contains_any(query_lower, HIGH_RISK);
    if high_risk {
        level = level.max_with(Level::R3);
        reason_codes.push("high_risk_domain".into());
        if dims[0] >= 3.0 {
            // deep reasoning in a risk domain → expert tier
            level = level.max_with(Level::R4);
        }
    }

    // (3) latest info → tool, no level change
    let needs_tool = contains_any(query_lower, LATEST_INFO);
    let tool_type = if needs_tool {
        reason_codes.push("needs_latest_info".into());
        Some("web_search".to_string())
    } else {
        None
    };

    let dlevel = (base_level.index() as i32 - learned_level.index() as i32).abs();
    let mut requires_verifier = false;
    let mut fallback_policy = "none".to_string();

    // (4)+(5) policy-specific
    match policy {
        Policy::Balanced => {
            if conf < 0.7 {
                level = level.shift(1);
                reason_codes.push("low_confidence".into());
            }
        }
        Policy::Strict => {
            if conf < 0.85 {
                level = level.shift(1);
                reason_codes.push("low_confidence".into());
            }
            if dlevel >= 1 {
                level = level.shift(1);
                reason_codes.push("estimator_disagreement".into());
            }
            if high_risk {
                level = level.max_with(Level::R4);
            }
            if base_level.index() >= Level::R3.index() {
                requires_verifier = true;
            }
        }
        Policy::Cheap => {
            if !high_risk && dims[5] <= 1.0 {
                level = level.shift(-1);
            }
            requires_verifier = true;
            fallback_policy = "upgrade_if_verifier_fails".to_string();
        }
    }

    // (6) strong disagreement → max + verifier (all policies)
    if dlevel >= 2 {
        level = level.max_with(Level::from_index(
            base_level.index().max(learned_level.index()),
        ));
        requires_verifier = true;
        if !reason_codes.iter().any(|c| c == "estimator_disagreement") {
            reason_codes.push("estimator_disagreement".into());
        }
    }

    Decision {
        level,
        confidence: conf,
        needs_tool,
        tool_type,
        requires_verifier,
        fallback_policy,
        reason_codes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: [f64; N_DIMS] = [0.0; N_DIMS];

    #[test]
    fn high_risk_floors_to_at_least_r3() {
        let d = decide(
            Policy::Balanced,
            "review this legal contract",
            &ZERO,
            1.0,
            Level::R0,
            Level::R0,
        );
        assert!(d.level.index() >= Level::R3.index());
        assert!(d.reason_codes.iter().any(|c| c == "high_risk_domain"));
    }

    #[test]
    fn latest_info_sets_tool_without_raising_level() {
        let d = decide(
            Policy::Balanced,
            "what is the latest exchange rate",
            &ZERO,
            10.0,
            Level::R2,
            Level::R2,
        );
        assert!(d.needs_tool);
        assert_eq!(d.tool_type.as_deref(), Some("web_search"));
        assert_eq!(d.level, Level::R2, "latest-info must not raise the level");
    }

    #[test]
    fn strong_disagreement_requires_verifier_and_takes_max() {
        let d = decide(Policy::Balanced, "x", &ZERO, 2.0, Level::R0, Level::R4);
        assert!(d.requires_verifier);
        assert_eq!(d.level, Level::R4);
        assert!(d.reason_codes.iter().any(|c| c == "estimator_disagreement"));
    }

    #[test]
    fn cheap_downgrades_low_risk_and_flags_fallback() {
        let d = decide(
            Policy::Cheap,
            "summarize this paragraph",
            &ZERO,
            9.0,
            Level::R2,
            Level::R2,
        );
        assert_eq!(d.level, Level::R1, "cheap downgrades a low-risk task");
        assert!(d.requires_verifier);
        assert_eq!(d.fallback_policy, "upgrade_if_verifier_fails");
    }

    #[test]
    fn strict_adds_verifier_for_hard_base() {
        let d = decide(
            Policy::Strict,
            "design a system",
            &ZERO,
            13.0,
            Level::R3,
            Level::R3,
        );
        assert!(d.requires_verifier);
    }

    #[test]
    fn reason_codes_pick_top_contributors() {
        let mut dims = ZERO;
        dims[0] = 4.0; // reasoning_depth
        dims[5] = 4.0; // error_cost
        let d = decide(Policy::Balanced, "x", &dims, 10.0, Level::R2, Level::R2);
        assert!(d.reason_codes.iter().any(|c| c == "multi_step_reasoning"));
        assert!(d.reason_codes.iter().any(|c| c == "high_error_cost"));
    }
}
