use crate::dataset::{CorpusQuery, LabeledExample};
use std::collections::HashMap;

/// Queries where claude and codex assigned a different difficulty — the gold pool.
/// Output is BLIND (`CorpusQuery`: query + category only, no difficulty/rating) so a
/// human can re-judge without anchoring. Ordered by `codex`'s order (= corpus order);
/// `category` is taken from codex. Queries absent from `claude` are skipped.
pub fn disagreements(claude: &[LabeledExample], codex: &[LabeledExample]) -> Vec<CorpusQuery> {
    let claude_by_q: HashMap<&str, f64> =
        claude.iter().map(|e| (e.query.as_str(), e.difficulty)).collect();
    codex
        .iter()
        .filter(|e| match claude_by_q.get(e.query.as_str()) {
            Some(&cd) => (cd - e.difficulty).abs() > 1e-9,
            None => false, // not in claude → can't compare → skip
        })
        .map(|e| CorpusQuery {
            query: e.query.clone(),
            category: e.category.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(q: &str, d: f64, c: &str) -> LabeledExample {
        LabeledExample {
            query: q.into(),
            difficulty: d,
            category: c.into(),
        }
    }

    #[test]
    fn disagreements_selects_only_differing_queries() {
        let claude = vec![ex("a", 0.25, "code"), ex("b", 0.5, "math"), ex("c", 1.0, "math")];
        let codex = vec![ex("a", 0.25, "code"), ex("b", 0.75, "math"), ex("c", 0.75, "math")];
        let pool = disagreements(&claude, &codex);
        let qs: Vec<&str> = pool.iter().map(|p| p.query.as_str()).collect();
        assert_eq!(qs, vec!["b", "c"], "only b and c differ");
    }

    #[test]
    fn disagreements_keeps_category_from_codex() {
        let claude = vec![ex("b", 0.5, "math")];
        let codex = vec![ex("b", 0.75, "math")];
        let pool = disagreements(&claude, &codex);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].category, "math");
        // CorpusQuery has no `difficulty` field — blindness is type-enforced.
    }

    #[test]
    fn disagreements_preserve_codex_order() {
        let claude = vec![ex("x", 0.0, "chat"), ex("y", 0.0, "chat"), ex("z", 0.0, "chat")];
        let codex = vec![ex("z", 0.25, "chat"), ex("y", 0.25, "chat"), ex("x", 0.25, "chat")];
        let pool = disagreements(&claude, &codex);
        let qs: Vec<&str> = pool.iter().map(|p| p.query.as_str()).collect();
        assert_eq!(qs, vec!["z", "y", "x"], "must follow codex order");
    }

    #[test]
    fn disagreements_skips_queries_missing_in_claude() {
        let claude = vec![ex("a", 0.25, "code")];
        let codex = vec![ex("a", 0.25, "code"), ex("orphan", 0.5, "math")];
        let pool = disagreements(&claude, &codex);
        assert!(pool.is_empty(), "a agrees; orphan not in claude → skipped");
    }
}
