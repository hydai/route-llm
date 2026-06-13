use crate::model::Difficulty;

pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Estimate query difficulty. Pure and deterministic. See SPEC §5.
pub fn score(query: &str) -> Difficulty {
    let lower = query.to_lowercase();
    let mut sum = -1.0_f64; // base bias
    let mut signals: Vec<String> = Vec::new();

    // Length: estimate tokens ~= chars/4; +0.001 per token, capped at +1.2.
    let est_tokens = query.chars().count() as f64 / 4.0;
    let length_contrib = (est_tokens * 0.001).min(1.2);
    sum += length_contrib;
    if length_contrib > 0.3 {
        signals.push("long_form".into());
    }

    // Code.
    let code_markers = [
        "```", "fn ", "def ", "class ", "import ", "function", "select ",
    ];
    if code_markers.iter().any(|&m| lower.contains(m)) {
        sum += 1.0;
        signals.push("code".into());
    }

    // Math / LaTeX.
    let math_markers = ["\\frac", "\\sum", "\\int", "∑", "∫"];
    if query.matches('$').count() >= 2 || math_markers.iter().any(|&m| query.contains(m)) {
        sum += 0.8;
        signals.push("math".into());
    }

    // Reasoning keywords: +0.5 each, capped at +1.5.
    let reasoning = [
        "prove",
        "derive",
        "step by step",
        "analyze",
        "analyse",
        "design",
        "explain why",
        "optimize",
        "optimise",
        "compare",
        "證明",
        "推導",
        "逐步",
        "分析",
        "設計",
        "比較",
    ];
    let hits = reasoning.iter().filter(|&&k| lower.contains(k)).count();
    if hits > 0 {
        sum += (hits as f64 * 0.5).min(1.5);
        signals.push("reasoning".into());
    }

    // Multi-part constraints: numbered list >= 3 items, or >= 3 questions.
    let numbered = count_numbered_items(query);
    let questions = query.matches('?').count() + query.matches('？').count();
    if numbered >= 3 || questions >= 3 {
        sum += 0.6;
        signals.push("multi_constraint".into());
    }

    // Structured output request.
    let structured = ["json", "table", "schema", "yaml", "csv", "格式", "表格"];
    if structured.iter().any(|&s| lower.contains(s)) {
        sum += 0.4;
        signals.push("structured_output".into());
    }

    // Explanation request.
    let explain = ["explain", "說明", "為什麼", "how does", "怎麼", "what is"];
    if explain.iter().any(|&s| lower.contains(s)) {
        sum += 0.4;
        signals.push("explanation_request".into());
    }

    Difficulty {
        score: sigmoid(sum),
        signals,
    }
}

/// Count lines that begin with `<digit>.` or `<digit>)`.
fn count_numbered_items(query: &str) -> usize {
    query
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            let rest = trimmed.trim_start_matches(|c: char| c.is_ascii_digit());
            rest.len() < trimmed.len() && matches!(rest.chars().next(), Some('.') | Some(')'))
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_digit_numbered_items_are_counted() {
        let q = "1. first\n2. second\n10. tenth";
        assert_eq!(count_numbered_items(q), 3);
    }

    #[test]
    fn sigmoid_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn trivial_query_is_easy() {
        let d = score("hi");
        assert!(d.score < 0.4, "score was {}", d.score);
    }

    #[test]
    fn hard_query_scores_high_with_expected_signals() {
        let q = "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.";
        let d = score(q);
        assert!(d.score > 0.6 && d.score < 0.85, "score was {}", d.score);
        assert!(d.signals.contains(&"reasoning".to_string()));
        assert!(d.signals.contains(&"explanation_request".to_string()));
    }

    #[test]
    fn code_query_flags_code_signal() {
        let q = "Write a function in Rust: ```fn main() {}``` and optimize it.";
        let d = score(q);
        assert!(d.signals.contains(&"code".to_string()));
        assert!(d.score > 0.5, "score was {}", d.score);
    }
}
