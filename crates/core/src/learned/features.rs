//! Pure-Rust feature extraction for the learned router. Shared by the trainer
//! and `LearnedRouter` — identical features at train and inference time.

/// Bump whenever the feature set changes; `weights.rs` carries the same value.
pub const SCHEMA_VERSION: u32 = 1;

/// Number of char-trigram hash bins appended after the explicit features.
pub const NGRAM_BINS: usize = 16;

/// Names of the explicit (non-ngram) features, in vector order.
pub const BASE_FEATURE_NAMES: &[&str] = &[
    "length_contrib",
    "has_code",
    "has_math",
    "reasoning_hits",
    "multi_constraint",
    "structured_output",
    "explanation_request",
    "char_count",
    "word_count",
    "avg_word_len",
    "sentence_count",
    "question_ratio",
    "uppercase_ratio",
    "digit_ratio",
    "lexical_diversity",
    "cjk_ratio",
    "code_fence_count",
    "url_present",
    "non_ascii_ratio",
];

/// Total feature dimension = explicit features + ngram bins.
pub fn feature_count() -> usize {
    BASE_FEATURE_NAMES.len() + NGRAM_BINS
}

/// Human-readable name for feature index `i` (for explainability/signals).
pub fn feature_name(i: usize) -> String {
    if i < BASE_FEATURE_NAMES.len() {
        BASE_FEATURE_NAMES[i].to_string()
    } else {
        format!("ngram_{}", i - BASE_FEATURE_NAMES.len())
    }
}

/// Extract a fixed-length, stable-order feature vector. Pure & deterministic.
pub fn features(query: &str) -> Vec<f64> {
    let lower = query.to_lowercase();
    let chars: Vec<char> = query.chars().collect();
    let char_count = chars.len() as f64;
    let words: Vec<&str> = query.split_whitespace().collect();
    let word_count = words.len() as f64;

    let mut v = vec![0.0_f64; feature_count()];

    // 0 length_contrib (same shape as v1: ~tokens * 0.001, capped 1.2)
    v[0] = ((char_count / 4.0) * 0.001).min(1.2);
    // 1 has_code
    let code_markers = [
        "```", "fn ", "def ", "class ", "import ", "function", "select ",
    ];
    v[1] = bool_f(code_markers.iter().any(|m| lower.contains(m)));
    // 2 has_math
    let math_markers = ["\\frac", "\\sum", "\\int", "∑", "∫"];
    v[2] =
        bool_f(query.matches('$').count() >= 2 || math_markers.iter().any(|m| query.contains(m)));
    // 3 reasoning_hits (capped at 3)
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
    let hits = reasoning
        .iter()
        .filter(|k| lower.contains(&k.to_lowercase()))
        .count();
    v[3] = (hits.min(3)) as f64;
    // 4 multi_constraint
    let numbered = count_numbered_items(query);
    let questions = query.matches('?').count() + query.matches('？').count();
    v[4] = bool_f(numbered >= 3 || questions >= 3);
    // 5 structured_output
    let structured = ["json", "table", "schema", "yaml", "csv", "格式", "表格"];
    v[5] = bool_f(structured.iter().any(|s| lower.contains(s)));
    // 6 explanation_request
    let explain = ["explain", "說明", "為什麼", "how does", "怎麼", "what is"];
    v[6] = bool_f(explain.iter().any(|s| lower.contains(&s.to_lowercase())));
    // 7 char_count
    v[7] = char_count;
    // 8 word_count
    v[8] = word_count;
    // 9 avg_word_len
    v[9] = if word_count > 0.0 {
        char_count / word_count
    } else {
        0.0
    };
    // 10 sentence_count
    let sentence_count = (query.matches(['.', '!', '?']).count()
        + query.matches(['。', '！', '？']).count())
    .max(1) as f64;
    v[10] = sentence_count;
    // 11 question_ratio
    v[11] = questions as f64 / sentence_count;
    // 12 uppercase_ratio (over ascii letters)
    let letters = chars.iter().filter(|c| c.is_ascii_alphabetic()).count() as f64;
    let uppers = chars.iter().filter(|c| c.is_ascii_uppercase()).count() as f64;
    v[12] = if letters > 0.0 { uppers / letters } else { 0.0 };
    // 13 digit_ratio
    v[13] = if char_count > 0.0 {
        chars.iter().filter(|c| c.is_ascii_digit()).count() as f64 / char_count
    } else {
        0.0
    };
    // 14 lexical_diversity
    v[14] = if word_count > 0.0 {
        let mut uniq: Vec<&str> = words.clone();
        uniq.sort_unstable();
        uniq.dedup();
        uniq.len() as f64 / word_count
    } else {
        0.0
    };
    // 15 cjk_ratio
    v[15] = if char_count > 0.0 {
        chars.iter().filter(|c| is_cjk(**c)).count() as f64 / char_count
    } else {
        0.0
    };
    // 16 code_fence_count
    v[16] = query.matches("```").count() as f64;
    // 17 url_present
    v[17] = bool_f(lower.contains("http://") || lower.contains("https://"));
    // 18 non_ascii_ratio
    v[18] = if char_count > 0.0 {
        chars.iter().filter(|c| !c.is_ascii()).count() as f64 / char_count
    } else {
        0.0
    };

    // 19.. char-trigram hash bins (normalized by trigram count)
    let base = BASE_FEATURE_NAMES.len();
    if chars.len() >= 3 {
        let mut total = 0.0_f64;
        for w in chars.windows(3) {
            let h = fnv3(w);
            v[base + (h % NGRAM_BINS)] += 1.0;
            total += 1.0;
        }
        if total > 0.0 {
            for b in 0..NGRAM_BINS {
                v[base + b] /= total;
            }
        }
    }

    v
}

fn bool_f(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x3040..=0x30FF)
}

/// FNV-1a over three chars → bin index source.
fn fnv3(w: &[char]) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for c in w {
        h ^= *c as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h as usize
}

/// Count lines beginning with one or more digits followed by `.` or `)`.
/// Matches v1's multi-digit logic (e.g. `10. item` is counted).
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
    fn vector_length_matches_feature_count() {
        assert_eq!(features("hello").len(), feature_count());
        assert_eq!(features("").len(), feature_count());
    }

    #[test]
    fn deterministic() {
        let q = "Explain step by step why this works.";
        assert_eq!(features(q), features(q));
    }

    #[test]
    fn code_query_flags_code_features() {
        let f = features("Write a function: ```fn main() {}```");
        assert!(f[1] > 0.0, "has_code"); // index 1
        assert!(f[16] > 0.0, "code_fence_count"); // index 16
    }

    #[test]
    fn reasoning_hits_capped_at_three() {
        let q = "prove derive analyze compare optimize design step by step";
        assert!(f_at(q, "reasoning_hits") <= 3.0);
        assert!(f_at(q, "reasoning_hits") >= 1.0);
    }

    #[test]
    fn empty_query_is_all_finite() {
        assert!(features("").iter().all(|v| v.is_finite()));
    }

    fn f_at(q: &str, name: &str) -> f64 {
        let i = BASE_FEATURE_NAMES.iter().position(|n| *n == name).unwrap();
        features(q)[i]
    }

    #[test]
    fn multi_digit_numbered_list_sets_multi_constraint() {
        // Three items with two-digit markers: 10, 11, 12
        let q = "10. first item\n11. second item\n12. third item";
        assert_eq!(
            f_at(q, "multi_constraint"),
            1.0,
            "multi_constraint must be 1.0 for a list with multi-digit markers"
        );
    }

    #[test]
    fn single_digit_numbered_list_sets_multi_constraint() {
        // Three items with single-digit markers should also work
        let q = "1. first\n2. second\n3. third";
        assert_eq!(
            f_at(q, "multi_constraint"),
            1.0,
            "multi_constraint must be 1.0 for a list with single-digit markers"
        );
    }
}
