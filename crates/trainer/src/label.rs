//! Local-LLM (Ollama) difficulty labeling for the learned router.
//! Only this module talks to the network, and only to a local Ollama at
//! request time of the offline `label` step — never in the inference path.

/// Extract the first standalone 1–5 integer from model output. Prefers a
/// `rating: N` cue but falls back to the first 1–5 token. Returns None if no
/// valid 1–5 rating is present.
// Consumed by `label::run()` in v2.1 Task 5; allow until then.
#[allow(dead_code)]
pub fn parse_rating(output: &str) -> Option<u8> {
    let lower = output.to_lowercase();
    // Prefer an explicit "rating: N" / "rating N".
    if let Some(idx) = lower.find("rating") {
        if let Some(n) = first_1_to_5(&output[idx..]) {
            return Some(n);
        }
    }
    first_1_to_5(output)
}

fn first_1_to_5(s: &str) -> Option<u8> {
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            // Only accept a single-digit 1..=5 not glued to another digit.
            let next_is_digit = chars.peek().map(|c| c.is_ascii_digit()).unwrap_or(false);
            if !next_is_digit {
                if let Some(n) = c.to_digit(10) {
                    if (1..=5).contains(&n) {
                        return Some(n as u8);
                    }
                }
            }
        }
    }
    None
}

/// Map a 1–5 rating to difficulty in {0.0, 0.25, 0.5, 0.75, 1.0}.
// Consumed by `label::run()` in v2.1 Task 5; allow until then.
#[allow(dead_code)]
pub fn rating_to_difficulty(rating: u8) -> f64 {
    (rating.clamp(1, 5) as f64 - 1.0) / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rating_extracts_1_to_5() {
        assert_eq!(parse_rating("RATING: 3\nbecause ..."), Some(3));
        assert_eq!(parse_rating("I'd say rating: 5 (very hard)"), Some(5));
        assert_eq!(parse_rating("1"), Some(1));
        assert_eq!(parse_rating("The difficulty is 4 out of 5."), Some(4));
    }

    #[test]
    fn parse_rating_rejects_out_of_range_or_missing() {
        assert_eq!(parse_rating("RATING: 9"), None);
        assert_eq!(parse_rating("no number here"), None);
        assert_eq!(parse_rating("0"), None);
    }

    #[test]
    fn rating_maps_to_unit_interval() {
        assert_eq!(rating_to_difficulty(1), 0.0);
        assert_eq!(rating_to_difficulty(3), 0.5);
        assert_eq!(rating_to_difficulty(5), 1.0);
    }
}
