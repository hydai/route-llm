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

use std::collections::HashMap;

use sha2::{Digest, Sha256};

/// Stable cache key for a (query, model) pair. Including the model means
/// switching models naturally invalidates old labels.
// Consumed by `label::run()` in v2.1 Task 5; allow until then.
#[allow(dead_code)]
pub fn cache_key(query: &str, model: &str) -> String {
    let mut h = Sha256::new();
    h.update(model.as_bytes());
    h.update([0u8]);
    h.update(query.as_bytes());
    format!("{:x}", h.finalize())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    key: String,
    rating: u8,
}

/// In-memory label cache, persisted as jsonl (one CacheEntry per line).
#[derive(Debug, Default)]
pub struct LabelCache {
    map: HashMap<String, u8>,
}

impl LabelCache {
    // get/insert consumed by `label::run()` in v2.1 Task 5; allow until then.
    #[allow(dead_code)]
    pub fn get(&self, key: &str) -> Option<u8> {
        self.map.get(key).copied()
    }

    #[allow(dead_code)]
    pub fn insert(&mut self, key: String, rating: u8) {
        self.map.insert(key, rating);
    }

    pub fn from_jsonl(text: &str) -> Self {
        let mut map = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(e) = serde_json::from_str::<CacheEntry>(line) {
                map.insert(e.key, e.rating);
            }
        }
        Self { map }
    }

    /// Deterministic (key-sorted) jsonl so the committed cache file is stable.
    pub fn to_jsonl(&self) -> String {
        let mut entries: Vec<(&String, &u8)> = self.map.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut s = entries
            .iter()
            .map(|(k, r)| {
                serde_json::to_string(&CacheEntry {
                    key: (*k).clone(),
                    rating: **r,
                })
                .unwrap()
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !s.is_empty() {
            s.push('\n');
        }
        s
    }

    // Consumed by `label::run()` in v2.1 Task 5; allow until then.
    #[allow(dead_code)]
    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_jsonl(&text),
            Err(_) => Self::default(),
        }
    }

    // Consumed by `label::run()` in v2.1 Task 5; allow until then.
    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> Result<(), String> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
        }
        std::fs::write(path, self.to_jsonl()).map_err(|e| format!("write {path}: {e}"))
    }
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

    #[test]
    fn cache_key_is_deterministic_and_model_sensitive() {
        let a = cache_key("hi", "m1");
        assert_eq!(a, cache_key("hi", "m1"));
        assert_ne!(a, cache_key("hi", "m2"));
        assert_ne!(a, cache_key("bye", "m1"));
    }

    #[test]
    fn cache_round_trips_and_looks_up() {
        let mut c = LabelCache::default();
        c.insert("k1".into(), 3);
        c.insert("k2".into(), 5);
        let restored = LabelCache::from_jsonl(&c.to_jsonl());
        assert_eq!(restored.get("k1"), Some(3));
        assert_eq!(restored.get("k2"), Some(5));
        assert_eq!(restored.get("missing"), None);
    }
}
