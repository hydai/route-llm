//! Local-LLM difficulty labeling for the learned router via an OpenAI-compatible
//! chat API (e.g. LM Studio, Ollama's /v1, llama.cpp server, vLLM). Only this
//! module talks to the network, and only to a local server at request time of
//! the offline `label` step — never in the inference path.

/// Extract the first standalone 1–5 integer from model output. Prefers a
/// `rating: N` cue but falls back to the first 1–5 token. Returns None if no
/// valid 1–5 rating is present.
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
pub fn rating_to_difficulty(rating: u8) -> f64 {
    (rating.clamp(1, 5) as f64 - 1.0) / 4.0
}

use std::collections::HashMap;

use sha2::{Digest, Sha256};

/// Stable cache key for a (query, model) pair. Including the model means
/// switching models naturally invalidates old labels.
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
    pub fn get(&self, key: &str) -> Option<u8> {
        self.map.get(key).copied()
    }

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

    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_jsonl(&text),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
        }
        std::fs::write(path, self.to_jsonl()).map_err(|e| format!("write {path}: {e}"))
    }
}

use crate::dataset::{self, LabeledExample};

/// Where to reach the local model. Defaults per SPEC-v2.1 §5.
pub struct LabelConfig {
    pub url: String,
    pub model: String,
}

impl LabelConfig {
    pub fn from_env() -> Self {
        Self {
            url: std::env::var("ROUTE_LLM_LABEL_URL")
                .unwrap_or_else(|_| "http://localhost:1234/v1".to_string()),
            model: std::env::var("ROUTE_LLM_LABEL_MODEL")
                .unwrap_or_else(|_| "google/gemma-4-31b-qat".to_string()),
        }
    }
}

/// ★ Rubric prompt (owner-tunable, SPEC-v2.1 §5). Asks for `RATING: <n>` + reason.
pub fn build_prompt(query: &str) -> String {
    format!(
        "You are rating how hard a user query is for an LLM to answer *well*.\n\
         Use a 1-5 scale:\n\
         1 = trivial chat/greeting\n\
         2 = simple lookup or extraction\n\
         3 = moderate (some reasoning or code)\n\
         4 = hard, multi-step reasoning or non-trivial implementation\n\
         5 = expert: rigorous proof, deep analysis, or intricate system design\n\
         Reply with EXACTLY one line `RATING: <n>` (n in 1..5), then one short reason line.\n\n\
         Query: {query}\n"
    )
}

/// One OpenAI-compatible chat completion → assistant message text. NETWORK; not
/// unit-tested. Works with LM Studio (default), Ollama's /v1, llama.cpp, vLLM.
fn chat_complete(cfg: &LabelConfig, prompt: &str) -> Result<String, String> {
    let body = serde_json::json!({
        "model": cfg.model,
        "messages": [{ "role": "user", "content": prompt }],
        "temperature": 0,
        "stream": false
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;
    let resp = client
        .post(format!("{}/chat/completions", cfg.url))
        .json(&body)
        .send()
        .map_err(|e| {
            format!(
                "LLM request to {}/chat/completions failed: {e}\n\
                 Is your OpenAI-compatible server running (e.g. LM Studio at {}) with `{}` loaded?",
                cfg.url, cfg.url, cfg.model
            )
        })?;
    let v: serde_json::Value = resp
        .json()
        .map_err(|e| format!("LLM returned bad JSON: {e}"))?;
    v["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("LLM response missing choices[0].message.content: {v}"))
}

/// Label one query: cache hit → reuse; else call the LLM → parse.
/// `Ok(Some(r))` on success; `Ok(None)` if the output can't be parsed after a
/// retry (skip, don't poison labels); `Err(e)` only if the server stays
/// unreachable across several retries with backoff (a persistent outage → abort).
/// Transient blips (one dropped request during a long run) are retried, not fatal.
fn label_one(cfg: &LabelConfig, cache: &mut LabelCache, query: &str) -> Result<Option<u8>, String> {
    let key = cache_key(query, &cfg.model);
    if let Some(r) = cache.get(&key) {
        return Ok(Some(r));
    }
    let mut parse_fails = 0u32;
    let mut net_fails = 0u32;
    loop {
        match chat_complete(cfg, &build_prompt(query)) {
            Ok(out) => {
                if let Some(r) = parse_rating(&out) {
                    cache.insert(key.clone(), r);
                    return Ok(Some(r));
                }
                parse_fails += 1;
                if parse_fails >= 2 {
                    return Ok(None); // unparseable after a retry → skip this query
                }
            }
            Err(e) => {
                net_fails += 1;
                if net_fails >= 5 {
                    return Err(e); // persistently unreachable → abort the run
                }
                // Transient blip: back off (1s, 2s, 3s, 4s) and retry.
                std::thread::sleep(std::time::Duration::from_secs(net_fails as u64));
            }
        }
    }
}

/// `label` subcommand: read corpus.jsonl, label each query via the local
/// OpenAI-compatible LLM, write labeled.jsonl (+ persist the cache). The only
/// networked step.
pub fn run() {
    let cfg = LabelConfig::from_env();
    let corpus = dataset::load_corpus("data/corpus.jsonl")
        .expect("load data/corpus.jsonl (run `trainer synth` first)");
    let mut cache = LabelCache::load("data/label_cache.jsonl");
    let mut labeled: Vec<LabeledExample> = Vec::new();
    let (mut ok, mut skipped) = (0usize, 0usize);

    for (i, q) in corpus.iter().enumerate() {
        match label_one(&cfg, &mut cache, &q.query) {
            Ok(Some(r)) => {
                labeled.push(LabeledExample {
                    query: q.query.clone(),
                    difficulty: rating_to_difficulty(r),
                    category: q.category.clone(),
                });
                ok += 1;
            }
            Ok(None) => {
                eprintln!("skip (unparseable) [{i}]: {}", q.query);
                skipped += 1;
            }
            Err(e) => {
                let _ = cache.save("data/label_cache.jsonl");
                eprintln!("\nlabel aborted after {ok} labeled / {skipped} skipped: {e}");
                std::process::exit(1);
            }
        }
        if (i + 1) % 50 == 0 {
            eprintln!("labeled {}/{} (model={})", i + 1, corpus.len(), cfg.model);
            let _ = cache.save("data/label_cache.jsonl"); // periodic checkpoint
        }
    }

    cache
        .save("data/label_cache.jsonl")
        .expect("save label cache");
    dataset::save("data/labeled.jsonl", &labeled).expect("save data/labeled.jsonl");
    eprintln!("label: {ok} labeled, {skipped} skipped -> data/labeled.jsonl");
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

    #[test]
    fn config_defaults_match_spec() {
        // With env unset, defaults come from the spec.
        std::env::remove_var("ROUTE_LLM_LABEL_URL");
        std::env::remove_var("ROUTE_LLM_LABEL_MODEL");
        let cfg = LabelConfig::from_env();
        assert_eq!(cfg.url, "http://localhost:1234/v1");
        assert_eq!(cfg.model, "google/gemma-4-31b-qat");
    }

    #[test]
    fn prompt_includes_query_and_scale() {
        let p = build_prompt("reverse a linked list");
        assert!(p.contains("reverse a linked list"));
        assert!(p.contains("1-5") || p.contains("1–5"));
        assert!(p.to_lowercase().contains("rating"));
    }

    #[test]
    #[ignore = "requires a running local OpenAI-compatible LLM server (e.g. LM Studio) with the model loaded"]
    fn chat_complete_round_trip_smoke() {
        // Run manually: `cargo test -p route-llm-trainer -- --ignored chat_complete`
        let cfg = LabelConfig::from_env();
        let out = chat_complete(&cfg, &build_prompt("hi")).expect("chat completion call");
        assert!(
            parse_rating(&out).is_some(),
            "expected a 1-5 rating, got: {out}"
        );
    }
}
