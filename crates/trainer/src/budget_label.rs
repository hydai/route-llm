//! Frontier-LLM labeling of the six reasoning-budget dimensions (offline only).
//! Reuses `label`'s OpenAI-compatible client. Writes `data/budget.<model>.jsonl`.
//! Never used at inference. See SPEC-v3 §4.3.

use crate::dataset::{self, dim_value, DimScores, DimsExample};
use crate::label::{self, LabelConfig};
use route_llm_core::learned::model::LinearModel;
use std::collections::HashMap;

/// Per-dimension max integer (matches `budget::dims::DIM_SCALES`).
const SCALES: [u8; 6] = [4, 4, 4, 4, 3, 4];

/// Rubric prompt: asks for one `DIMS:` line with six integers in canonical order.
pub fn build_dims_prompt(query: &str) -> String {
    format!(
        "You rate how much reasoning a user query needs, on SIX independent axes.\n\
         Give an integer for each (ranges in brackets):\n\
         1. reasoning_depth [0-4]: layers of reasoning / problem decomposition\n\
         2. verification_difficulty [0-4]: how hard the answer is to check\n\
         3. constraint_density [0-4]: how many constraints must hold at once\n\
         4. context_integration [0-4]: how much context must be integrated\n\
         5. ambiguity [0-3]: how many reasonable interpretations\n\
         6. error_cost [0-4]: cost of being wrong\n\
         Reply with EXACTLY one line:\n\
         DIMS: <reasoning_depth> <verification_difficulty> <constraint_density> \
         <context_integration> <ambiguity> <error_cost>\n\
         then one short reason line.\n\n\
         Query: {query}\n"
    )
}

/// Parse six integers from a `DIMS:` line (preferred) or the first six ints found.
/// Each is clamped to its axis range. Returns None if fewer than six integers.
pub fn parse_dims(output: &str) -> Option<[u8; 6]> {
    let lower = output.to_lowercase();
    let slice = match lower.find("dims") {
        Some(idx) => &output[idx..],
        None => output,
    };
    let nums: Vec<u8> = slice
        .split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse::<u8>().ok())
        .take(6)
        .collect();
    if nums.len() < 6 {
        return None;
    }
    let mut out = [0u8; 6];
    for i in 0..6 {
        out[i] = nums[i].min(SCALES[i]);
    }
    Some(out)
}

/// Fit six per-dimension LinearModels by reusing the v2 logistic fitter: dimension
/// `i`'s target is `dim_i / scale_i ∈ [0,1]` (SPEC-v3 §4.2).
pub fn fit_dims(data: &[DimsExample]) -> Vec<LinearModel> {
    (0..6)
        .map(|i| {
            let examples: Vec<dataset::LabeledExample> = data
                .iter()
                .map(|d| dataset::LabeledExample {
                    query: d.query.clone(),
                    difficulty: (dim_value(&d.dims, i) / SCALES[i] as f64).clamp(0.0, 1.0),
                    category: d.category.clone(),
                })
                .collect();
            crate::logreg::fit(&examples, &crate::logreg::FitConfig::default())
        })
        .collect()
}

/// Sanitize a model id into a filename fragment (`/` and `:` → `-`).
fn sanitize(model: &str) -> String {
    model.replace(['/', ':', ' '], "-")
}

/// Abort `label --dims` after this many *consecutive* network failures (a real
/// outage, not a transient blip). Mirrors v2.1 `label`'s outage guard.
const OUTAGE_LIMIT: u32 = 15;

/// Outcome of labeling one query's six dims (mirrors v2.1 `label::Outcome`).
enum DimsOutcome {
    /// Six integers parsed from the model output.
    Rated([u8; 6]),
    /// Output unparseable after a retry — skip this query (a re-run retries it).
    Unparseable,
    /// Request failed after local retries — skip; the caller tracks consecutive
    /// failures and aborts on a sustained outage.
    NetFail(String),
}

/// Pack six clamped integers into the canonical `DimScores`.
fn dims_scores(d: [u8; 6]) -> DimScores {
    DimScores {
        reasoning_depth: d[0],
        verification_difficulty: d[1],
        constraint_density: d[2],
        context_integration: d[3],
        ambiguity: d[4],
        error_cost: d[5],
    }
}

/// Build a `query → dims` reuse map from previously-saved rows, so a resumed run
/// skips queries it already labeled. Deterministic last-wins on duplicate
/// queries (labeling runs at `temperature=0`, so repeats agree). Pure; tested.
fn reuse_cache(existing: &[DimsExample]) -> HashMap<String, DimScores> {
    let mut m = HashMap::new();
    for e in existing {
        m.insert(e.query.clone(), e.dims.clone());
    }
    m
}

/// Label one query's six dims with local retries (2 parse-fails → `Unparseable`;
/// 3 net-fails with 1s/2s backoff → `NetFail`). Mirrors v2.1 `label::label_one`.
/// NETWORK; not unit-tested.
fn label_one_dims(
    client: &reqwest::blocking::Client,
    cfg: &LabelConfig,
    query: &str,
) -> DimsOutcome {
    let mut parse_fails = 0u32;
    let mut net_fails = 0u32;
    loop {
        match label::chat_complete(client, cfg, &build_dims_prompt(query)) {
            Ok(out) => match parse_dims(&out) {
                Some(d) => return DimsOutcome::Rated(d),
                None => {
                    parse_fails += 1;
                    if parse_fails >= 2 {
                        return DimsOutcome::Unparseable;
                    }
                }
            },
            Err(e) => {
                net_fails += 1;
                if net_fails >= 3 {
                    return DimsOutcome::NetFail(e);
                }
                std::thread::sleep(std::time::Duration::from_secs(net_fails as u64));
            }
        }
    }
}

/// Assemble the dims output for `corpus` **in corpus order**, reusing `cache`
/// (query → dims) and calling `fetch` on cache misses. Each corpus line emits one
/// row; a repeated query reuses its first label via the cache, so the output
/// stays 1:1 with the corpus minus unparseable skips — preserving the duplicate
/// rows the corpus intentionally allows and honoring the "N in → N out" contract
/// in `prompts/label.budget.prompt.md`. `checkpoint` persists partial progress
/// every 50 rows and once more right before an outage abort. Returns `Err` after
/// `OUTAGE_LIMIT` consecutive net failures. Network and disk are injected, so
/// this is unit-tested with fakes.
fn assemble_dims<F, C>(
    corpus: &[dataset::CorpusQuery],
    mut cache: HashMap<String, DimScores>,
    mut fetch: F,
    mut checkpoint: C,
) -> Result<Vec<DimsExample>, String>
where
    F: FnMut(&str) -> DimsOutcome,
    C: FnMut(&[DimsExample]),
{
    let mut labeled: Vec<DimsExample> = Vec::new();
    let mut consecutive_net = 0u32;
    for (i, q) in corpus.iter().enumerate() {
        if let Some(dims) = cache.get(&q.query) {
            labeled.push(DimsExample {
                query: q.query.clone(),
                category: q.category.clone(),
                dims: dims.clone(),
            });
            continue;
        }
        match fetch(&q.query) {
            DimsOutcome::Rated(d) => {
                let dims = dims_scores(d);
                cache.insert(q.query.clone(), dims.clone());
                labeled.push(DimsExample {
                    query: q.query.clone(),
                    category: q.category.clone(),
                    dims,
                });
                consecutive_net = 0;
            }
            DimsOutcome::Unparseable => {
                eprintln!("skip (unparseable) [{i}]: {}", q.query);
                consecutive_net = 0;
            }
            DimsOutcome::NetFail(e) => {
                consecutive_net += 1;
                eprintln!(
                    "net-skip [{i}] (consecutive {consecutive_net}): {}",
                    e.lines().next().unwrap_or("network error")
                );
                if consecutive_net >= OUTAGE_LIMIT {
                    checkpoint(&labeled);
                    return Err(format!(
                        "{consecutive_net} consecutive network failures — the server appears down.\n{e}"
                    ));
                }
            }
        }
        if (i + 1) % 50 == 0 {
            checkpoint(&labeled);
            eprintln!(
                "progress {}/{} (labeled {})",
                i + 1,
                corpus.len(),
                labeled.len()
            );
        }
    }
    Ok(labeled)
}

/// `label --dims`: read corpus, ask the LLM for six dims per query, write
/// `data/budget.<model>.jsonl`. Resumable (reuses already-labeled queries, keeps
/// duplicate rows) and outage-safe (aborts after a sustained outage, keeping the
/// last checkpoint). The only networked step. NETWORK; not unit-tested.
pub fn run_dims() {
    let cfg = LabelConfig::from_env();
    let client = label::http_client().expect("build HTTP client");
    let corpus = dataset::load_corpus("data/corpus.jsonl")
        .expect("load data/corpus.jsonl (run `trainer synth` first)");
    let out_path = format!("data/budget.{}.jsonl", sanitize(&cfg.model));

    let cache = dataset::load_dims(&out_path)
        .map(|existing| reuse_cache(&existing))
        .unwrap_or_default();

    let result = assemble_dims(
        &corpus,
        cache,
        |q| label_one_dims(&client, &cfg, q),
        |labeled| {
            let _ = dataset::save_dims(&out_path, labeled); // checkpoint
        },
    );

    match result {
        Ok(labeled) => {
            dataset::save_dims(&out_path, &labeled).expect("save budget labels");
            eprintln!(
                "label --dims: {} labeled (of {} corpus rows) -> {out_path}",
                labeled.len(),
                corpus.len()
            );
        }
        Err(reason) => {
            // The last checkpoint was saved inside assemble_dims before aborting.
            eprintln!("\nlabel --dims aborted: {reason}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dims_reads_six_integers() {
        assert_eq!(
            parse_dims("DIMS: 3 2 2 1 1 2\nbecause ..."),
            Some([3, 2, 2, 1, 1, 2])
        );
    }

    #[test]
    fn parse_dims_clamps_to_axis_ranges() {
        // ambiguity (idx 4) max is 3; error_cost (idx 5) max is 4.
        assert_eq!(parse_dims("DIMS: 9 9 9 9 9 9"), Some([4, 4, 4, 4, 3, 4]));
    }

    #[test]
    fn parse_dims_rejects_too_few() {
        assert_eq!(parse_dims("DIMS: 3 2 1"), None);
        assert_eq!(parse_dims("no numbers here"), None);
    }

    #[test]
    fn prompt_lists_all_six_axes() {
        let p = build_dims_prompt("reverse a linked list");
        for axis in [
            "reasoning_depth",
            "verification_difficulty",
            "constraint_density",
            "context_integration",
            "ambiguity",
            "error_cost",
        ] {
            assert!(p.contains(axis), "missing {axis}");
        }
        assert!(p.contains("reverse a linked list"));
    }

    #[test]
    fn fit_dims_returns_six_models() {
        let data = vec![
            DimsExample {
                query: "hi".into(),
                category: "chat".into(),
                dims: DimScores {
                    reasoning_depth: 0,
                    verification_difficulty: 0,
                    constraint_density: 0,
                    context_integration: 0,
                    ambiguity: 0,
                    error_cost: 0,
                },
            },
            DimsExample {
                query: "prove and derive step by step; analyze".into(),
                category: "math".into(),
                dims: DimScores {
                    reasoning_depth: 4,
                    verification_difficulty: 4,
                    constraint_density: 3,
                    context_integration: 2,
                    ambiguity: 1,
                    error_cost: 3,
                },
            },
        ];
        let models = fit_dims(&data);
        assert_eq!(models.len(), 6);
    }

    fn corpus_q(query: &str) -> dataset::CorpusQuery {
        dataset::CorpusQuery {
            query: query.into(),
            category: "chat".into(),
        }
    }

    fn ex(query: &str, d: [u8; 6]) -> DimsExample {
        DimsExample {
            query: query.into(),
            category: "chat".into(),
            dims: dims_scores(d),
        }
    }

    #[test]
    fn reuse_cache_dedups_duplicate_queries() {
        let existing = vec![
            ex("foo", [1, 0, 0, 0, 0, 0]),
            ex("bar", [2, 0, 0, 0, 0, 0]),
            ex("foo", [1, 0, 0, 0, 0, 0]),
        ];
        let m = reuse_cache(&existing);
        assert_eq!(m.len(), 2);
        assert_eq!(m["foo"].reasoning_depth, 1);
        assert_eq!(m["bar"].reasoning_depth, 2);
    }

    #[test]
    fn assemble_preserves_duplicate_rows() {
        // The corpus intentionally allows duplicate queries; the output must keep
        // BOTH rows (1:1 with corpus), unlike the old HashSet-by-query resume.
        let corpus = vec![corpus_q("foo"), corpus_q("bar"), corpus_q("foo")];
        let mut calls: Vec<String> = Vec::new();
        let out = assemble_dims(
            &corpus,
            HashMap::new(),
            |q| {
                calls.push(q.to_string());
                DimsOutcome::Rated([3, 0, 0, 0, 0, 0])
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(out.len(), 3, "duplicate query row must be preserved");
        assert_eq!(out[0].query, "foo");
        assert_eq!(out[2].query, "foo");
        // The repeat reused the cache: the labeler ran once per UNIQUE query.
        assert_eq!(calls, vec!["foo".to_string(), "bar".to_string()]);
    }

    #[test]
    fn assemble_reuses_existing_cache_without_refetching() {
        let corpus = vec![corpus_q("foo"), corpus_q("bar"), corpus_q("foo")];
        let mut cache = HashMap::new();
        cache.insert("foo".to_string(), dims_scores([4, 0, 0, 0, 0, 0]));
        let mut fetched: Vec<String> = Vec::new();
        let out = assemble_dims(
            &corpus,
            cache,
            |q| {
                fetched.push(q.to_string());
                DimsOutcome::Rated([1, 0, 0, 0, 0, 0])
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].dims.reasoning_depth, 4); // reused from cache
        assert_eq!(out[2].dims.reasoning_depth, 4);
        assert_eq!(fetched, vec!["bar".to_string()]); // only the miss hit the LLM
    }

    #[test]
    fn assemble_skips_unparseable_rows() {
        let corpus = vec![corpus_q("foo"), corpus_q("bar")];
        let out = assemble_dims(
            &corpus,
            HashMap::new(),
            |q| {
                if q == "foo" {
                    DimsOutcome::Unparseable
                } else {
                    DimsOutcome::Rated([2, 0, 0, 0, 0, 0])
                }
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].query, "bar");
    }

    #[test]
    fn assemble_aborts_on_sustained_outage() {
        let corpus: Vec<dataset::CorpusQuery> = (0..OUTAGE_LIMIT + 5)
            .map(|i| corpus_q(&format!("q{i}")))
            .collect();
        let mut checkpoints = 0u32;
        let result = assemble_dims(
            &corpus,
            HashMap::new(),
            |_| DimsOutcome::NetFail("connection refused".into()),
            |_| checkpoints += 1,
        );
        assert!(result.is_err(), "sustained outage must abort");
        assert!(
            checkpoints >= 1,
            "partial progress checkpointed before abort"
        );
    }
}
