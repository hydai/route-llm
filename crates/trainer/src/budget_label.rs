//! Frontier-LLM labeling of the six reasoning-budget dimensions (offline only).
//! Reuses `label`'s OpenAI-compatible client. Writes `data/budget.<model>.jsonl`.
//! Never used at inference. See SPEC-v3 §4.3.

use crate::dataset::{self, dim_value, DimScores, DimsExample};
use crate::label::{self, LabelConfig};
use route_llm_core::learned::model::LinearModel;

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

/// `label --dims`: read corpus, ask the LLM for six dims per query, write
/// `data/budget.<model>.jsonl`. Resumable: existing output rows are skipped.
/// The only networked step. NETWORK; not unit-tested.
pub fn run_dims() {
    let cfg = LabelConfig::from_env();
    let client = label::http_client().expect("build HTTP client");
    let corpus = dataset::load_corpus("data/corpus.jsonl")
        .expect("load data/corpus.jsonl (run `trainer synth` first)");
    let out_path = format!("data/budget.{}.jsonl", sanitize(&cfg.model));

    let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut labeled: Vec<DimsExample> = Vec::new();
    if let Ok(existing) = dataset::load_dims(&out_path) {
        for e in existing {
            done.insert(e.query.clone());
            labeled.push(e);
        }
    }

    let (mut ok, mut skip) = (0usize, 0usize);
    for (i, q) in corpus.iter().enumerate() {
        if done.contains(&q.query) {
            continue;
        }
        match label::chat_complete(&client, &cfg, &build_dims_prompt(&q.query)) {
            Ok(out) => match parse_dims(&out) {
                Some(d) => {
                    labeled.push(DimsExample {
                        query: q.query.clone(),
                        category: q.category.clone(),
                        dims: DimScores {
                            reasoning_depth: d[0],
                            verification_difficulty: d[1],
                            constraint_density: d[2],
                            context_integration: d[3],
                            ambiguity: d[4],
                            error_cost: d[5],
                        },
                    });
                    ok += 1;
                }
                None => {
                    eprintln!("skip (unparseable) [{i}]: {}", q.query);
                    skip += 1;
                }
            },
            Err(e) => {
                eprintln!(
                    "net-skip [{i}]: {}",
                    e.lines().next().unwrap_or("network error")
                );
                skip += 1;
            }
        }
        if (i + 1) % 50 == 0 {
            let _ = dataset::save_dims(&out_path, &labeled); // checkpoint
            eprintln!("progress {}/{} (ok {ok}, skip {skip})", i + 1, corpus.len());
        }
    }
    dataset::save_dims(&out_path, &labeled).expect("save budget labels");
    eprintln!("label --dims: {ok} labeled, {skip} skipped -> {out_path}");
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
}
