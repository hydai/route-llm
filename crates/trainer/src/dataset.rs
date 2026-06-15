use serde::{Deserialize, Serialize};

/// A corpus query with no label — what `synth` produces and `label` consumes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CorpusQuery {
    pub query: String,
    pub category: String,
}

pub fn parse_corpus_jsonl(text: &str) -> Result<Vec<CorpusQuery>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let q: CorpusQuery =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        out.push(q);
    }
    Ok(out)
}

pub fn to_corpus_jsonl(items: &[CorpusQuery]) -> String {
    let mut s = items
        .iter()
        .map(|x| serde_json::to_string(x).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

pub fn load_corpus(path: &str) -> Result<Vec<CorpusQuery>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    parse_corpus_jsonl(&text)
}

pub fn save_corpus(path: &str, items: &[CorpusQuery]) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(path, to_corpus_jsonl(items)).map_err(|e| format!("write {path}: {e}"))
}

/// One labeled training example. `difficulty` is the target in 0..1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabeledExample {
    pub query: String,
    pub difficulty: f64,
    #[serde(default)]
    pub category: String,
}

pub fn parse_jsonl(text: &str) -> Result<Vec<LabeledExample>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ex: LabeledExample =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        out.push(ex);
    }
    Ok(out)
}

pub fn to_jsonl(items: &[LabeledExample]) -> String {
    let mut s = items
        .iter()
        .map(|x| serde_json::to_string(x).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

pub fn load(path: &str) -> Result<Vec<LabeledExample>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    parse_jsonl(&text)
}

pub fn save(path: &str, items: &[LabeledExample]) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(path, to_jsonl(items)).map_err(|e| format!("write {path}: {e}"))
}

/// One query's six budget-dimension integer ratings (frontier-LLM labeled).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimScores {
    pub reasoning_depth: u8,
    pub verification_difficulty: u8,
    pub constraint_density: u8,
    pub context_integration: u8,
    pub ambiguity: u8,
    pub error_cost: u8,
}

/// A 6-dim labeled example: `data/budget.<labeler>.jsonl` line shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimsExample {
    pub query: String,
    #[serde(default)]
    pub category: String,
    pub dims: DimScores,
}

pub fn parse_dims_jsonl(text: &str) -> Result<Vec<DimsExample>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ex: DimsExample =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        out.push(ex);
    }
    Ok(out)
}

pub fn to_dims_jsonl(items: &[DimsExample]) -> String {
    let mut s = items
        .iter()
        .map(|x| serde_json::to_string(x).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

pub fn load_dims(path: &str) -> Result<Vec<DimsExample>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    parse_dims_jsonl(&text)
}

pub fn save_dims(path: &str, items: &[DimsExample]) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(path, to_dims_jsonl(items)).map_err(|e| format!("write {path}: {e}"))
}

/// Dimension value by canonical index 0..6 (matches `budget::dims::DIM_NAMES`).
pub fn dim_value(d: &DimScores, i: usize) -> f64 {
    [
        d.reasoning_depth,
        d.verification_difficulty,
        d.constraint_density,
        d.context_integration,
        d.ambiguity,
        d.error_cost,
    ][i] as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let items = vec![LabeledExample {
            query: "hi".into(),
            difficulty: 0.2,
            category: "chat".into(),
        }];
        assert_eq!(parse_jsonl(&to_jsonl(&items)).unwrap(), items);
    }

    #[test]
    fn skips_blank_lines() {
        let s = "{\"query\":\"a\",\"difficulty\":0.5}\n\n";
        assert_eq!(parse_jsonl(s).unwrap().len(), 1);
    }

    #[test]
    fn corpus_query_round_trip() {
        let items = vec![
            CorpusQuery {
                query: "hi".into(),
                category: "chat".into(),
            },
            CorpusQuery {
                query: "prove X".into(),
                category: "math".into(),
            },
        ];
        let s = to_corpus_jsonl(&items);
        assert_eq!(parse_corpus_jsonl(&s).unwrap(), items);
    }

    #[test]
    fn corpus_query_skips_blank_lines() {
        let s = "{\"query\":\"a\",\"category\":\"chat\"}\n\n";
        assert_eq!(parse_corpus_jsonl(s).unwrap().len(), 1);
    }

    #[test]
    fn dims_example_round_trips() {
        let items = vec![DimsExample {
            query: "prove X".into(),
            category: "math".into(),
            dims: DimScores {
                reasoning_depth: 4,
                verification_difficulty: 3,
                constraint_density: 1,
                context_integration: 0,
                ambiguity: 2,
                error_cost: 1,
            },
        }];
        let s = to_dims_jsonl(&items);
        assert_eq!(parse_dims_jsonl(&s).unwrap(), items);
        assert!(s.contains("\"reasoning_depth\":4"));
    }
}
