use serde::{Deserialize, Serialize};

/// A corpus query with no label — what `synth` produces and `label` consumes.
// Wired into `corpus.rs`/`label.rs` in later v2.1 tasks; allow until then.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CorpusQuery {
    pub query: String,
    pub category: String,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn to_corpus_jsonl(items: &[CorpusQuery]) -> String {
    let mut s = items
        .iter()
        .map(|x| serde_json::to_string(x).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    s.push('\n');
    s
}

#[allow(dead_code)]
pub fn load_corpus(path: &str) -> Result<Vec<CorpusQuery>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    parse_corpus_jsonl(&text)
}

#[allow(dead_code)]
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
}
