use crate::dataset::{self, LabeledExample};

/// ★ (category, target difficulty in 0..1, query templates). Owner-tunable.
fn templates() -> Vec<(&'static str, f64, Vec<&'static str>)> {
    vec![
        (
            "chat",
            0.10,
            vec!["hi", "thanks!", "what time is it?", "tell me a joke"],
        ),
        (
            "extraction",
            0.30,
            vec![
                "Summarize this paragraph in one sentence.",
                "Extract the names from: Alice, Bob, Carol.",
            ],
        ),
        (
            "multilingual",
            0.55,
            vec![
                "請逐步說明為什麼這段程式碼會出錯，並提供修正。",
                "比較這兩個演算法的時間複雜度並證明。",
            ],
        ),
        (
            "code",
            0.65,
            vec![
                "Write a Rust function to reverse a linked list and explain it.",
                "Debug this: ```fn main(){ let x: i32 = \"s\"; }```",
            ],
        ),
        (
            "math",
            0.70,
            vec![
                "Compute the integral $\\int_0^1 x^2 dx$ and justify each step.",
                "Prove that the square root of 2 is irrational.",
            ],
        ),
        (
            "reasoning",
            0.88,
            vec![
                "Prove step by step why Paxos guarantees safety and derive its invariant.",
                "Analyze, compare, and design a consensus protocol; justify each choice.",
            ],
        ),
    ]
}

/// Build the labeled dataset deterministically from templates.
pub fn build() -> Vec<LabeledExample> {
    let mut out = Vec::new();
    for (cat, diff, qs) in templates() {
        for q in qs {
            out.push(LabeledExample {
                query: q.to_string(),
                difficulty: diff,
                category: cat.to_string(),
            });
        }
    }
    out
}

/// `synth` subcommand: write corpus + interim labels.
pub fn run() {
    let items = build();
    dataset::save("data/labeled.jsonl", &items).expect("write data/labeled.jsonl");
    dataset::save("data/corpus.jsonl", &items).expect("write data/corpus.jsonl");
    eprintln!("synth: wrote {} labeled examples to data/", items.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_is_nonempty_and_in_unit_interval() {
        let items = build();
        assert!(items.len() >= 12);
        assert!(items
            .iter()
            .all(|x| x.difficulty > 0.0 && x.difficulty < 1.0));
    }

    #[test]
    fn deterministic() {
        assert_eq!(build(), build());
    }

    #[test]
    fn easy_and_hard_bands_present() {
        let items = build();
        assert!(items.iter().any(|x| x.difficulty < 0.2));
        assert!(items.iter().any(|x| x.difficulty > 0.8));
    }
}
