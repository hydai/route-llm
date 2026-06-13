//! Offline trainer for the learned router. Subcommands added per task:
//! `synth` (Task 6), `fit` (Task 7/8), `eval` (Task 9). `label` (LLM) deferred.

mod corpus;
mod dataset;
mod emit;
mod eval;
mod logreg;

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "synth" => corpus::run(),
        "fit" => {
            let data = dataset::load("data/labeled.jsonl").expect("load labeled.jsonl");
            let model = logreg::fit(&data, &logreg::FitConfig::default());
            emit::write(&model, "crates/core/src/learned/weights.rs").expect("write weights.rs");
            eprintln!(
                "fit: {} examples -> crates/core/src/learned/weights.rs (bias={:.3})",
                data.len(),
                model.bias
            );
        }
        "eval" => eval::run(),
        // arms wired in later tasks
        "label" => {
            eprintln!("`label` (LLM re-labeling) is deferred; see SPEC-v2 §7/§16.");
            std::process::exit(2);
        }
        other => {
            eprintln!("usage: trainer <synth|fit|eval>");
            if !other.is_empty() {
                eprintln!("unknown subcommand: {other:?}");
            }
            std::process::exit(2);
        }
    }
}
