//! Offline trainer for the learned router. Subcommands added per task:
//! `synth` (Task 6), `fit` (Task 7/8), `eval` (Task 9). `label` (LLM) deferred.
#![allow(dead_code)] // helper modules are wired incrementally across Tasks 6–9

mod corpus;
mod dataset;

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "synth" => corpus::run(),
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
