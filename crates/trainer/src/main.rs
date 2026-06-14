//! Offline trainer for the learned router.
//! Subcommands are dispatched in `main`; run with no/invalid args to print usage.

mod corpus;
mod dataset;
mod emit;
mod eval;
mod gold;
mod label;
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
        "eval" => {
            let args: Vec<String> = std::env::args().collect();
            match eval::parse_in_flag(&args) {
                Some(path) => eval::run_path(&path),
                None => eval::run(),
            }
        }
        "compare" => {
            let files: Vec<String> = std::env::args().skip(2).collect();
            eval::compare(&files);
        }
        "gold-pool" => gold::run_pool(),
        "label" => label::run(),
        other => {
            eprintln!("usage: trainer <synth|label|fit|eval [--in <file>]|compare <files...>|gold-pool>");
            if !other.is_empty() {
                eprintln!("unknown subcommand: {other:?}");
            }
            std::process::exit(2);
        }
    }
}
