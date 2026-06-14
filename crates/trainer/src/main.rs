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
            let rest: Vec<String> = std::env::args().skip(2).collect();
            if let Some(gold) = eval::parse_flag(&rest, "--gold") {
                eval::run_gold(&gold);
            } else if let Some(path) = eval::parse_in_flag(&rest) {
                eval::run_path(&path);
            } else {
                eval::run();
            }
        }
        "compare" => {
            let rest: Vec<String> = std::env::args().skip(2).collect();
            let (gold, files) = eval::parse_compare_args(&rest);
            match gold {
                Some(g) => eval::compare_gold(&g, &files),
                None => eval::compare(&files),
            }
        }
        "gold-pool" => gold::run_pool(),
        "crosseval" => {
            let files: Vec<String> = std::env::args().skip(2).collect();
            eval::crosseval(&files);
        }
        "label" => label::run(),
        other => {
            eprintln!("usage: trainer <synth|label|fit|eval [--in <file>|--gold <file>]|compare [--gold <file>] <files...>|crosseval [files...]|gold-pool>");
            if !other.is_empty() {
                eprintln!("unknown subcommand: {other:?}");
            }
            std::process::exit(2);
        }
    }
}
