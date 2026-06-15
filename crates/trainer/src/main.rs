//! Offline trainer for the learned router.
//! Subcommands are dispatched in `main`; run with no/invalid args to print usage.

mod budget_label;
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
            // A flag present without a value is a usage error, not a silent fallback.
            for flag in ["--gold", "--in"] {
                if rest.iter().any(|a| a == flag) && eval::parse_flag(&rest, flag).is_none() {
                    eprintln!(
                        "usage: trainer eval [--in <file>|--gold <file>] — {flag} requires a value"
                    );
                    std::process::exit(2);
                }
            }
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
            match eval::parse_compare_args(&rest) {
                Ok((Some(g), files)) => eval::compare_gold(&g, &files),
                Ok((None, files)) => eval::compare(&files),
                Err(e) => {
                    eprintln!("usage: trainer compare [--gold <file>] <files...> — {e}");
                    std::process::exit(2);
                }
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
