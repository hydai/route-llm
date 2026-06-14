use crate::dataset::{self, CorpusQuery};

/// ★ (category, patterns with a single `{}` slot, topic fills). Owner-tunable.
/// Patterns within a category deliberately range easy→hard; the LLM assigns the
/// actual difficulty at label time, so intra-category spread becomes signal.
fn specs() -> Vec<(&'static str, Vec<&'static str>, Vec<&'static str>)> {
    vec![
        (
            "chat",
            vec![
                "hi {}",
                "thanks, {}!",
                "what's up with {}?",
                "tell me about {}",
                "good morning, any thoughts on {}?",
                "quick question about {}",
                "how do you feel about {}?",
                "small talk about {}",
                "say hello and mention {}",
                "got a minute to chat about {}?",
                "any fun stories about {}?",
            ],
            vec![
                "the weather",
                "your day",
                "coffee",
                "weekend plans",
                "music",
                "movies",
                "cats",
                "the news",
                "lunch",
                "travel",
                "books",
                "sports",
                "the office",
                "hobbies",
                "nothing much",
                "games",
                "food",
            ],
        ),
        (
            "extraction",
            vec![
                "What is {}?",
                "Define {} in one sentence.",
                "Summarize {} briefly.",
                "List three facts about {}.",
                "Translate '{}' to Spanish.",
                "Extract the key entities from a passage about {}.",
                "Give a one-line summary of {}.",
                "When did {} happen?",
                "Reformat this note about {} as bullet points.",
                "What does the acronym {} stand for?",
                "Pull out the main idea from a short text on {}.",
            ],
            vec![
                "photosynthesis",
                "the Eiffel Tower",
                "JSON",
                "the water cycle",
                "World War II",
                "HTTP",
                "the stock market",
                "DNA",
                "gravity",
                "the internet",
                "machine learning",
                "the French Revolution",
                "blockchain",
                "the solar system",
                "TCP",
                "REST",
                "OAuth",
            ],
        ),
        (
            "multilingual",
            vec![
                "請用一句話說明 {}。",
                "比較 {} 的優缺點並舉例。",
                "逐步解釋 {} 的運作原理。",
                "分析 {} 的效能瓶頸並提出優化。",
                "設計一個與 {} 相關的系統並說明取捨。",
                "證明關於 {} 的一個重要性質。",
                "為什麼 {} 重要?請推導。",
                "用中文比較 {} 的兩種實作方式。",
                "簡單說明 {} 是什麼。",
                "請評估 {} 的可擴展性並提出改進。",
            ],
            vec![
                "遞迴",
                "快速排序",
                "TCP 與 UDP",
                "梯度下降",
                "分散式快取",
                "資料庫索引",
                "一致性雜湊",
                "垃圾回收",
                "微服務架構",
                "並行控制",
                "B+ 樹",
                "向量時鐘",
                "共識演算法",
                "RSA 加密",
                "事件溯源",
                "讀寫分離",
            ],
        ),
        (
            "code",
            vec![
                "Fix this typo in {} code.",
                "Write a hello-world in {}.",
                "Explain what this {} snippet does.",
                "Implement a binary search in {}.",
                "Write unit tests for a {} function.",
                "Implement a thread-safe LRU cache in {}.",
                "Design and implement a rate limiter in {}.",
                "Implement a lock-free concurrent queue in {} and discuss ABA.",
                "Refactor a tangled {} module and justify each change.",
                "Profile and optimize a hot loop in {}.",
                "Format this {} code snippet.",
            ],
            vec![
                "Rust",
                "Python",
                "TypeScript",
                "Go",
                "Java",
                "C++",
                "Ruby",
                "Kotlin",
                "Scala",
                "Swift",
                "Elixir",
                "Haskell",
                "C",
                "SQL",
                "PHP",
            ],
        ),
        (
            "math",
            vec![
                "Compute {} + 7.",
                "Simplify the expression for {}.",
                "Solve a basic equation involving {}.",
                "Differentiate a function of {}.",
                "Prove a standard identity about {}.",
                "Derive the closed form for {} from first principles.",
                "Prove by induction a statement about {}.",
                "Analyze the convergence of a series involving {}.",
                "Evaluate {} for a small value.",
                "State a basic property of {}.",
            ],
            vec![
                "x",
                "a quadratic",
                "a geometric series",
                "the sine function",
                "primes",
                "the harmonic series",
                "a 2x2 matrix",
                "logarithms",
                "the binomial coefficients",
                "an integral of x^2",
                "eigenvalues",
                "the Fibonacci sequence",
                "modular arithmetic",
                "a limit",
                "a derivative",
                "a probability",
            ],
        ),
        (
            "reasoning",
            vec![
                "Briefly: why might {} matter?",
                "Compare two approaches to {}.",
                "Analyze the trade-offs in {} and recommend one.",
                "Prove a key property of {} and derive its complexity.",
                "Design {}, prove its correctness, and analyze failure modes.",
                "Step by step, derive and justify the design of {} under partitions.",
                "Prove the lower bound for {} and design an optimal strategy.",
                "In one line, what problem does {} solve?",
            ],
            vec![
                "consensus protocols",
                "the CAP theorem",
                "A* search",
                "Byzantine fault tolerance",
                "quicksort's worst case",
                "Dijkstra's algorithm",
                "two-phase commit",
                "garbage collection",
                "a distributed lock",
                "MVCC",
                "Raft",
                "a bloom filter",
                "rate limiting at scale",
                "leader election",
                "a CRDT",
                "sharding",
            ],
        ),
    ]
}

/// Build the corpus deterministically: for each category, every pattern × every
/// topic fill. Stable iteration order → reproducible corpus.jsonl.
pub fn build() -> Vec<CorpusQuery> {
    let mut out = Vec::new();
    for (category, patterns, fills) in specs() {
        for pat in &patterns {
            for fill in &fills {
                out.push(CorpusQuery {
                    query: pat.replace("{}", fill),
                    category: category.to_string(),
                });
            }
        }
    }
    out
}

/// `synth` subcommand: write the queries-only corpus. (No labels — `label` adds those.)
pub fn run() {
    let items = build();
    dataset::save_corpus("data/corpus.jsonl", &items).expect("write data/corpus.jsonl");
    eprintln!("synth: wrote {} queries to data/corpus.jsonl", items.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_reaches_target_size() {
        let n = build().len();
        assert!(
            (900..=1200).contains(&n),
            "corpus size {n} outside ~1000 target"
        );
    }

    #[test]
    fn deterministic() {
        assert_eq!(build(), build());
    }

    #[test]
    fn every_category_present_and_nonempty() {
        let items = build();
        for cat in [
            "chat",
            "extraction",
            "multilingual",
            "code",
            "math",
            "reasoning",
        ] {
            assert!(
                items.iter().any(|q| q.category == cat),
                "missing category {cat}"
            );
        }
    }

    #[test]
    fn queries_are_unique_enough() {
        let items = build();
        let mut q: Vec<&str> = items.iter().map(|x| x.query.as_str()).collect();
        q.sort_unstable();
        q.dedup();
        // No `{}` slots left unfilled.
        assert!(items.iter().all(|x| !x.query.contains("{}")));
        // Mostly-unique queries (combinatorial fill shouldn't collide much).
        assert!(q.len() as f64 > items.len() as f64 * 0.95);
    }
}
