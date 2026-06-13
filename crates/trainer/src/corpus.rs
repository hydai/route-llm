use crate::dataset::{self, LabeledExample};

/// ★ (category, target difficulty in 0..1, query templates). Owner-tunable.
fn templates() -> Vec<(&'static str, f64, Vec<&'static str>)> {
    vec![
        (
            "chat",
            0.10,
            vec![
                "hi",
                "thanks!",
                "what time is it?",
                "tell me a joke",
                "hello there",
                "good morning",
                "bye",
                "how are you?",
                "yes please",
                "ok",
                "sounds good",
                "sure",
            ],
        ),
        (
            "extraction",
            0.30,
            vec![
                "Summarize this paragraph in one sentence.",
                "Extract the names from: Alice, Bob, Carol.",
                "List the main points from the following text.",
                "What is the capital city of France?",
                "Translate 'hello' to Spanish.",
                "Convert 100 USD to EUR.",
                "What does JSON stand for?",
                "Extract all email addresses from this text.",
                "Give me a one-line summary of photosynthesis.",
                "What year was the Eiffel Tower built?",
                "Define the word 'ephemeral'.",
                "Reformat this date: 2024-01-15 to January 15th, 2024.",
            ],
        ),
        (
            "multilingual",
            0.55,
            vec![
                "請逐步說明為什麼這段程式碼會出錯，並提供修正。",
                "比較這兩個演算法的時間複雜度並證明。",
                "解釋為什麼遞迴可以替代迭代，並給出例子。",
                "分析這段 Python 程式碼的效能瓶頸並優化。",
                "用中文說明 TCP 與 UDP 的差異，並給出適用場景。",
                "設計一個資料庫 schema 來管理線上書店的訂單，並說明每個欄位的用途。",
                "為什麼快速排序的平均複雜度是 O(n log n)？請推導。",
                "比較 React 和 Vue 的狀態管理方式，哪個更適合大型專案？",
                "請解釋梯度下降法的原理，並說明學習率的選擇對收斂的影響。",
                "設計一個分散式快取系統並說明一致性的處理方式。",
            ],
        ),
        (
            "code",
            0.65,
            vec![
                "Write a Rust function to reverse a linked list and explain it.",
                "Debug this: ```fn main(){ let x: i32 = \"s\"; }```",
                "Implement a binary search tree in Python with insert, delete, and search.",
                "Write a function to detect cycles in a directed graph using DFS.",
                "Create a REST API endpoint in Rust using axum that handles JSON.",
                "Implement a thread-safe LRU cache in Rust using Arc and Mutex.",
                "Write a recursive function to solve the Tower of Hanoi problem.",
                "Debug this SQL query: SELECT * FROM users WHERE id = '123' AND active;",
                "Implement merge sort in TypeScript and analyze its time complexity.",
                "Write a regex pattern to validate email addresses and explain each part.",
                "Create a Python decorator that caches function results with TTL.",
                "Implement a state machine for a traffic light system in Rust.",
                "Write a function to parse CSV files handling quoted fields and newlines.",
                "Design a rate limiter using the token bucket algorithm in Go.",
            ],
        ),
        (
            "math",
            0.70,
            vec![
                "Compute the integral $\\int_0^1 x^2 dx$ and justify each step.",
                "Prove that the square root of 2 is irrational.",
                "Solve the differential equation $y'' + 4y' + 4y = 0$ with initial conditions.",
                "Prove by induction that the sum of the first n natural numbers is n(n+1)/2.",
                "Find the eigenvalues and eigenvectors of the matrix [[1,2],[3,4]].",
                "Compute $\\sum_{n=1}^{\\infty} \\frac{1}{n^2}$ and prove your answer.",
                "Derive the quadratic formula from first principles.",
                "Prove the Cauchy-Schwarz inequality for inner product spaces.",
                "Calculate the determinant of a 4x4 matrix using cofactor expansion.",
                "Solve the system of linear equations: 2x+3y=7, 4x-y=5; show all steps.",
                "Prove that there are infinitely many prime numbers.",
                "Derive the formula for the volume of a sphere using integration.",
                "Show that e is irrational using the Taylor series expansion.",
            ],
        ),
        (
            "reasoning",
            0.88,
            vec![
                "Prove step by step why Paxos guarantees safety and derive its invariant.",
                "Analyze, compare, and design a consensus protocol; justify each choice.",
                "Explain why the CAP theorem implies trade-offs in distributed systems; design a system that maximizes availability while proving it sacrifices consistency.",
                "Derive the time complexity of the A* search algorithm and prove it is optimal for admissible heuristics; analyze edge cases.",
                "Compare Byzantine fault tolerance vs crash fault tolerance; prove that BFT requires 3f+1 nodes; design a protocol for f=2.",
                "Analyze why quicksort degrades to O(n^2) in the worst case; prove this bound; design a pivot selection strategy to avoid it.",
                "Prove that P != NP implies RSA is secure; analyze what happens if P = NP; compare RSA with elliptic curve cryptography.",
                "Design a lock-free concurrent hash map in Rust; prove it is deadlock-free; analyze ABA problem risks and mitigations.",
                "Analyze the trade-offs in microservices vs monolith architectures; prove that network partition handling increases complexity by O(n^2); design a hybrid approach.",
                "Step by step: prove the correctness of Dijkstra's algorithm; derive its complexity; explain why it fails with negative weights; compare with Bellman-Ford.",
                "Analyze garbage collection algorithms (mark-sweep, generational, reference counting); derive their complexity; prove that no algorithm is optimal for all workloads.",
                "Design a distributed transaction protocol; prove its correctness under network partitions; analyze the two-phase commit problem and derive a solution.",
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
