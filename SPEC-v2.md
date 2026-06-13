# route-llm v2 — 設計規格：學習式難度路由器 (Learned Difficulty Router)

> 狀態：v2 已實作（本 PR：core/server/trainer/data 全數完成；待合併）
> 日期：2026-06-13
> 關係：本規格在 v1（見 `SPEC.md`）之上做**加法**，不更動 v1 的 API、回應形狀與行為。

## 1. 概述與動機

v1 用手寫啟發式（關鍵字規則 + 人工猜定的權重）估計 query 難度。它可用、可決定、零相依，但難度估計本身偏粗糙、權重未經資料校準。

v2 的目標是**只升級「估計難度」這一步**：用一個**離線訓練、線上零網路推論**的學習式模型取代手寫權重，提升選模推薦的準確度——同時完整保留 v1 的產品身分：

- 仍是**預測式 pre-flight picker**：只回推薦排序，**推論時不呼叫任何 LLM**。
- 推論仍是**純 Rust、可決定性、零網路、可完整單元測試**。
- v1 的啟發式策略**原封不動保留**，作為 fallback 與比較基準。

## 2. 目標與非目標

### 目標 (v2)

1. 新增第二種 `Router` 策略 `LearnedRouter`：以**離線擬合的邏輯迴歸**，從更豐富的純 Rust 特徵估計難度。
2. **沿用 v1 的排序器**（成本-品質權衡 `adequacy − λ·cost`）**不變**。
3. 新增 `trainer` 離線管線：語料合成 → 標註（初期取自合成的目標難度；後續可改 LLM）→ 擬合 → 內嵌權重。
4. 推論時**零網路、可決定性、可單元測試**，與 v1 同等。
5. 可用環境變數在 `heuristic` / `learned` 間切換；**HTTP API / DTO / 回應形狀完全不變**。
6. 提供 `eval`，以**證據**證明 `learned` 在留出集上勝過 `heuristic`。

### 非目標 (v2，YAGNI)

- 推論時**不**呼叫任何 LLM / embedding API（網路只可能出現在延後的離線 `label` 步驟）。
- **不**引入重量級 ML runtime（無 candle / ONNX / 神經 embedding 模型）；只用純 Rust 線性模型。
- **不**更動 v1 的 `difficulty.rs` / `HeuristicRouter`（凍結）。
- **不**做 v1 §15 其餘項目（named pool、從外部檔案載入模型表、串流、auth、rate limiting…）。
- **不**改 API、DTO、`Recommendation` 形狀。

## 3. 架構與隔離邊界

整個設計的核心決策是 **v1 與 v2 的難度估計器彼此獨立、互不污染**。判準是一條清楚的線：

> **會各自獨立演化的東西要隔離；刻意要保持一模一樣的東西才共用。**
> 難度估計器各自演化（v1 凍結、v2 學習）→ **隔離**。
> 排序器、`Router` trait、領域型別刻意保持相同 → **共用**。

### 模組樹

```
crates/core/src/
  model.rs        共用型別：Difficulty、Recommendation、RankedModel…（穩定介面）
  router.rs       Router trait + HeuristicRouter          ← v1，凍結、不動一行
  difficulty.rs   v1 專屬、凍結：HeuristicRouter 的難度估計
  ranker.rs       共用：成本-品質排序（v2 沿用不變，只吃抽象 Difficulty）
  registry.rs     v1，沿用不變
  learned/        ★ v2 隔離子系統（全新增）
    mod.rs          LearnedRouter (impl Router)
    features.rs     特徵抽取 fn features(&str) -> FeatureVec（只跟 trainer 共用）
    model.rs        LinearModel + 標準化參數 + 難度計算
    weights.rs      內嵌的擬合權重（trainer 產生、提交進 repo）
crates/server/    不動（除 §9 的 router 選擇）
crates/trainer/   ★ 離線 bin crate（網路相依只關在這裡）
data/             ★ corpus.jsonl（語料）、labeled.jsonl（標註，提交）
```

- **v1 路徑**（`difficulty.rs` + `HeuristicRouter`）整段凍結，永遠是穩定 fallback / 參考基準，不會因 v2 而回歸。
- **v2 路徑**（`learned/`）自成隔離子系統，特徵想多豐富都行，**只跟 `trainer` 共用**，碰不到 v1。
- 兩者唯一交會點是乾淨介面 **`Router` trait**；server 靠它在啟動時選策略，彼此看不到對方內部。
- `ranker`、`Router` trait、`Recommendation` / `Difficulty` 型別**共用**——它們是刻意保持相同的核心，且 `ranker` 只操作抽象的 `Difficulty` 數值，不碰任一方的難度內部。

### 資料流（線上推論，零網路）

```
query ─► learned::features(query) ─► 標準化 ─► LinearModel ─► Difficulty
                                                                  │
                                              （與 HeuristicRouter 殊途同歸）
                                                                  ▼
                                              ranker::rank(Difficulty, profiles, prefs)
                                                                  ▼
                                                           Recommendation
```

### 資料流（離線訓練，可碰網路、偶爾手動）

```
data/corpus.jsonl ──(初期)合成標註──► data/labeled.jsonl ──fit(純Rust)──► crates/core/src/learned/weights.rs
   （以合成為主）  （後續可 LLM 重標，延後）   （提交進 repo）   （決定性、零網路、CI 可重跑）   （內嵌權重）
```

**防漂移關鍵**：`trainer` 與 `LearnedRouter` 透過 `route-llm-core::learned::features` 共用**同一份**抽特徵程式，杜絕「訓練時 / 推論時特徵不一致」這個靜默 bug。

## 4. 特徵 (`learned/features.rs`)

`fn features(query: &str) -> FeatureVec`——純函式、決定性、零 I/O，回傳**固定長度、順序穩定**的數值向量，並附帶 `SCHEMA_VERSION`。

特徵集 = v1 既有訊號的數值化 + 新增的廉價純 Rust 特徵：

| 類別 | 特徵（示意起始集，最終清單見 §15） |
|------|------|
| 沿用 v1 訊號 | length_contrib、has_code、has_math、reasoning_hits（clamp 後）、multi_constraint、structured_output、explanation_request |
| 規模 | char_count、word_count、avg_word_len、sentence_count |
| 形態比例 | question_ratio、uppercase_ratio、digit_ratio、punctuation_ratio |
| 詞彙 | lexical_diversity（distinct/total token）、code_fence_count、url_present |
| 多語 | cjk_ratio、non_ascii_ratio |
| 廉價 lexical | hashed char-ngram bins（固定 K 個 bucket；K 見 §15） |

- 連續特徵在抽取時即做必要的 clamp（沿用 v1「貢獻上限」的語意），使其能被純線性模型表達。
- `SCHEMA_VERSION`：特徵集一旦變更就 bump。`weights.rs` 帶相同版本；載入時斷言一致（見 §10），防止特徵 / 權重錯配。

## 5. 模型 (`learned/model.rs`)

```rust
pub struct LinearModel {
    pub schema_version: u32,
    pub weights: Vec<f64>,       // 長度 = 特徵維度
    pub bias: f64,
    pub feature_means: Vec<f64>, // 標準化用
    pub feature_stds: Vec<f64>,  // 標準化用（std≈0 時以保護值代入）
}
```

- **標準化**：`z_i = (x_i − mean_i) / std_i`。
- **難度**：`score = sigmoid(Σ_i w_i · z_i + bias) ∈ (0,1)`——天然落在排序器要的 0..1 區間。
- **signals**：取貢獻值 `w_i · z_i` 最大的前 k 個特徵名，維持 v1 的可解釋性（`Difficulty.signals` 語意不變）。
- **不變式**：`weights` / `feature_means` / `feature_stds` 長度一致且等於特徵維度；`schema_version` 與 `features.rs` 一致。

## 6. LearnedRouter (`learned/mod.rs`)

```rust
pub struct LearnedRouter; // 權重來自內嵌的 weights.rs

impl Router for LearnedRouter {
    fn recommend(&self, query, models, prefs) -> Recommendation {
        let difficulty = model::difficulty(query); // features → 標準化 → sigmoid
        let ranking = ranker::rank(&difficulty, models, prefs); // ★ 共用 v1 排序器
        Recommendation { difficulty, ranking }
    }
}
```

權重以內嵌方式提供（生成的 `weights.rs`，`const` 或 `include!`），與 v1「模型表編譯進二進位檔」一致——推論零檔案相依。

## 7. 離線管線 (`crates/trainer`)

一個獨立 bin crate，三個子指令：

| 子指令 | 作用 | 網路 | 決定性 |
|--------|------|------|--------|
| `synth` | 合成多樣化 query 並賦予**目標難度級別**，寫 `corpus.jsonl` + 初期 `labeled.jsonl`（分級 1–5 → 映射 [0,1]） | 否 | 是（固定 seed / 模板） |
| `label` | （**延後**）以 LLM 重新標註以提升標籤品質；用哪個 LLM 待定（§16） | 是（離線、偶爾手動） | 否 |
| `fit` | 讀**已提交的** `labeled.jsonl` → `learned::features` → 標準化 → 擬合邏輯迴歸（純 Rust 梯度下降，固定 seed）→ 計算 means/stds → 產出 `weights.rs` | 否 | **是** |
| `eval` | 留出切分；報告 `learned` vs `heuristic` vs `always-strongest` 的指標（見 §12） | 否 | 是 |

- **相依限定**：LLM client（`reqwest` + `serde` 等）只出現在 `trainer` crate；`core` / `server` 不受影響、維持精簡。
- **標註規格**：難度分級 rubric（1–5）於實作階段定稿；合成的目標難度即依此 rubric 賦予。LLM prompt 模板待選定 LLM 後再定（§16）。

## 8. 資料 (`data/`)

- `corpus.jsonl`：**以合成為主**的多樣化 query（chat / code / math / reasoning / extraction / multilingual / trivial），規模數百~數千筆；亦可少量納入真實 prompt 樣本。
- `labeled.jsonl`：**提交進 repo**，作為可重現的訓練輸入。
- **標註來源（分階段）**：
  - **初期**：直接採用合成時賦予的目標難度級別（零外部相依），讓 `features → fit → eval → LearnedRouter` 整條管線能先端到端跑通並測試。
  - **後續（延後決定）**：以 LLM 重新標註以提升標籤品質——但**用哪個 LLM 等其他細節確定後再定**（§16）。
- **可重現性模型**：把「不可重現的那一刻」（合成或 LLM 標註）凍結成提交的 `labeled.jsonl`；任何人對固定的 `labeled.jsonl` 跑 `fit` 都得到**一模一樣**的 `weights.rs`。重新標註是顯式、偶發的步驟，網路永不進入 `fit` / 推論 / CI。

## 9. 設定與策略選擇

- 環境變數 `ROUTE_LLM_ROUTER=learned|heuristic`（**預設 `learned`**；未知值 → 啟動即失敗，同 `ROUTE_LLM_PORT` 的處理）。
- 兩種 router 都編譯進二進位檔。
- **HTTP API / DTO / `Recommendation` 形狀完全不變**；v1 的跨方言一致性測試仍成立。
- （可選）`/health` 或回應 meta 帶上 active strategy 名稱以增透明度——標為可選，預設不加。

## 10. 錯誤處理

- **推論無新錯誤面**：難度估計是純函式、權重恆在，不引入新的 4xx/5xx。
- `ROUTE_LLM_ROUTER` 未知值 → 啟動失敗（fail fast）。
- `weights.rs` 的 `schema_version` 與 `features.rs` 不符 → 編譯期 / 啟動期 assert（內嵌情況下理論上不會發生，但保留防線）。
- `trainer` 自身錯誤（LLM API 失敗、語料解析錯誤）屬離線工具範疇，不影響線上服務。

## 11. 相依套件

- **`core` 推論**：無新增 runtime 相依（純 Rust 線性模型 + 特徵）。
- **`trainer`**：`reqwest`（或等價 HTTP client）、`serde`/`serde_json`、（可能）`rand`（GD 洗牌 / seed）。**全部限定於 `trainer` crate**。
- 維持專案 Rust 規範：`[profile.dev.package."*"] debug = false`；一律 release build。

## 12. 測試策略 (TDD)

- **`features`**：固定長度 / 順序、決定性、特定 query 的特徵斷言（如含程式碼框 → `code_fence_count > 0`）。
- **`HeuristicRouter` 凍結回歸**：對 v1 `SPEC.md` §9 的範例向量，難度輸出**與 v1 逐位一致**——證明 v2 未污染 v1。
- **`LearnedRouter`**：
  - 手設小權重 → 驗單調性與 `signals` 取 top 貢獻者。
  - 用**出貨權重** → 「trivial query 難度 < hard query 難度」的 sanity 測試（守住內嵌權重不退化）。
- **`ranker` 共用**：兩個 router 各自算出的 `Difficulty` 都能正確排序；三方言對同一輸入排序一致。
- **`trainer`**：可分離合成資料 → `fit` 收斂到已知分界；`fit` 決定性（固定 seed + 資料 → 相同 `weights`）；`weights.rs` 序列化 round-trip。
- **eval 關卡（「變聰明」的證據）**：在留出集上，`learned` 需**勝過** `heuristic` 才採用，並同時看兩類指標：
  - **對標註的擬合度**：Spearman 相關 + 序數準確率。
  - **固定品質下的成本節省**：在固定的「充分率」（query 被路由到 adequate 模型——`adequacy ≥ 門檻`——的比例）下比較平均成本，並對照 `always-strongest` 基線；`learned` 應在相同充分率下更省。

## 13. 向後相容

- v1 的 `difficulty.rs` / `HeuristicRouter` / `router.rs` / `registry.rs` **不動**；`Router` trait、`Recommendation`、`ranker` 不變。
- v2 純為**加法**：新模組 `learned/` + 新 crate `trainer` + `data/`；現有測試應全數維持綠燈。
- 預設切到 `learned`，但設一個環境變數即可回退 `heuristic`。

## 14. 不做（YAGNI 重申）

推論時網路 / embedding API、重量級 ML runtime、更動 API、動 v1 scorer、v1 §15 其餘項目（named pool、外部模型表、串流、auth、rate limiting）。

## 15. 實作階段決定的常數 / 待定（給定起始預設以確保規格完整）

- **特徵最終清單**與 hashed char-ngram 的 bucket 數 `K`（起始：§4 表 + `K = 16`）。
- **難度標註 rubric**（1–5 各級的定義）與 LLM prompt 模板。
- **邏輯迴歸超參數**：學習率、迭代數、L2 正則化強度、隨機 seed（起始：lr=0.1、iters=1000、L2=1e-3、seed 固定）。
- **語料規模與類別配比**（起始：~500 筆，七類別大致均衡）。
- **eval 指標**與「勝過 heuristic」門檻（起始：(a) 對標註的 Spearman 相關 + 序數準確率；(b) 固定充分率下的平均成本 vs `always-strongest`；learned 須兩者皆嚴格優於 heuristic）。
- **出貨權重格式**：生成的 `.rs` `const`（起始選擇）vs `include_str!` JSON。

## 16. 決議與延後事項（複審回饋）

1. **難度標註 LLM**：**延後決定**——等其他細節確定後再選。初期改用合成時賦予的目標難度（§8），不阻塞其餘開發。
2. **語料來源**：**以合成為主**（§8）。
3. **eval 指標**：**加入「固定品質下的成本節省」proxy**，與對標註的擬合度並列為採用門檻（§12、§15）。
