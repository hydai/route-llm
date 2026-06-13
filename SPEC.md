# route-llm — 設計規格 (v1)

> 狀態：v1 設計定稿，待實作
> 日期：2026-06-13

## 1. 專案概述

`route-llm` 是一個用 Rust 撰寫的 HTTP 服務。使用者送入「一個 LLM 請求」與「一份候選 LLM 清單」，服務在**不實際呼叫任何 LLM** 的前提下，預測並回傳這份清單的**推薦排序**。

核心是一個受 [RouteLLM](https://github.com/lm-sys/RouteLLM) 啟發的**難度導向 router**：先估計 query 的難度，再用「成本-品質權衡」對候選模型排序。產品定位是一個**飛行前的選模器 (pre-flight model picker)** —— client 先問 route-llm「這個 query、這些候選，選哪個？」，拿到推薦後再自行去呼叫選中的模型。

### 與 RouteLLM 的關係

- **借鑑**：RouteLLM 對每個 query 預測一個「難度 / win-rate」分數，用一個 threshold 在強/弱模型間做成本取捨。我們沿用「依難度路由以省成本」這個精神。
- **差異 1（推廣）**：RouteLLM 是「二選一」（strong vs weak）；我們是「排序 N 個模型」。
- **差異 2（不呼叫）**：RouteLLM 是 proxy，會真的呼叫選中的模型並回傳 completion；我們是**預測式**，只回推薦、不呼叫。
- **第一種策略**：v1 採用「啟發式難度 router」。未來的策略（similarity-weighted、matrix factorization 等）透過 `Router` trait 擴充。

## 2. 目標與非目標

### 目標 (v1)

1. 提供 HTTP 服務，輸入 query + 候選清單，輸出推薦排序、每個模型的分數、估計難度、與簡短理由。
2. 用純 Rust 的啟發式難度估計，零外部相依、零網路、可完整單元測試。
3. 內建模型能力/成本表，並允許使用者在請求中覆寫。
4. 提供三種請求方言：原生 `/v1/recommend`、OpenAI 形狀 `/v1/chat/completions`、Anthropic 形狀 `/v1/messages`，三者共用同一個排序核心。
5. 成本-品質權衡可由請求中的 `cost_bias` 旋鈕調整。

### 非目標 (v1，YAGNI)

- 不實際呼叫任何 LLM、不串接任何 provider。
- 不使用 embedding / ML / 訓練好的權重（留給未來 router 策略）。
- 不做認證、rate limiting、persistence/DB、串流 (SSE)。
- 不做伺服器端「named pool」設定、不從外部檔案載入模型表（留給未來）。

## 3. 系統架構

採用 Cargo workspace，把「純排序邏輯」與「HTTP 層」隔離，使核心能用同步、零 I/O 的測試覆蓋。

```
route-llm/
├── Cargo.toml                    # workspace；含 [profile.dev.package."*"] debug=false
├── crates/
│   ├── core/                     # package: route-llm-core (lib)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs          # ModelProfile, RoutingPreferences, Recommendation, Difficulty, RankedModel
│   │       ├── difficulty.rs     # 啟發式難度估計器
│   │       ├── registry.rs       # 內建模型表 + 覆寫合併 + 解析
│   │       ├── ranker.rs         # 成本-品質排序
│   │       └── router.rs         # Router trait + HeuristicRouter
│   └── server/                   # package: route-llm-server (bin)，binary 名稱 route-llm
│       └── src/
│           ├── main.rs           # axum 啟動、路由表、tracing 初始化
│           ├── error.rs          # ApiError → HTTP 狀態碼/JSON 對映
│           ├── dto.rs            # 三方言的 request/response serde 結構
│           └── handlers.rs       # 三個 handler：抽 query + 蒐集候選 → Router::recommend
```

### 資料流

```
                 ┌─ POST /v1/recommend         {query, models, preferences}
request ────────-┼─ POST /v1/chat/completions  {model?, messages, models(ext), preferences(ext)}
                 └─ POST /v1/messages          {model?, system?, messages, models(ext), preferences(ext)}
                          │  各方言 handler：抽出 query 文字 + 蒐集候選清單 + 解析 preferences
                          ▼
            registry.resolve(候選 ids, 覆寫) ──► Vec<ModelProfile>
                          │
            Router::recommend(query, &profiles, &prefs)
                          │  difficulty.score(query) ──► Difficulty
                          │  ranker.rank(difficulty, profiles, prefs) ──► Vec<RankedModel>
                          ▼
                   Recommendation { difficulty, ranking }
                          │  各方言 handler：包裝成對應回應形狀
                          ▼
                   HTTP 200 + JSON
```

## 4. 核心領域型別 (`core/src/model.rs`)

```rust
/// 一個候選模型的能力/成本輪廓；quality 與 cost 都正規化到 0.0..=1.0。
pub struct ModelProfile {
    pub id: String,
    pub quality: f64, // 能力分數，0..1，越高越強
    pub cost: f64,    // 相對成本，0..1，越高越貴
}

/// 排序偏好（旋鈕）。
pub struct RoutingPreferences {
    pub cost_bias: f64, // 0.0 = 品質優先, 1.0 = 成本優先；預設 0.5
}
impl Default for RoutingPreferences { /* cost_bias: 0.5 */ }

/// query 難度估計結果。
pub struct Difficulty {
    pub score: f64,            // 0..1
    pub signals: Vec<String>,  // 觸發的特徵名（供解釋）
}

/// 單一模型的排序結果。
pub struct RankedModel {
    pub id: String,
    pub score: f64,     // 排序分數（可為負）
    pub reason: String, // 人類可讀理由
}

/// router 的最終輸出。
pub struct Recommendation {
    pub difficulty: Difficulty,
    pub ranking: Vec<RankedModel>, // 由高分到低分排序
}
```

## 5. 難度估計器 (`core/src/difficulty.rs`)

純函式 `fn score(query: &str) -> Difficulty`，無 I/O、可決定性（deterministic）。從 query 抽取一組特徵，每個特徵貢獻加權證據，加總後用 logistic 函式壓到 0..1。觸發的特徵名收進 `signals`。

**特徵與預設權重**（這些常數是可調的；見 §16）：

| 特徵 | 偵測方式 | 預設權重 | signal 名稱 |
|------|----------|----------|-------------|
| 基礎偏置 | 常數 | `-1.0` | （不列入 signals） |
| 長度 | 估計 token 數（約 `len/4`），每 token `+0.0010`，上限 `+1.2` | 可變 | `long_form`（當貢獻 > 0.3） |
| 程式碼 | 含 ` ``` ` 框或常見程式關鍵字 | `+1.0` | `code` |
| 數學/LaTeX | `$...$`、`\frac`、`∑`、`∫` 等符號 | `+0.8` | `math` |
| 推理關鍵字 | prove / derive / step by step / analyze / design / explain why / optimize / compare（中英對照） | 每命中 `+0.5`，上限 `+1.5` | `reasoning` |
| 多段約束 | 編號清單 ≥ 3 項，或多個問句 | `+0.6` | `multi_constraint` |
| 結構化輸出 | 要求 JSON / table / schema / 特定格式 | `+0.4` | `structured_output` |
| 解釋請求 | explain / 說明 / 為什麼 / how does | `+0.4` | `explanation_request` |

`score = sigmoid(sum_of_weights)`，其中 `sigmoid(x) = 1 / (1 + e^-x)`。

> 實作備註：難度特徵的權重與偵測規則，是「商業邏輯有多種合理解」之處，將在實作階段由專案擁有者親自撰寫（約 5–10 行核心邏輯）。上表為起始預設值。

## 6. 模型表 (`core/src/registry.rs`)

內建一份常見模型 → `ModelProfile` 的對照表，能力分數預先正規化到 0..1。請求中提供的 `quality` / `cost` 會**覆寫**內建值。

**解析規則** `resolve(candidates) -> Result<Vec<ModelProfile>, Vec<String>>`：

1. 對每個候選項：
   - 若請求項同時提供 `quality` 與 `cost` → 直接採用（即使內建表沒有此 id）。
   - 否則查內建表：找到 → 採用內建值（請求若只覆寫其一，則該欄覆寫、另一欄用內建）。
   - 找不到且未提供完整覆寫 → 收進「未知清單」。
2. 若「未知清單」非空 → 回 `Err(unknown_ids)`（handler 轉成 400）。
3. 去重：同一 id 出現多次時以最後一筆為準。

**內建種子值（示意，非權威 benchmark，需校準；見 §16）：**

| id | quality | cost |
|----|---------|------|
| claude-opus-4-8 | 0.97 | 0.90 |
| claude-sonnet-4-6 | 0.90 | 0.45 |
| claude-haiku-4-5 | 0.75 | 0.12 |
| gpt-4o | 0.88 | 0.50 |
| gpt-4o-mini | 0.62 | 0.10 |
| gemini-1.5-pro | 0.85 | 0.40 |

## 7. 排序器 (`core/src/ranker.rs`)

`fn rank(difficulty: &Difficulty, profiles: &[ModelProfile], prefs: &RoutingPreferences) -> Vec<RankedModel>`。

**核心公式：**

```
required_capability = difficulty.score
adequacy(m)         = sigmoid(K · (m.quality − required_capability))   // 模型「夠不夠用」, K = 8.0
λ                   = prefs.cost_bias                                  // 0..1
score(m)            = adequacy(m) − λ · m.cost
```

- **adequacy** 表示「模型能力是否足以可靠處理此難度」：`quality ≫ 難度` 時趨近 1，`quality ≪ 難度` 時趨近 0。
- **λ·cost** 是成本懲罰；`cost_bias` 越高，成本越被在意。
- 直覺驗證：
  - **簡單 query**（難度低）→ 多數模型 adequacy 接近 1 → 成本項主導 → 便宜且夠用的模型排前。
  - **困難 query**（難度高）→ 只有高能力模型 adequacy 撐得住 → 強模型排前（即使較貴）。
- **重要 caveat（成本傾向）**：因為 `λ = cost_bias` 直接乘上 cost，在預設 `cost_bias = 0.5` 下成本懲罰相當強。當一個較便宜的模型 `quality` 恰好略高於難度（adequacy 已接近飽和）時，它可能在**中等難度**就勝過更強但更貴的模型。這是省成本的預期行為；若要更偏品質，調低 `cost_bias`。

**排序與 tie-break**：先依 `score` 由大到小；同分時依序比較 `quality` 大者優先、`cost` 小者優先、`id` 字典序，確保決定性。

**理由產生**：依主導因素產生 `reason`：

- `adequacy(m) < 0.5` → 「能力可能不足以可靠處理此難度 (difficulty {d})」。
- 難度高（≥ 0.6）且為最高 quality → 「高難度，最強模型最可靠」。
- 難度低（< 0.4）且為 adequate 中最便宜 → 「低難度，便宜且足夠」。
- 其餘 → 「在品質與成本間取得平衡」。

> 實作備註：`K`、`λ` 的對映、理由門檻同樣是專案擁有者將親自撰寫的核心邏輯（約 5–10 行）。上述為起始預設。

## 8. Router trait (`core/src/router.rs`)

```rust
pub trait Router {
    fn recommend(
        &self,
        query: &str,
        models: &[ModelProfile],
        prefs: &RoutingPreferences,
    ) -> Recommendation;
}

/// v1 的第一種策略。
pub struct HeuristicRouter; // 之後可帶設定欄位（K、權重等）

impl Router for HeuristicRouter {
    fn recommend(&self, query, models, prefs) -> Recommendation {
        let difficulty = difficulty::score(query);
        let ranking = ranker::rank(&difficulty, models, prefs);
        Recommendation { difficulty, ranking }
    }
}
```

## 9. HTTP API (`server/`)

框架：axum。三個方言 handler 各自只負責「抽 query 文字 + 蒐集候選 + 解析 preferences」，然後呼叫同一個 `Router::recommend`，最後包裝成對應回應形狀。

### 共用語意

- **候選清單**來自非標準擴充欄位 `models: [...]`（OpenAI/Anthropic SDK 用 `extra_body` 送）。
- OpenAI/Anthropic 的標準 `model` 欄（若有）→ 視為「也請納入考慮」的提示，若不在 `models` 內則併入候選集。
- 候選集為空（`models` 與 `model` 皆缺）→ 400 `empty_candidates`。
- query 文字為空 → 400 `empty_query`。
- `cost_bias` 不在 0..=1 → 400 `invalid_preferences`。
- 任一候選 id 無法由內建表或覆寫解析 → 400 `unknown_models`，`details.unknown` 列出 id。

> **範例數值說明**：§9 內所有數值（難度、分數）皆為示意，依 §7 公式與 §16 預設常數計算，並採各範例顯示的 `cost_bias`；調整 §16 常數後數值會變動。以下範例刻意採用一個會算出 `difficulty ≈ 0.71` 的困難 query，並用 `cost_bias = 0.3`（偏品質）以同時展示三種理由分支。提醒：在預設 `cost_bias = 0.5` 下，router 更積極省成本，較便宜且「剛好夠用」的模型可能在中等難度時勝過更強的模型（見 §7 caveat）。這組數值也作為實作時的測試向量。

### 9.1 `GET /health`

→ `200 {"status":"ok"}`。

### 9.2 `GET /v1/models`

→ `200` 內建模型表：`{"models":[{"id","quality","cost"}, ...]}`。

### 9.3 `POST /v1/recommend`（原生）

**Request**
```jsonc
{
  "query": "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition.",
  "models": [
    {"id": "claude-opus-4-8"},
    {"id": "claude-haiku-4-5"},
    {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
  ],
  "preferences": {"cost_bias": 0.3}   // 可省略，預設 0.5；此範例用 0.3（偏品質）
}
```

**Response 200**（原生乾淨形狀，無外殼）
```jsonc
{
  "difficulty": {"score": 0.71, "signals": ["reasoning","explanation_request"]},
  "ranking": [
    {"id": "claude-opus-4-8", "score": 0.62, "reason": "高難度，最強模型最可靠"},
    {"id": "claude-haiku-4-5","score": 0.54, "reason": "在品質與成本間取得平衡"},
    {"id": "gpt-4o-mini",     "score": 0.19, "reason": "能力可能不足以可靠處理此難度 (difficulty 0.71)"}
  ]
}
```

### 9.4 `POST /v1/chat/completions`（OpenAI 形狀）

**Request**
```jsonc
{
  "model": "gpt-4o-mini",                 // 可選；併入候選集
  "messages": [{"role": "user", "content": "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition."}],
  "models": [                              // 擴充 (extra_body)：候選清單
    {"id": "claude-opus-4-8"},
    {"id": "claude-haiku-4-5"},
    {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
  ],
  "preferences": {"cost_bias": 0.3}        // 擴充，可選
}
```

**Response 200**（completion 外殼；`model` = #1 推薦）
```jsonc
{
  "id": "rec-01H...",
  "object": "chat.completion",
  "model": "claude-opus-4-8",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Recommended: claude-opus-4-8 (difficulty 0.71). Order: claude-opus-4-8 > claude-haiku-4-5 > gpt-4o-mini."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "route_llm": {
    "difficulty": {"score": 0.71, "signals": ["reasoning","explanation_request"]},
    "ranking": [
      {"id": "claude-opus-4-8", "score": 0.62, "reason": "高難度，最強模型最可靠"},
      {"id": "claude-haiku-4-5","score": 0.54, "reason": "在品質與成本間取得平衡"},
      {"id": "gpt-4o-mini",     "score": 0.19, "reason": "能力可能不足以可靠處理此難度 (difficulty 0.71)"}
    ]
  }
}
```

`usage` 全歸零以誠實反映「未呼叫模型」。`content` 明示為「推薦」而非答案。

### 9.5 `POST /v1/messages`（Anthropic 形狀）

**Request**
```jsonc
{
  "model": "claude-haiku-4-5",            // 可選；併入候選集
  "system": "You are concise.",            // 可選；併入 query 文字
  "messages": [{"role": "user", "content": "Explain why the Paxos consensus algorithm guarantees safety, derive the invariant it maintains, and analyze step by step how it handles a network partition."}],
  "models": [
    {"id": "claude-opus-4-8"},
    {"id": "claude-haiku-4-5"},
    {"id": "gpt-4o-mini", "quality": 0.55, "cost": 0.10}
  ],
  "preferences": {"cost_bias": 0.3}        // 擴充，可選
}
```

**Response 200**（messages 外殼；`model` = #1 推薦）
```jsonc
{
  "id": "rec-01H...",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-4-8",
  "content": [{"type": "text", "text": "Recommended: claude-opus-4-8 (difficulty 0.71). Order: claude-opus-4-8 > claude-haiku-4-5 > gpt-4o-mini."}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "route_llm": { /* difficulty + ranking，同 §9.4 */ }
}
```

### query 文字抽取

- OpenAI：串接所有 `messages[].content` 的文字（以換行分隔）。
- Anthropic：先放 `system`（若有），再串接 `messages[].content`。
- 原生：直接用 `query` 欄。

## 10. 錯誤處理 (`server/src/error.rs`)

統一錯誤回應結構：
```jsonc
{ "error": { "code": "unknown_models", "message": "...", "details": { "unknown": ["foo"] } } }
```

| code | HTTP | 觸發 |
|------|------|------|
| `invalid_json` | 400 | body 無法解析 |
| `empty_query` | 400 | 抽取後 query 為空 |
| `empty_candidates` | 400 | 候選集為空 |
| `unknown_models` | 400 | 有無法解析的候選 id（`details.unknown` 列出） |
| `invalid_preferences` | 400 | `cost_bias` 超出 0..=1 |

## 11. 設定

- 監聽埠：環境變數 `ROUTE_LLM_PORT`（預設 `8080`）。
- 監聽位址：環境變數 `ROUTE_LLM_HOST`（預設 `0.0.0.0`）。
- 內建模型表：v1 編譯進二進位檔（未來再支援外部檔案）。
- 日誌等級：`RUST_LOG`（透過 `tracing-subscriber` 的 EnvFilter）。

## 12. 相依套件

執行期（精簡）：
- `axum`、`tokio`（`rt-multi-thread`, `macros`）
- `serde`（derive）、`serde_json`
- `tracing`、`tracing-subscriber`（EnvFilter）
- `thiserror`

開發/測試：
- `axum-test`（HTTP 整合測試）

`Cargo.toml` 遵守專案 Rust 規範：
```toml
[profile.dev.package."*"]
debug = false
```
一律使用 release build（`cargo build --release`）。

## 13. 測試策略（TDD）

採測試先行。

- **core/difficulty**：表格驅動。給定 query → 斷言觸發的 signals 與 score 落在預期區間（如「含程式碼框的 query 難度應 > 0.5」）。
- **core/ranker**：性質測試。
  - 簡單 query（難度低）+ 兩個皆 adequate 的模型 → 便宜者排前。
  - 困難 query（難度高）→ 高 quality 者排前。
  - `cost_bias=0`（品質優先）vs `cost_bias=1`（成本優先）產生不同順序。
  - 排序具決定性（tie-break 可重現）。
- **core/registry**：覆寫合併、未知 id 回 `Err`、去重。
- **server/handlers**：用 `axum-test` 對三方言各驗一條 happy path（200 + 正確外殼）與各種 4xx。
- **server**：OpenAI/Anthropic/原生三方言對同一 query+候選，`route_llm.ranking` 應一致（共用核心）。§9 的範例可直接作為跨方言一致性的測試向量。

## 14. 不做（YAGNI，重申）

認證、rate limiting、實際呼叫 LLM、provider 串接、embedding/ML/訓練權重、DB/persistence、串流 (SSE)、named server-side pools、從外部檔案載入模型表。

## 15. 未來工作

- 更多 `Router` 策略：similarity-weighted ranking（embedding + 參考資料集，最忠於 RouteLLM 且天生支援 N 模型）、matrix factorization、classifier。
- 在 `model` 欄支援 named pool（伺服器端預設候選集）。
- 從外部檔案（TOML/JSON）載入並熱更新模型表。
- completion 外殼的串流 (SSE) 形狀（若有 client 需要）。

## 16. 實作待定常數（起始預設值，可調）

這些是「商業邏輯有多種合理解」、將由專案擁有者在實作階段親自撰寫的核心參數；此處給定起始預設以確保規格完整：

- 難度特徵權重：見 §5 表格。
- 難度 logistic：`score = sigmoid(sum)`。
- 排序 `K = 8.0`；`λ = cost_bias`。
- 理由門檻：`adequacy < 0.5` 視為能力不足；難度 `≥ 0.6` 為高、`< 0.4` 為低。
- 模型表種子值：見 §6（示意值，需校準）。

### 範例驗算（§9 一致性檢查）

以 §9 範例（`difficulty = 0.71`、`cost_bias = 0.3`、`K = 8`）代入 §7 公式：

| 模型 | quality | cost | adequacy = σ(8·(q−0.71)) | score = adequacy − 0.3·cost |
|------|---------|------|--------------------------|------------------------------|
| claude-opus-4-8 | 0.97 | 0.90 | σ(2.08) ≈ 0.889 | ≈ 0.62 |
| claude-haiku-4-5 | 0.75 | 0.12 | σ(0.32) ≈ 0.579 | ≈ 0.54 |
| gpt-4o-mini | 0.55 | 0.10 | σ(−1.28) ≈ 0.218 | ≈ 0.19 |

排序 `opus > haiku > gpt-4o-mini`，與 §9 一致。
