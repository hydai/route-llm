# route-llm v2.1 — 設計規格：真實難度標註（本地 LLM）

> 狀態：v2.1 設計定稿，待複審 → 待實作
> 日期：2026-06-14
> 關係：本規格在 v2（見 `SPEC-v2.md`）之上做**加法**——只換「訓練標註」與「語料」，不動推論路徑、不動 v1。

## 1. 概述與動機

v2 造好了完整、正確、測試齊全的學習式 router，但 `eval`（修正 Spearman tie 後）顯示 **learned ≈ heuristic**，贏不了。根因不在模型，而在**教材**：

`crates/trainer/src/corpus.rs` 對**同一類別的每一題都標同一個固定難度**（所有 `code` = 0.65、`reasoning` = 0.88…），而且那些數字是照 heuristic 的直覺挑的。等於拿「heuristic 的答案」當標準答案訓練——模型最多學成 heuristic 的複製品，**數學上不可能超越**。

例：以下兩題現在**都被標 0.65**——`Debug this: let x: i32 = "s";`（一行型別錯誤，極易）與 `Implement a lock-free concurrent hash map with ABA mitigation`（極難）。heuristic 也分不出，**「同類別內的難易差異」正是 heuristic 看不到、學習式模型有機會抓到的訊號**。

v2.1 的修法：**讓本地 LLM（Ollama）逐題打難度分數**，取代照抄 heuristic 的固定標籤，並把語料擴展到 ~1000 題以提供足夠且多樣的訊號；重新 `fit`、`eval`，由結果決定預設策略。

## 2. 目標與非目標

### 目標 (v2.1)

1. 新增 `trainer label` 子指令：呼叫**本地 Ollama**，對每題 query 逐題評難度，產出真實、逐題的 `labeled.jsonl`。
2. 把 `synth` 從「固定清單」改為**決定性組合式產生器**，產出 ~1000 題、含同類別內難易差異的 query（**只出題、不附難度**）。
3. 重新 `fit` 產生新的內嵌 `weights.rs`；`eval` 比較 learned vs heuristic vs always-strongest。
4. **eval-driven 完成定義**：由留出集結果決定伺服器預設 router（learned 贏則維持 `learned`，輸則誠實切回 `heuristic`），並提交可重現的結論。
5. 標註步驟全程**本地、零外部網路**（只連 `localhost` 的 Ollama）；推論路徑維持零網路。

### 非目標 (v2.1，YAGNI)

- **不**改推論路徑的網路特性：`core`/`server` 推論仍零網路、零新相依。
- **不**用雲端 / API-key LLM；只用本地 Ollama。
- **不**改 API、DTO、`Recommendation` 形狀，**不**新增端點。
- **不**動 v2 的 `features` / `model` / `ranker` / `LinearModel`（只有內嵌 `weights.rs` 因更好的標註而重新生成）。
- **不**動 v1（凍結）。
- **不**做 §15 其餘未來工作（named pool、外部模型表、串流、auth…）。

## 3. 架構與管線

trainer 是一條四站生產線；v2.1 只新增/改動前兩站，後兩站不動。

```
synth（出題）   → data/corpus.jsonl     query + 類別，「不附難度」；組合式產生 ~1000 題
   │                                     （決定性、離線、可重現）
   ▼
label（評分）★  → 讀 corpus.jsonl，逐題呼叫本地 Ollama（localhost）→ 1–5 分 → 映射 [0,1]
   │              → 寫 data/labeled.jsonl（真實逐題標註）
   │              → 寫 data/label_cache.jsonl（以 query+model 的 hash 為 key 的快取）
   ▼
fit（訓練）      → 讀已提交的 labeled.jsonl → learned::features → 擬合 → crates/core/src/learned/weights.rs
   │              （不變；只是吃到更好的標註）
   ▼
eval（驗收）     → learned vs heuristic vs always-strongest（Spearman / ordinal / 固定充分率成本）
```

**職責分離**：`synth`（出題）離線、決定性；`label`（評分）會變、需本地模型。分開後 `labeled.jsonl` 成為**凍結、可重現的訓練輸入**（沿用 SPEC-v2 §8）。

**網路界線**：`reqwest`（blocking）只進 `crates/trainer`；`label` 是唯一會連線的步驟，且只連 `localhost`。`core`/`server`/推論完全不受影響。

### 模組變更（皆在 `crates/trainer`）

- `src/label.rs`（**新**）：Ollama HTTP client、rubric prompt、解析、快取、`label` 子指令進入點。
- `src/corpus.rs`（**改**）：組合式產生器；`synth` 改寫 queries-only 的 `corpus.jsonl`。
- `src/main.rs`（**改**）：`label` arm 從 stub 改為呼叫 `label::run()`；usage 字串更新。
- `Cargo.toml`（**改**）：新增 `reqwest`（`blocking`, `json` features）；（可能）`sha2` 供快取 hash。

## 4. 語料產生 (`corpus.rs` 的組合式 `synth`)

- 每個類別定義一組**難度階梯句型**（slot 樣板，大致由易到難）× 一組**參數池**（主題、語言、資料結構、限制條件…）。以巢狀迴圈**決定性枚舉**組合出 query。
- 目標總量 **~1000 題**，各類別大致均衡，且**每類別都同時含簡單與困難**的樣板（提供同類別內難易差異）。
- 輸出 `corpus.jsonl`：每行 `{ "query": ..., "category": ... }`，**無 difficulty 欄位**。
- 決定性：同程式 → 同 1000 題；`corpus.jsonl` 可重現並提交。
- 仍是 ★ 擁有者可調點：句型與參數池由擁有者撰寫 / 擴充（SPEC-v2 §15 精神）。

## 5. 標註 (`label.rs`)

- **設定（env，僅 label 步驟）**：
  - `ROUTE_LLM_LABEL_URL`（預設 `http://localhost:11434`）
  - `ROUTE_LLM_LABEL_MODEL`（預設 `google/gemma-4-31b-qat`）
  - `ROUTE_LLM_LABEL_CONCURRENCY`（預設 `1`；可調小並發加速，見 §10）
- **呼叫**：對每題 query，POST 到 Ollama（`/api/generate`，`stream:false`，`options.temperature = 0` 以求穩定/近決定性）。
- **Rubric（1–5 → [0,1]）**：prompt 要求模型對「處理此 query 的難度」給 **1–5 整數評分**（1 = 瑣碎閒聊；5 = 專家級多步推理）外加一句簡短理由，並以可解析格式輸出（如 `RATING: <n>`）。映射 `difficulty = (n − 1) / 4` → {0.0, 0.25, 0.5, 0.75, 1.0}。
  - 離散刻度對較弱標註者更穩定；31B QAT 模型品質足夠，雜訊低。
- **解析與容錯**：抽出 1–5 整數；解析失敗 → 重試一次；仍失敗 → **跳過該題並記錄**（不寫入壞標註，避免污染）。
- **快取**：`data/label_cache.jsonl`，key = `hash(query + model)`，value = rating。`label` 執行時：命中快取則沿用，否則呼叫 Ollama 並寫入快取。換模型（key 含 model）會自然失效。
- **輸出**：`labeled.jsonl`，每行 `{ "query", "difficulty"(來自 LLM), "category" }`，供 `fit`/`eval` 使用（格式與 v2 相同）。

## 6. 可重現性與資料 (`data/`)

- `corpus.jsonl`：queries-only，組合式產生、決定性、提交。
- `labeled.jsonl`：LLM 逐題標註，**提交**，是 `fit` 的**凍結、可重現輸入**（對固定 `labeled.jsonl`，`fit` 決定性 → 相同 `weights.rs`）。
- `label_cache.jsonl`：提交，使重標增量化、重跑便宜。
- **可重現性模型**：把「不可重現的那一刻」（LLM 標註）凍結成提交的 `labeled.jsonl`；重標是顯式、偶發的步驟。網路永不進入 `fit` / 推論 / CI。

## 7. fit / eval（沿用 v2，不改邏輯）

- `fit`：不變——讀 `labeled.jsonl`、抽特徵、標準化、擬合邏輯迴歸、輸出 `weights.rs`。
- `eval`：不變——80/20 留出切分，報 learned vs heuristic vs always-strongest 的 Spearman、ordinal accuracy、固定充分率（adequacy ≥ 門檻）下的平均成本。

## 8. 驗收與預設切換（完成定義）

- 重標 + `fit` 後執行 `eval`，取得留出集上的 learned vs heuristic 三項指標。
- **判定規則（勝出）**：`Spearman(learned) ≥ Spearman(heuristic)` **且** `ordinal(learned) ≥ ordinal(heuristic)` **且** 固定充分率下 `cost(learned)` 不差於 `cost(heuristic)`。
  - 勝出 → 維持伺服器預設 `learned`（`choose_router` 不動）。
  - 未勝出 → 把伺服器預設切回 `heuristic`（`crates/server/src/main.rs` 的 `choose_router` 一行：未設定時回 `heuristic` 而非 `learned`），並記錄結論。
- **完成 = 一份可重現、有證據的結論**（不論勝負）；結論連同 `eval` 數據提交（短記於本檔 §16 或 PR 描述）。

## 9. 錯誤處理

- **推論無新錯誤面**：`core`/`server` 不變。
- `label`：Ollama 無法連線 → 清楚錯誤（提示「請先啟動 ollama 並 pull 模型」），非零退出；模型輸出無法解析 → 重試一次後跳過並記錄。
- `synth`/`fit`/`eval`：沿用既有錯誤處理。

## 10. 相依與效能

- **`crates/trainer`**：新增 `reqwest`（`blocking` + `json`；blocking 因 trainer 是簡單 CLI，無需 async）、（可能）`sha2`。**全部限定於 trainer**；`core`/`server` 不新增任何相依，推論零網路不變。
- **效能**：以本地 31B QAT 逐題標 ~1000 題是一次性離線工作（視硬體數十分鐘~數小時），有快取故重跑便宜。`ROUTE_LLM_LABEL_CONCURRENCY > 1` 可對 Ollama 發數個並行請求加速；預設循序（安全）。
- 維持 `[profile.dev.package."*"] debug = false`；release build。

## 11. 測試策略 (TDD)

- `label.rs`：單元測試**純函式部分**——rubric 解析（`RATING: n` → 1–5 → [0,1]；壞輸出處理）、映射、快取命中/寫入邏輯；**以 mock / 不實際連線** Ollama 的方式測。
- **真實 Ollama 呼叫**：integration-only，以 `#[ignore]`（或 env gate）標記，**不進 `cargo test` / CI**（CI 無 daemon、無網路）。
- `corpus.rs`：組合式產生器測試——總量達標（~1000）、各類別非空、含易與難、決定性、`corpus.jsonl` 為 queries-only。
- `fit`/`eval`：沿用；`fit` 對固定 `labeled.jsonl` 決定性。
- 推論側（core/server）測試不變且維持全綠。

## 12. 向後相容 / 隔離

- 推論路徑不變，唯一可能變動是 §8 的「預設 router 一行切換」（依驗收結果）。
- v1 凍結；v2 的 `features`/`model`/`ranker`/`LinearModel` 不動——只有內嵌 `weights.rs` 重新生成。
- `synth` 輸出形狀改變（`corpus.jsonl` 不再附合成難度）→ 同步更新其測試。
- `core`/`server` 的 `Cargo.toml` 不變（無新相依）。

## 13. 不做（YAGNI）

雲端 / API-key LLM、推論時網路、改 API/端點、動 v1 或 v2 的推論核心、§15 其餘未來工作。

## 14. 待定 / 起始預設（可調）

- 預設標註模型 `google/gemma-4-31b-qat`、Ollama URL `localhost:11434`（皆 env 可覆寫）。
- rubric：1–5 → `(n−1)/4`；prompt 模板與分級定義於實作階段定稿。
- 語料規模 ~1000、各類別配比與「難度階梯句型 × 參數池」的具體內容（★ 擁有者撰寫）。
- 標註並發預設 1；逾時 / 重試次數於實作定稿。
- 快取 key 雜湊用 `sha2`（或等價）。

## 15. 開放問題

1. prompt 模板的精確措辭與 1–5 分級定義（影響標註品質，實作階段可迭代）。
2. 若 31B 標註仍無法讓 learned 勝出，是否要在 v2.1 內嘗試「更豐富特徵」或留待 v2.2（目前 §8 採「誠實切回 heuristic + 記錄」）。

## 16. 驗收結論（實作後填入）

> 待 `eval` 執行後填入：learned vs heuristic 的 Spearman / ordinal / 成本數據，勝負判定，與最終預設 router 決定。
