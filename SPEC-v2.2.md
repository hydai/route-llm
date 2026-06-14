# route-llm v2.2 — 設計規格：標籤獨立的人工黃金集（可信判決）

> 狀態：v2.2 已實作並驗收（見 §16 結論）;PR #10 待 review/merge。
> 日期：2026-06-14（設計）/ 2026-06-15（實作 + 驗收）
> 關係：本規格在 v2.1（見 `SPEC-v2.1.md`）之上做**加法**——新增「標籤獨立的人工黃金集」與「跨 labeler 評估」,用以**權威重決出貨**;不動推論路徑、不動 v1/v2 核心、零網路、零新相依。

## 1. 概述與動機

v2.1 出貨了學習式 router（以 codex 標註擬合），判決為「learned 三方一致勝出」。但那個判決是**自我參照**的：

每個 router 的 Spearman / ordinal 都是**對它自己那套 labeler 的標籤**評分（見 SPEC-v2.1 §16「各指標 vs 該套自身標籤」）。這量的是「該套標籤有多**可學/自洽**」,**不是**「相對某個外部真值有多**正確**」。三套標籤裡唯一**標籤無關**的欄位只有 `avg_cost`（同一批 holdout query、不同 router）。

換句話說：我們證明了 learned 能很好地**複製每個 labeler**,但還沒有一把**獨立於受測模型**的尺去問「到底誰更接近**人類**判斷」。v2.1 §16 也記錄了 Claude 與 Codex **逐題僅 85.5% 完全一致**——剩下的分歧正是「難度估計真的困難」之處,而我們從未請人類裁決過。

v2.2 的修法：**建一把人工黃金尺**,並把稀缺的人工判斷**集中在最能區分 router 的地方**——兩個強 labeler（claude、codex）**真正分歧**的查詢——然後**只在這些人類判過的題上**評分。以此**權威重決** v2.1 的兩個出貨決定。

### 為何聚焦「claude≠codex 的分歧」且排除 gemma

對現有三套標籤實測。三方共同 986 題的分布：

| 分歧來源 | 數量 | 性質 |
|---|---|---|
| 三方完全一致 | 716（72.6%） | 簡單題,每個 router 都對 → **無鑑別力**,只稀釋訊號 |
| 有分歧（非三方一致） | 270（27.4%） | |
| └ claude==codex,僅 gemma 不同 | 128 | 多為 **gemma 雜訊**（兩個強模型已一致） |
| └ **claude ≠ codex** | **142** | **真正模稜兩可**的難題 → **最能區分 router** |

排除 gemma 後,gold 池 = **claude≠codex**（claude/codex 各 987 題,直接兩兩比對）= **143 題（14.5%）**。（與上表 142 差 1 題：gemma 缺第 564 題,故三方共同少 1;以 claude–codex 比對不受此影響。）

- 排除 gemma 把人工裁決範圍從 270（三方分歧）縮到 **143**（claude≠codex）：丟掉的主要是 ~128 題 gemma 雜訊,留下兩個強模型的真分歧。
- 這 143 題**全部只差一級**（difficulty 相差 0.25）→ 人工裁決快（在兩個相鄰值間二選一）。
- 類別分布：**math 80、multilingual 32、reasoning 16、code 15**;**chat / extraction 完全沒有分歧**（簡單類別兩強模型都同意）。gold 集自動聚焦在難度估計真正困難的地方。

`learned` 用的 `LinearModel` **容量極低**（少數特徵權重、無 per-query 參數）,數學上**無法背下個別查詢**;因此即使這 143 題曾出現在訓練語料中,用**人類標籤**評分仍公平——任何 router 都從未見過人類標籤。故 v2.2 **無需為防洩漏而重 fit**。

## 2. 目標與非目標

### 目標 (v2.2)

1. 新增 `trainer gold-pool` 子指令：從 `labeled.claude.jsonl` 與 `labeled.codex.jsonl` 決定性算出「claude≠codex」的 143 題,輸出**無標籤**待判清單 `data/gold.unlabeled.jsonl`（只給題目,盲判用）。
2. 由**擁有者（人類）盲判**這 143 題難度（沿用 v2.1 的 1–5 rubric）,提交為 `data/gold.jsonl`——一把**標籤獨立**的黃金尺。
3. 擴充 `eval`：`eval --gold <file>` 與 `compare --gold <file> <labeled...>`,讓所有候選 router（learned-fit-各 labeler、heuristic）對**同一套人類標籤**評分（Spearman / ordinal）。
4. 新增 `trainer crosseval`：fit 在 labeler A、評 B 的 holdout,產出跨 labeler 泛化矩陣（零新標註的診斷）。
5. **gold-driven 重決出貨**：依 gold 上的結果**權威重決**兩軸——(a) 伺服器預設 `learned` vs `heuristic`、(b) 出貨哪個 labeler 的 `weights.rs`;與 v2.1 不同則重 `fit` / 翻預設（人工核可）。提交可重現、有證據的結論。
6. 全程**離線、零網路、零新相依**（純讀 JSONL + 集合運算 + 既有 `fit`/`eval` 數學;人工盲標為唯一手動步驟）。

### 非目標 (v2.2，YAGNI)

- **不**改推論路徑：`core`/`server` 不變（唯一可能變動是 §8 的「預設一行切換」與 `weights.rs` 重 fit,皆依驗收結果）。
- **不**動 v2 的 `features`/`model`/`ranker`/`LinearModel`。
- **不**新增 labeler、**不**用任何 LLM（gold 由**人類**判,不是再叫一個模型）。
- **不**做「更豐富特徵」（留待 v2.3）。
- **不**自動重出貨：判決→動作需**人工核可**。
- **不**動 v1（凍結）。

## 3. 架構與管線

v2.1 是「synth → label → fit → eval」四站。v2.2 在**評估端**加一條**標籤獨立**的支線,不動前三站,也不改既有 `eval` 的「對自身標籤」行為（向後相容,`--gold` 為加法）。

```
labeled.claude.jsonl ┐
                      ├─▶ gold-pool ─▶ data/gold.unlabeled.jsonl   claude≠codex 的 143 題（無標籤、決定性）
labeled.codex.jsonl  ┘                         │
                                               ▼  （★ 人工盲判,沿用 1–5 rubric）
                                        data/gold.jsonl            143 題人類難度（標籤獨立的黃金尺）
                                               │
labeled.{claude,codex,gemma}.jsonl ──▶ compare --gold ──▶ 每個 learned router + heuristic
weights.rs（已出貨）           ──▶ eval --gold    ──▶   對 143 題「人類」標籤評分 → 權威判決
labeled.{gemma,claude,codex}  ──▶ crosseval     ──▶   fit A / 評 B 的泛化矩陣（標籤 vs 標籤,診斷）
```

**職責分離**：`gold-pool`（出題,決定性、離線）與**人工盲判**（產生 gold）分離;`gold.jsonl` 一旦提交即為**凍結、可重現的評估輸入**。`eval --gold` / `crosseval` 皆為**唯讀**（不寫 artifact）。

**網路與相依界線**：v2.2 **不連任何網路、不加任何相依**（連 v2.1 的 `reqwest` 都用不到）。僅讀寫 `data/*.jsonl` 並沿用既有 `serde_json` 與 `logreg`/`eval` 邏輯。

### 模組變更（皆在 `crates/trainer`）

- `src/gold.rs`（**新**）：`gold-pool` 建構（讀兩套標籤、算 `claude≠codex`、寫無標籤清單）;gold 評估輔助（載入 `gold.jsonl`、對指定 router 在 gold 上算 Spearman/ordinal）。
- `src/eval.rs`（**改**）：新增 `--gold <file>` 路徑（候選 router 對人類 gold 評分）;`compare` 支援 `--gold`（同一 gold 尺、多 labeler 並排）;新增 `crosseval` 入口。沿用既有 `evaluate()` / `EvalReport` / Spearman / ordinal 純函式。
- `src/main.rs`（**改**）：dispatch `gold-pool`、`crosseval`;`eval`/`compare` 解析 `--gold`;usage 字串更新。
- `Cargo.toml`：**不變**（無新相依）。
- `prompts/README.md`（**改**）：補一段 gold 盲判流程（沿用 `label.prompt.md` 的 1–5 rubric,人工逐題判,不看模型答案）。

## 4. Gold 池產生 (`gold-pool`)

- 讀 `data/labeled.claude.jsonl` 與 `data/labeled.codex.jsonl`,以 `query` 為鍵 join。
- 選出 `difficulty(claude) != difficulty(codex)` 的查詢（現值 143 題）。
- 輸出 `data/gold.unlabeled.jsonl`,每行 `{ "query", "category" }`——**只給題目,不含 difficulty、不含任一模型的 rating/標籤**（確保人類盲判,不被錨定）。
- **決定性**：依 `labeled.codex.jsonl` 的行序（即 `corpus.jsonl` 語料原始順序）輸出;同樣兩套標籤 → 同樣 143 題、同樣順序 → 可提交、可重現。
- 印出摘要（總數、各類別分布、與兩套標籤的對齊檢查）。
- 若日後 `labeled.claude/codex.jsonl` 改變,重跑 `gold-pool` 重新產生（gold 集隨之需重判增量部分）。

## 5. 人工盲標 → `data/gold.jsonl`

- **誰判**：repo 擁有者（人類）。這是 gold 之所以「獨立」的根本——標籤不能來自任何受測模型。
- **怎麼判**：沿用 v2.1 的 1–5 rubric（`prompts/label.prompt.md`：1 = 瑣碎閒聊…5 = 專家級）,逐題給整數評分;**只看題目文字,不看任何模型答案**。`difficulty = (rating − 1) / 4`。
- **產出**：`data/gold.jsonl`,每行 `{ "query", "difficulty", "category", "rating" }`,與 `gold.unlabeled.jsonl` **行數相同、順序相同、query 逐字相同**。
- 因 gold 池的分歧全為相鄰一級,實務上每題是在兩個相鄰值間二選一,~143 題約 15–20 分鐘。
- 工具便利可用 `prompts/label.prompt.md` 的格式,但**判斷必須是人**（不可再丟給模型,否則退化成第四個 labeler、失去獨立性）。

## 6. 可重現性與資料 (`data/`)

- `data/gold.unlabeled.jsonl`：`gold-pool` 產出,queries-only,決定性,**提交**（可由兩套標籤重現）。
- `data/gold.jsonl`：人工盲標,**提交**,是 gold 評估的**凍結、可重現輸入**。
- **可重現性模型**（同 v2.1 精神）：把「不可重現的那一刻」（這裡是**人類判斷**,而非 LLM）凍結成提交的 `gold.jsonl`;之後 `eval --gold` / `compare --gold` / `crosseval` 對固定輸入皆決定性。網路永不介入。

## 7. gold 評估 / cross-eval

### `eval --gold <gold.jsonl>`
- 測試集 = gold 的 143 題,**真值 = 人類 `difficulty`**。
- 評**已出貨的** `weights.rs`（learned）與 `heuristic`,各自對 143 題預測難度 → 與人類標籤算 **Spearman**、**ordinal accuracy**。
- 這是「**部署中的 router 在難題上有多貼近人類**」的忠實檢查。

### `compare --gold <gold.jsonl> <labeled-A> <labeled-B> ...>`
- 對每個 `labeled-*.jsonl`：以其**全量**擬合 learned（沿用 `fit`/`logreg`）→ 在 gold 143 題上預測 → 對人類標籤算 Spearman/ordinal。
- `heuristic` 評一次（無需擬合）。
- 印一張表：`labeler | spearman_vs_gold | ordinal_vs_gold | avg_cost`。**這是「該出貨哪個 labeler」的決策表**——所有列用**同一把人類尺**,跨 labeler 公平可比。
- 因 `LinearModel` 容量極低（§1）,full-fit **不會記住個別查詢**,且這正反映「實際會出貨的那個 router」的行為。惟 gold 是 `train` 的子集,故 `learned` 相對 train-free 的 `heuristic` 仍有**輕微 in-sample 優勢**:margin 大時可忽略,接近時應以 §15.4 的 leakage-free 變體交叉確認。

### `crosseval`
- 對 `{gemma, claude, codex}` 的每個有序對 (A, B)：fit 在 A、評 B 的 holdout（沿用既有 80/20 與指標）→ Spearman/ordinal。
- 印 3×3 矩陣：對角線 ≈ 自評,非對角線 = 跨 labeler 泛化。
- **純標籤 vs 標籤**的診斷（無人類、無網路）,回答「在 A 上學到的難度感能多大程度遷移到 B」。輔助理解,不主導出貨決定。

### 成本指標
- gold 為**難題子集**,difficulty 分布偏高,v2.1 的「可達成充分率上限」在此可能退化,故 **gold 判決主軸用 Spearman + ordinal**（gold 能對人類乾淨量測的排序品質）。
- `avg_cost`（標籤無關）仍列出作參考;`cost@ceil` 視實作於 gold 子集是否有意義決定是否保留,**不作為 gating 條件**。

## 8. 驗收與出貨重決（完成定義）

人工盲標完成、提交 `gold.jsonl` 後,跑 `compare --gold` 與 `eval --gold`,依下列規則**權威重決兩軸**：

- **軸 A — 預設 router（learned vs heuristic）**：在 gold 上 `Spearman(learned) ≥ Spearman(heuristic)` **且** `ordinal(learned) ≥ ordinal(heuristic)`。
  - 成立 → 維持伺服器預設 `learned`（`choose_router` 不動）。
  - 不成立 → 把預設切回 `heuristic`（`crates/server/src/main.rs` 的 `choose_router` 一行）,並記錄結論。
- **軸 B — 出貨 labeler**：在 gold 上 learned-router 排序最佳者（先比 Spearman、再比 ordinal）出貨。
  - 若為 codex（現狀）→ `weights.rs` 不變。
  - 若為他者（如 claude）→ `cp data/labeled.<winner>.jsonl data/labeled.jsonl && trainer fit` 重生 `weights.rs`,並重跑 `eval --gold` 確認後出貨。
- **動作需人工核可**：判決先記於 §16 / PR 描述,再執行 `fit` / 預設切換（符合「先複審、後實作」）。
- **完成 = 一份基於標籤獨立黃金尺、可重現、有證據的結論**（不論是否改變現狀）。

## 9. 錯誤處理

- **推論無新錯誤面**：`core`/`server` 不變。
- `gold-pool`：缺 `labeled.claude/codex.jsonl` → 清楚錯誤、非零退出;兩套 query 集不一致 → 以 join 後交集計算並在摘要中報告差異。
- `eval --gold` / `compare --gold`：`gold.jsonl` 解析失敗 → 沿用既有「附行號的明確錯誤」;gold query 與 labeled query 對不上 → 報告並以可對齊者評估。
- `crosseval`：缺某套標籤 → 跳過相關列並記錄。

## 10. 相依與效能

- **零新相依**：v2.2 全部用既有 `serde`/`serde_json` 與既有 `logreg`/`eval`;**連網路都不需要**。
- 維持 `[profile.dev.package."*"] debug = false`;release build。
- 效能：`gold-pool`/`eval --gold`/`compare --gold`/`crosseval` 皆為千題級的記憶體內運算,亞秒級。唯一耗時是**人工盲標**（一次性 ~15–20 分鐘）。

## 11. 測試策略 (TDD)

- `gold.rs`：
  - `gold-pool` 對小型 fixture 精確選出 `claude≠codex` 集（含計數與順序決定性）。
  - 輸出為**盲**（無 `difficulty` 欄位）。
  - 載入 `gold.jsonl` + 對指定 router 在 gold 上算 Spearman/ordinal 的純函式測試（用 fixture 標籤,不依賴真實 gold）。
- `eval.rs`：
  - `--gold` 解析（`parse_*_flag`）。
  - `compare --gold` 對固定 fixture 產生跨 labeler 表,指標落在合理範圍。
  - `crosseval` 產出 N×N、對角線存在。
- **回歸**：`gold.unlabeled.jsonl` 行數 = 計算出的分歧數;`gold.jsonl`（若存在）parse、`difficulty ∈ {0,.25,.5,.75,1}`、行數對齊 gold 池。
- 推論側（core/server）測試不變且維持全綠。

## 12. 向後相容 / 隔離

- 推論路徑不變;唯一可能變動是 §8 依驗收結果的「預設一行切換」與/或 `weights.rs` 重 fit（資料變更）。
- 既有 `eval` / `compare`（對自身標籤、自身 holdout）**行為不變**;`--gold` 與 `crosseval` 為**純加法**。
- v1 凍結;v2 的 `features`/`model`/`ranker`/`LinearModel` 不動。
- `core`/`server` 的 `Cargo.toml` 不變（無新相依）。

## 13. 不做（YAGNI）

任何 LLM 參與 gold（gold 必須是人）、雲端、推論時網路、改 API/端點、動 v1 或 v2 推論核心、自動重出貨、更豐富特徵（v2.3）、§15 其餘未來工作。

## 14. 待定 / 起始預設（可調）

- gold 池 = `claude≠codex`（現值 143 題,決定性源自現有兩套標籤;標籤若變則重生）。
- gold 由 repo 擁有者**人工盲標**,沿用 1–5 rubric。
- `crosseval` 是否保留 gemma 列：保留（免費、具參考價值）。
- gold 子集上的成本指標細節（是否保留 `cost@ceil`）於實作定稿;判決主軸為 Spearman + ordinal。
- 子指令命名（`gold-pool` / `crosseval`）與 `--gold` 旗標於實作定稿。

## 15. 開放問題

1. **代表性 vs 鑑別力**：gold 僅含難題（無 chat/extraction）,故判決講的是「**有爭議難題**上的排序品質」,而非整個語料。簡單題本就無爭議,此取捨可接受;必要時可加一小撮三方一致的易題盲驗作錨點。
2. **統計強度**：143 題對**整體**判決足夠,但**per-category**（code 15、reasoning 16）偏少。若某項比較過於接近,可擴充 gold（如再盲判被排除的 128 題 gemma-分歧、或補易題樣本）。
3. **軸 B 翻轉的後果**：若 gold 判 claude > codex,出貨 `weights.rs` 將改變 → 需重 fit + 重跑 `eval --gold` 確認後才併入。
4. **In-sample 優勢（holdout-free 的取捨）**：`eval --gold` / `compare --gold` 採 full-fit（反映實際出貨的 router）,但 gold 143 題是訓練語料的子集,故 `learned` 相對 train-free 的 `heuristic` 有輕微 in-sample 優勢。低容量線性模型下偏差很小;**若某判決邊際接近**,應補一個 leakage-free 變體(fit 時剔除 gold 題再對 gold 評分)交叉確認。

## 16. 驗收結論

**結論：獨立人工 gold 確認 v2.1 的判決 → 預設維持 `learned`、出貨維持 codex 標註;`choose_router` 與 `weights.rs` 皆不變(零程式/權重改動)。**

擁有者盲標 143 題 claude≠codex 的查詢(rating 分布 2:31 / 3:32 / 4:48 / 5:32,無 1——全為爭議難題),提交 `data/gold.jsonl`。各 router 對**同一套人類標籤**評分(`compare --gold`):

| labeler（在 gold 上,n=143） | Spearman vs human | ordinal vs human | avg_cost | 備註 |
|---|---|---|---|---|
| heuristic | 0.670 | 0.322 | 0.164 | 基準(難題上分級近乎失效) |
| learned-fit-claude | 0.850 | 0.727 | 0.367 | |
| **learned-fit-codex（現出貨）** | **0.932** | **0.874** | 0.318 | **gold 最佳** |
| learned-fit-gemma（參考） | 0.872 | 0.678 | 0.265 | |

- **軸 A 決定（預設 router）：維持 `learned`。** 最佳 learned（codex）`Spearman 0.932 ≥ heuristic 0.670` 且 `ordinal 0.874 ≥ 0.322`,依 §8 勝出。margin 極寬（Spearman 差 0.26、ordinal 差 0.55），遠超 §15.4 的 in-sample 優勢所能解釋,故**無需** leakage-free 重 fit;`choose_router` 不動。
- **軸 B 決定（出貨 labeler）：維持 codex。** codex 在 gold 上 Spearman/ordinal **雙項最佳**,即現出貨者 → `weights.rs` 不變。(gold 題對三套 labeler-router 皆為 in-sample,故此三方比較對稱、無偏。)
- **成本說明**：learned 的 avg_cost（0.318）高於 heuristic（0.164）是**正確行為**——gold 為難題集(32 題難度 1.0),正確路由本就該選高品質(高成本)模型;heuristic 的「低成本」源自低估難度而**充分率不足**。成本為資訊性指標、非 gating(§7)。
- **跨 labeler 診斷（`crosseval`,80/20 holdout）**：fit-on-row 預測 col 之 holdout 的 Spearman 介於 0.864–0.913,off-diagonal（跨 labeler 遷移）達 0.86–0.91,顯示難度訊號**真實且跨 labeler 穩健**,非 codex 專屬;gemma 欄最低(~0.86–0.87),印證其為最雜訊的標註者。
- **意義**：v2.1 的判決原是**自我參照**(各 router vs 自身標籤);v2.2 以**獨立於受測模型的人工 gold** 重跑,結論一致——出貨的 codex-learned router 確實最貼近人類判斷,且難題上對 heuristic 的優勢決定性。無需任何出貨變更。
