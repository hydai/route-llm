# route-llm v3 — 設計規格：Reasoning Budget Router（預算式路由）

> 狀態：v3 已實作並驗收（見 §16 結論）。判決：budget 輸 gold gate → 採「learned 主幹 + budget 決策層」，出貨 codex 6-head，預設維持 `learned`。
> 日期：2026-06-15（設計）/ 2026-06-16（實作 + 驗收）
> 關係：本規格在 v2（learned router，見 `SPEC-v2.md`）與 v2.1/v2.2（真實標註 + 可信判決，見 `SPEC-v2.1.md`/`SPEC-v2.2.md`）之上做**加法**——新增**第三個路由策略** `BudgetRouter`，與 `heuristic`/`learned` 平起平坐。核心思路改寫自朋友提出的 **Reasoning Budget Classifier（RBC）**：不先問「這是哪一類任務」，而是估算「這個 prompt 需要多少 reasoning budget 才能穩定做對」。v1/v2 推論核心**凍結不動**；inference 仍**零網路、純決定性**；frontier LLM 僅用於**離線**產生標籤。

## 1. 概述與動機

v1/v2 把每個 query 收斂成**單一純量** `difficulty ∈ [0,1]`，再交給共用 `ranker` 依 adequacy（`sigmoid(K·(quality − difficulty))`）與成本排序候選模型。這條管線簡潔且已被 v2.2 的人工 gold 證明有效（learned 在 143 題難題上 Spearman 0.932）。

但「單一難度分數」混合了三件本質不同的事：**任務類型**、**推理深度**、**模型路由結果**。朋友的觀察是：表面任務類型不等於 reasoning 難度（一行 regex 是 R1，大型 concurrency bug 是 R4；一般翻譯 R0，法律/醫療翻譯 R3）。與其用一個不透明分數，不如把難度**拆解成可解釋的維度**，再結合**錯誤成本**、**驗證難度**、**模糊度**與**不確定性**決定路由。

v3 的修法：新增 `BudgetRouter`，把 prompt 拆成 **6 個維度**（由多個 frontier LLM **離線**標註、各自學成一個低容量線性 head），加權成 **budget_score → R0–R4**，再套一層**決策層**（風險升級、工具需求、信心/分歧處理、policy 模式），最後**仍收斂成統一的 `Recommendation`（ranking）**。

**為何仍是「加法、不破壞推論路徑」**：v3 是**獨立策略**。不同策略吃**相同輸入**、可以有**不同中間產物**，但最後都輸出同一個 `Recommendation`。`Recommendation` 只**新增一個選用欄位** `budget: Option<BudgetBreakdown>`（`#[serde(skip_serializing_if)]`）——`heuristic`/`learned` 留 `None`，舊輸出**逐字不變**，只有 `BudgetRouter` 填它。

**為何「6 維 learned + LLM 標註」而非純手調公式**：v2.2 §16 量過——在 143 題難題 gold 上，v1 純 heuristic 只有 Spearman 0.670，learned 是 0.932。純手調公式在最該準的難題上會大幅落後。因此 v3 的 6 維採**學習式**（沿用 v2 的「LLM 離線標、純 Rust 模型上線」管線，只是從單一 difficulty 擴成 6 維），並把「難度主幹是否改用 budget_score」這件事**交給 gold 裁決**（§5、§8），不預先賭。

**為何「多個 frontier LLM 標註」剛好補上 RBC 的弱點**：朋友設計裡「Rule 4：模型分歧 = uncertainty」原本是空中樓閣。但 v2.2 已有 `crosseval`／claude≠codex 的跨 labeler 分歧機制。v3 把標註擴成 6 維後，可**逐維度**算跨 labeler 分歧，用以**校準** runtime 的 `confidence` 與 escalation——讓「分歧 = 不確定」**接上真實資料**，而非憑空發明。

## 2. 目標與非目標

### 目標 (v3)

1. 新增 core 策略 `BudgetRouter`（`crates/core/src/budget/`，比照 `learned/` 隔離），實作 `Router` trait，經 `ROUTE_LLM_ROUTER=budget` 選用。
2. 6 個 per-dimension learned head（共用既有 feature 向量），各 fit 在該維的 LLM 標籤；以朋友的加權公式組成 `budget_score` 與 R0–R4。
3. 決策層（純決定性、雙語關鍵字、零網路）：硬性風險升級、最新資訊→`needs_tool`、信心/雙估計器分歧處理、`Balanced/Strict/Cheap` policy 模式。
4. 輸出層**加法**：`Recommendation.budget: Option<BudgetBreakdown>`；ranking 仍由既有 ranker 在使用者候選清單上產生（= RBC 的 `minimum_sufficient_model`）。
5. trainer 擴充：`label --dims`（6 維標註，每題一次 call）、`fit-budget`（6 head）、`eval-budget` 與 `crosseval --dims`（逐維跨 labeler 診斷）。新 `prompts/label.budget.prompt.md`。
6. **gold-gated 出貨**：v3 的 `difficulty` 在 v2.2 的 143 題人工 gold 上 `Spearman/ordinal ≥ learned` 才讓 `budget_score` 取代難度主幹；否則 learned 留主幹、budget 當決策/解釋層。判決寫進 §16（先複審、後實作）。
7. 全程 inference **零網路、純決定性、零新 runtime 相依**；frontier LLM 僅出現在離線 `label`。

### 非目標 (v3，YAGNI)

- **不**在 inference 時呼叫任何 LLM（6 維由離線標籤學成；上線是純 Rust head）。
- **不**讓服務**執行** verify / fallback / 工具呼叫——`needs_tool`/`requires_verifier`/`fallback_policy` 為**建議旗標**（本服務只「推薦」，不代呼叫模型，符合 repo 身分）。
- **不**動 v1（`difficulty.rs` 凍結）與 v2（`learned/`、`features.rs`、`ranker.rs`、`LinearModel` 不動）。
- **不**新增 API 端點（三種 dialect 沿用；budget 隨 `route_llm`/`Recommendation` 一起序列化）。
- **不**改動 v2.2 凍結資料（`labeled.*.jsonl`、`gold.jsonl` 不動；6 維標籤寫**新檔** `data/budget.*.jsonl`）。
- **不**做 per-dimension 的人工 gold（留待 v3.1；v3 的人工裁決沿用 v2.2 的單一難度 gold 當主幹 gate）。

## 3. 架構與管線

```
                       ┌─ HeuristicRouter (v1, 凍結) ─┐
query, models, prefs ──┼─ LearnedRouter   (v2, 凍結) ─┼─▶ Recommendation { difficulty, ranking,
                       └─ BudgetRouter    (v3, 新)  ──┘                     budget: Option<BudgetBreakdown> }
                                                          ROUTE_LLM_ROUTER ∈ {heuristic, learned, budget}
```

`BudgetRouter` 內部管線：

```
query ─▶ features (v2 既有, 不動)
          │
          ├─▶ 6× dimension head (LinearModel) ─▶ [reasoning_depth, verification_difficulty,
          │                                        constraint_density, context_integration,
          │                                        ambiguity, error_cost]  ∈ 各自尺度
          │            │
          │            ▼  Σ friend-weights
          │     budget_score ─▶ R-level (R0..R4) ─▶ recommended_model_tier
          │            │                              + per-dimension reason_codes
          │            ▼
          ├─▶ 決策層: escalation rules + confidence + policy  (純決定性)
          │            │  high-risk floor / needs_tool / 雙估計器分歧 / Balanced|Strict|Cheap
          │            ▼
          │     最終 level + needs_tool + requires_verifier + fallback_policy + confidence
          │            │
          ▼            ▼
     difficulty(=budget_score 正規化 或 learned 主幹, 視 §5 gold gate)
          │
          ▼
     共用 ranker (v1, 不動) ─▶ ranking  ─────────────▶ Recommendation (+ budget 區塊)
```

**離線訓練管線**（沿用 v2.1「synth → label → fit → eval」精神，擴成 6 維）：

```
data/corpus.jsonl ──▶ label --dims ──▶ data/budget.{claude,codex,gemma}.jsonl  (每題 6 個整數)
                                              │
                                              ▼ fit-budget (per dimension, 共用 logreg)
                                       crates/core/src/budget/weights.rs  (6 head 出貨權重)
                                              │
data/gold.jsonl (v2.2 人工難度) ──▶ eval-budget ──▶ budget difficulty vs human → §5 gate / §16 判決
data/budget.{...} ─────────────▶ crosseval --dims ──▶ 逐維跨 labeler 分歧矩陣 (診斷 + confidence 校準)
```

### 模組變更

- **`crates/core/src/budget/`（新目錄，比照 `learned/`）**
  - `mod.rs`：`BudgetRouter`（實作 `Router`）；組裝 dims → level → 決策層 → difficulty → `ranker::rank`。
  - `dims.rs`：6 個 `LinearModel` head 的載入與評分；維度名稱、尺度常數、`Dimensions` 結構。
  - `level.rs`：朋友加權公式、`budget_score`、R0–R4 分桶、level↔tier↔difficulty 映射、門檻常數。
  - `escalation.rs`：風險域偵測、最新資訊偵測、信心公式、雙估計器分歧、`Policy` 套用。
  - `weights.rs`：出貨的 6 head 權重（由 `trainer fit-budget` 產生）+ `BUDGET_SCHEMA_VERSION`。
- **`crates/core/src/model.rs`（改，加法）**：新增 `BudgetBreakdown`、`DimensionScores`、`Policy`；`Recommendation` 加選用 `budget: Option<BudgetBreakdown>`。`RoutingPreferences` 與 `Router` trait **不變**（`Policy` 走 BudgetRouter 啟動設定，見 §6.3）；v1/v2 的 `recommend()` 各加一行 `budget: None`（序列化後逐字不變）。
- **`crates/core/src/lib.rs`（改）**：`pub mod budget;`、re-export `BudgetRouter`、`Policy`、`BudgetBreakdown`。
- **`crates/server/src/dto.rs`（不變）**：responses 沿用 `Recommendation` 序列化，budget 自動隨附；無新欄位。
- **`crates/server/src/main.rs`（改）**：`choose_router` 加 `"budget"`；新增 `choose_policy(ROUTE_LLM_POLICY)` → `BudgetRouter::with_policy(...)`，與 `choose_router` 平行。
- **`crates/server/src/handlers.rs`（改，最小）**：`summary_line` 在 `budget` 存在時可補一句（R-level/tier）；其餘不動。
- **`crates/trainer/`（改）**：`label.rs` 加 6 維模式；`dataset.rs` 解析 6 維；`logreg.rs` 重用（fit 6 次）；`eval.rs`/`gold.rs` 加 `eval-budget`、`crosseval --dims`；`main.rs` dispatch 新子指令。
- **`prompts/label.budget.prompt.md`（新）**：6 維 rubric（見 §4）。
- **`data/budget.{claude,codex,gemma}.jsonl`（新）**：6 維標籤。`corpus.jsonl`、`gold.jsonl` 重用、不動。
- **`Cargo.toml`（core/server）**：不變（零新 runtime 相依）。trainer 沿用既有 `reqwest`（離線 label）。

## 4. 六維評分模型（heads + 標註）

### 4.1 維度、尺度與加權

沿用朋友 §5–§6 的維度定義與權重：

| # | 維度 | 標註尺度 | 權重 | 直覺 |
|---|---|---|---|---|
| 0 | reasoning_depth | 0–4 | 1.4 | 需要幾層推理／是否要拆解問題 |
| 1 | verification_difficulty | 0–4 | 1.1 | 答案多難檢查（短但難驗 → 升級） |
| 2 | constraint_density | 0–4 | 1.0 | 同時要滿足幾個條件（格式/語氣/來源/成本…） |
| 3 | context_integration | 0–4 | 1.0 | 需整合多少上下文（長文／多文件／多來源） |
| 4 | ambiguity | 0–3 | 0.8 | 是否多種合理解讀／需自建框架 |
| 5 | error_cost | 0–4 | 1.2 | 做錯的代價（金錢/法律/健康/安全/不可逆） |

```
budget_score = 1.4·reasoning_depth + 1.1·verification_difficulty
             + 1.0·constraint_density + 1.0·context_integration
             + 0.8·ambiguity + 1.2·error_cost
```

理論最大值 `= 1.4·4 + 1.1·4 + 1.0·4 + 1.0·4 + 0.8·3 + 1.2·4 = 25.2`。

### 4.2 6 個 per-dimension head

- 每維一個 `LinearModel`（**重用 v2 的型別與標準化**：`weights/bias/means/stds` + logistic link），**共用同一份 `learned::features::features(query)` 向量**（零特徵改動、train/inference 一致）。
- head `i` 輸出 `p_i ∈ [0,1]`（logistic）；還原成該維整數尺度：`dim_i = p_i · scale_i`（`scale = [4,4,4,4,3,4]`）。供加權與 reason_codes 用。
- **per-dimension contribution**：`contrib_i = weight_i · dim_i`；reason_codes 取 contribution 最高的數維（如 `["reasoning_depth", "error_cost"]`），對應 RBC 輸出的 `reason_codes`。
- 容量極低（少數權重、無 per-query 參數）→ 數學上無法背下個別 query；即使 gold 題曾在訓練語料，用**人類**難度評分仍公平（同 v2.2 §1 論證）。

### 4.3 標註 → `data/budget.<labeler>.jsonl`

- **誰標**：多個 frontier LLM（沿用 v2.1 既有 claude/codex/gemma 三方；可加別的 frontier model）。
- **怎麼標**：`prompts/label.budget.prompt.md` 給 6 維 rubric，模型對**每題一次 call** 回傳 6 個整數（在各自尺度內）。**不看模型答案，只看題目**。
- **產出**：每行 `{ "query", "category", "dims": { "reasoning_depth", "constraint_density", "ambiguity", "context_integration", "verification_difficulty", "error_cost" } }`，行序對齊 `corpus.jsonl`。
- **可重現性**（同 v2.1 精神）：把「不可重現的那一刻」（LLM 標註）凍結成提交的 `data/budget.*.jsonl`；之後 `fit-budget`/`eval-budget`/`crosseval` 對固定輸入皆決定性。inference 永不連網。
- **成本**：6 維併在一次 prompt 回傳 → 標註呼叫數**與 v2.1 同級**（每 (query, labeler) 一次），非 6×。

## 5. budget_score → R-level → difficulty（難度主幹與 gold gate）

### 5.1 level 分桶與 tier（朋友 §6 起始值，gold 可校準）

| budget_score | Level | recommended_model_tier |
|---:|---|---|
| 0 – 3 | R0 | tiny / small |
| 4 – 7 | R1 | small |
| 8 – 11 | R2 | medium |
| 12 – 16 | R3 | strong |
| 17+ | R4 | best |

門檻為**起始預設**；實作後以 143 題 gold 校準（R0–R4 ↔ 人工 rating 1–5）使 ordinal 最佳化（§8）。`recommended_model_tier` 為**資訊性**——實際 top pick 仍由 ranker 在使用者候選清單上挑（候選不含該 tier 時，ranker 仍選最佳可得者並於 reason 說明）。

### 5.2 difficulty 給 ranker（**gold-gated**，不預賭）

`ranker` 需要一個純量 difficulty。兩個候選來源：

- **(主幹候選) budget_score 正規化**：`difficulty = budget_score / 25.2 ∈ [0,1]`（單調，Spearman 不受線性縮放影響；ordinal 由 §5.1 的 R-level↔rating 桶決定）。
- **(備援) 沿用 learned 主幹**：`difficulty = LearnedRouter` 的純量（v2.2 gold 已證 0.932），budget 區塊僅作決策/解釋層疊加。

**裁決規則（§8 軸 A）**：v3 的「budget 正規化 difficulty」在 143 題 gold 上 `Spearman ≥ learned` **且** `ordinal ≥ learned` → 採 budget 當主幹；否則採備援（learned 主幹 + budget 決策層）。無論何者，budget 區塊都照常輸出；差別只在「ranking 由誰的難度驅動」。判決寫進 §16。

### 5.3 raw 估計 difficulty vs runtime difficulty（消歧）

兩者刻意分離：

- **raw 估計 difficulty**（= `budget_score / 25.2`，**escalation 之前**）：純粹是「6 head 對難度的估計」。**§8 軸 A 的 gold gate 只評這個**——因為人工 gold 評的是**難度**，不是風險/policy。escalation 是路由政策疊加，不該污染「估計器有多準」的量測。
- **runtime difficulty**（餵 ranker，§6 step 7）：取 raw 與「escalation 後最終 level 下界」的較大者——`difficulty = max( budget_score/25.2, lower_bound(final_level)/25.2 )`。確保被 high_risk floor 或 policy 升級的 query，ranker 確實看到更高難度而選更強模型；未觸發 escalation 時即等於 raw。

## 6. 決策層（escalation / confidence / policy）

純決定性、雙語關鍵字（沿用 v1/v2 的 EN+中文 風格）、零網路。執行順序（鏡像朋友 §9–§10）：

```
1. dims → budget_score → base_level                          (§4–§5)
2. high_risk_domain          → level = max(level, R3)
   + requires_expert_judgment → level = max(level, R4)
3. requires_latest_info      → needs_tool = true; tool_type = "web_search"  (不單獨升級)
4. confidence < τ(policy)    → level = upgrade(level, 1)
5. policy 調整               → Balanced | Strict | Cheap     (§6.3)
6. |Δlevel| (budget vs learned) ≥ 2 → level = max; requires_verifier = true
7. level → tier;  runtime difficulty (§5.3) → ranker → ranking
```

### 6.1 硬性升級偵測（關鍵字，可擴充）

- **high_risk_domain**（→ floor R3）：法律/legal、醫療/medical/health、金融投資/finance/investment、資安/security、生產環境/production/deploy、個資/PII/personal data。其中需「具體建議/判斷」者 → R4。
- **requires_latest_info**（→ `needs_tool`）：today/now/latest/current/最新/今天/現在/匯率/股價/news/CEO is…。reasoning 不一定高，故只設工具旗標、不升 level。

清單為起始值，集中於 `escalation.rs` 常數；擴充即改一處。

### 6.2 confidence（決定性，雙估計器代理）

朋友的「classifier confidence / Opus-vs-GPT 分歧」在純決定性 core 裡改寫為兩個**可在 inference 算出**的訊號：

- `boundary_margin ∈ [0,1]`：`budget_score` 距最近 level 門檻的距離 ÷ 該 level 寬度（越靠邊界越不確定）。邊界 level 的寬度以 `[0, 25.2]` 封口（R0 下界 0、R4 上界 25.2）。
- `estimator_agreement ∈ [0,1]`：`1 − |R-level(budget) − R-level(learned)| / 4`（兩個獨立估計器一致 → 高）。

```
confidence = clamp( 0.5·boundary_margin + 0.5·estimator_agreement , 0, 1 )
```

其「該升級的門檻 τ」與權重，由**離線逐維跨 labeler 分歧**（`crosseval --dims`）校準：分歧大的特徵區應對應較低 confidence。`Δlevel` 即 RBC「Rule 4：模型分歧」的決定性化身——`≥1` 視為 uncertainty（升一級）、`≥2` 取 max 並要求 verifier。

### 6.3 policy 模式

`Policy` 為 **`BudgetRouter` 啟動設定**（`ROUTE_LLM_POLICY=balanced|strict|cheap`，預設 `balanced`），與 `ROUTE_LLM_ROUTER` 平行——**不入** `RoutingPreferences`、不改 `Router` trait（保持 v1/v2 共用型別與 `ranker.rs`/`router.rs` 完全凍結）。per-request policy（走 API body）留待 v3.1（YAGNI）。三種模式：

- **Balanced**（預設）：borderline（confidence < 0.7）升一級；high_risk floor R3。
- **Strict**（品質優先）：confidence < 0.85 升一級；`Δlevel ≥ 1` 升一級；high_risk 直接 R4；複雜任務（base ≥ R3）`requires_verifier = true`。
- **Cheap**（成本優先）：低風險（無 high_risk 且 error_cost 低）可降一級；一律 `requires_verifier = true`、`fallback_policy = "upgrade_if_verifier_fails"`。Cheap 不是用爛模型，而是「cheap-first + 建議驗證 + 建議 fallback」（旗標為 advisory，見 §2）。

`cost_bias` 與 `policy` 正交：前者調 ranker 的成本權衡，後者調 level 升降與 verifier 旗標。

## 7. 輸出契約（BudgetBreakdown，加法）

`Recommendation` 既有欄位（`difficulty`、`ranking`）**不變**，新增：

```jsonc
"budget": {                          // 只有 BudgetRouter 填；其餘策略此欄不出現
  "level": "R3",
  "budget_score": 13.4,
  "recommended_model_tier": "strong",
  "confidence": 0.78,
  "dimensions": {                    // 還原到各自尺度 + contribution
    "reasoning_depth": 3, "verification_difficulty": 2, "constraint_density": 2,
    "context_integration": 1, "ambiguity": 1, "error_cost": 2
  },
  "reason_codes": ["multi_step_reasoning", "needs_validation", "constraint_dense"],
  "needs_tool": false,
  "tool_type": null,
  "requires_verifier": false,
  "fallback_policy": "none"
}
```

- 序列化以 `#[serde(skip_serializing_if = "Option::is_none")]` 保證 v1/v2 輸出**逐字不變**（回歸測試守住）。
- `route_llm`（OpenAI dialect）與 `route_llm`（Anthropic dialect）同樣自動隨附 budget（因兩者都內嵌 `Recommendation`）。
- `reason_codes` 由 per-dimension contribution + 觸發的 escalation 規則組成（穩定、決定性順序）。

## 8. 驗收與出貨重決（完成定義）

完成 6 維離線標註並提交 `data/budget.*.jsonl`、跑 `fit-budget`/`eval-budget`/`compare --gold`/`crosseval --dims` 後，依下列規則**裁決**：

- **軸 A — 難度主幹（budget vs learned）**：在 143 題 gold 上 `Spearman(budget) ≥ Spearman(learned)` **且** `ordinal(budget) ≥ ordinal(learned)`。
  - 成立 → budget_score 正規化驅動 ranker 的 difficulty。
  - 不成立 → 採 learned 主幹 + budget 決策層（§5.2 備援）。budget 區塊兩種情況都輸出。
- **軸 B — 出貨 labeler（6 維）**：在 gold 上整體最佳的 labeler 的 6-head 權重出貨（先比 difficulty 對人類的 Spearman、再比 ordinal；逐維分歧由 `crosseval --dims` 輔助判斷哪幾維可信）。
- **level 門檻校準**：以 gold 的 R-level↔rating 對齊微調 §5.1 門檻使 ordinal 最佳化；門檻定案寫入 `level.rs` 常數。
- **預設 router 不變**：`ROUTE_LLM_ROUTER` 預設仍 `learned`（v2.2 已驗）。`budget` 為**可選**第三策略；是否改預設為 `budget` 視軸 A margin 與穩定度，**人工核可**後另行決定（保守起見 v3 預設不改）。
- **動作需人工核可**：判決先記於 §16 / PR 描述，再執行 `fit-budget` 出貨與任何門檻/預設變更（先複審、後實作）。
- **完成 = 一份基於 v2.2 人工 gold、可重現、有證據的結論**（含 budget 是否取代主幹、出貨哪個 labeler 的 6-head、門檻定值）。

## 9. 錯誤處理

- **推論**：`BudgetRouter` 與 v1/v2 一樣**不新增可失敗路徑**（純函式、無 I/O）；head 維度與 feature 長度不符 → `debug_assert`（同 `LinearModel`）。空 query / 空候選 / 非法 `cost_bias` 沿用 `handlers::process` 既有檢查。未知 `policy` 字串 → DTO 反序列化回退 `Balanced`（不 fail）。
- **離線**：`label --dims` 某題回傳缺維/越界 → 記錄並跳過該題（沿用 v2.1 的 skip-and-continue），不中止整批；持續性網路故障才 abort。`fit-budget` 缺某 labeler 檔 → 清楚錯誤、非零退出。`eval-budget` 的 gold 對不上 → 以可對齊者評估並報告差異（沿用 v2.2）。

## 10. 相依與效能

- **零新 runtime 相依**：core/server `Cargo.toml` 不變；`BudgetRouter` 只用既有 `serde` 與重用的 `LinearModel`/`features`/`ranker`。trainer 沿用既有 `reqwest`（離線 label）。
- 維持 `[profile.dev.package."*"] debug = false`；release build。
- **inference 效能**：6 個低維 head + 一次 feature 抽取 + 常數時間決策層 → 與 v2 同級（微秒～亞毫秒），純記憶體、零網路。
- **離線效能**：`fit-budget` 為 6 次千題級 logistic 擬合（秒級）；`label --dims` 受 LLM 端速率限制，與 v2.1 同量級（每題一次 call）。

## 11. 測試策略 (TDD)

- **`budget/dims.rs`**：head 輸出有限、決定性、落在尺度內；空 query 全有限；contribution 排序穩定。
- **`budget/level.rs`**：加權公式對固定 dims 算出預期 budget_score；門檻邊界（3/7/11/16/17）分桶正確；level↔tier↔difficulty 映射；正規化 difficulty ∈ [0,1]。
- **`budget/escalation.rs`**：high_risk 關鍵字 → floor R3/R4；latest-info → `needs_tool` 且**不**升 level；confidence 公式邊界；`Δlevel ≥ 2` → max + verifier；Balanced/Strict/Cheap 的升降與旗標差異。
- **`budget/mod.rs`**：`BudgetRouter` 產生完整 ranking（長度＝候選數）；難題 budget_score > 易題；budget 區塊欄位齊全。
- **回歸（守住加法不破壞）**：`heuristic`/`learned` 的 `Recommendation` 序列化**不含** `budget` 欄位（逐字不變）；既有 core/server 測試全綠。
- **server**：`ROUTE_LLM_ROUTER=budget` 啟動；三 dialect 回應含 `route_llm.budget`；未知/省略 `policy` 回退 Balanced。
- **trainer**：`label --dims` 對 fixture 解析 6 維；`fit-budget` 產 6 head；`eval-budget`/`crosseval --dims` 指標落在合理範圍、矩陣對角線存在。

## 12. 向後相容 / 隔離

- 推論路徑加法：新增策略不改既有兩者；`Recommendation` 新欄位為 `Option` 且 skip-if-none（v1/v2 `recommend()` 各加一行 `budget: None`，序列化逐字不變，回歸測試守住）。
- `RoutingPreferences` 與 `Router` trait **完全不變**：`Policy` 走 `ROUTE_LLM_POLICY` 啟動設定，故 `ranker.rs`/`router.rs`（含其測試的 prefs 字面建構）零改動。
- v1（`difficulty.rs`）凍結；v2（`features`/`model`/`ranker`/`LinearModel`/`learned`）**不動**——6 head 重用其型別與特徵，但不修改它們。
- v2.2 資料凍結：`labeled.*`、`gold.jsonl` 不動；6 維標籤進新檔。
- core/server `Cargo.toml` 不變。`ROUTE_LLM_ROUTER` 預設維持 `learned`。

## 13. 不做（YAGNI）

inference 時呼叫 LLM、服務代執行 verify/fallback/工具、改 v1/v2 推論核心、新 API 端點、per-request policy 走 API body（v3.1）、per-dimension 人工 gold（v3.1）、動 v2.2 凍結資料、自動重出貨或自動改預設 router。

## 14. 待定 / 起始預設（可調）

- 維度尺度 `[4,4,4,4,3,4]` 與權重 `[1.4,1.1,1.0,1.0,0.8,1.2]`、level 門檻 `3/7/11/16/17`：朋友 §5–§6 起始值，gold 校準後定案（§8）。
- 標註陣容：起始沿用 claude/codex/gemma；可加 frontier model（不影響管線形狀）。
- confidence 權重（0.5/0.5）與升級門檻 τ（0.7/0.85）：起始值，`crosseval --dims` 校準。
- 子指令命名（`fit-budget`/`eval-budget`/`label --dims`/`crosseval --dims`）、env 命名（`ROUTE_LLM_POLICY`）與旗標於實作定稿。
- 是否將預設 router 改為 `budget`：v3 保守維持 `learned`，視軸 A 結果由人工另行核可。

## 15. 開放問題

1. **難度主幹是否真的勝出**：6 維 learned 的 budget_score 能否在 gold 上追平/超過 learned 的 0.932？若僅打平，採備援（learned 主幹）即可，budget 仍提供解釋與決策層價值。
2. **per-dimension 統計強度**：143 題 gold 偏難題（無 chat/extraction），某些維度（如 ambiguity）樣本分布可能偏斜；逐維判讀以 `crosseval --dims` 的跨 labeler 一致度輔助，必要時擴標。
3. **逐維跨 labeler 分歧 → confidence 校準的形式**：起始用簡單線性組合；是否值得訓一個「分歧預測 head」（第 7 個 head，預測 labeler 變異）留待 v3.1。
4. **tier 與候選清單的綁定**：`recommended_model_tier` 目前為資訊性；是否需要在候選不含該 tier 時主動警示，於實作評估（預設沿用 ranker 的 reason 說明）。
5. **Cheap 模式的 advisory 邊界**：本服務不執行 verify/fallback，故 Cheap 的「失敗才升級」只能輸出建議；若未來要真正閉環，需在呼叫端實作（超出 v3 範圍）。

## 16. 驗收結論

**結論：budget 估計器輸掉 gold gate（決定性）→ 採「learned 主幹 + budget 決策層」；出貨 codex 6-head；預設維持 `learned`，`budget` 為 opt-in 第三策略。難度路由零退步，額外得到可解釋／風險升級層。**

擁有者用 **claude、codex** 兩個 frontier labeler 對 987 題語料盲標 6 維（gemma 太慢、暫缺；claude≠codex 本就是 v2.2 gold 的分歧軸，2 labeler 足以裁決）。實測：兩套標籤逐題 90.3% 至少一維不同，最爭議維是 ambiguity／error_cost／verification（~48–52% exact），最客觀是 context_integration（88.9%）。各 budget-router（fit 各 labeler）與 learned 對 143 題人工 gold 評分：

| router（n=143） | Spearman vs human | ordinal vs human | 備註 |
|---|---|---|---|
| heuristic（v2.2） | 0.670 | 0.322 | 基準 |
| budget-fit-claude | 0.876 | 0.266 | |
| **budget-fit-codex（出貨）** | 0.871 | 0.406 | 軸 B 勝出 |
| **learned-fit-codex（現役）** | **0.932** | **0.874** | gold 最佳 |

- **軸 A 決定：`learned` 續任難度主幹（budget 落敗，決定性）。** budget Spearman 0.871/0.876 < learned 0.932，且 ordinal 0.27–0.41 << 0.874；依 §8 軸 A（Spearman 且 ordinal 雙過）→ 不成立。
- **校準無法翻案。** `ordinal_accuracy` 的 `bucket()` 為 3 桶（<0.4／<0.7／≥0.7）。budget 系統性低估難題——分桶 {易 57, 中 70, 難 16} vs 人工 {31, 32, **80**}；learned {29, 36, **78**} 幾乎完美複製人工（這正是其 ordinal 0.874 的來源）。budget_raw 範圍 [0.131, 0.844]，多數難題卡在 ~0.62（<0.7 難門檻）。**最佳單調重映（in-sample 上界）ordinal 天花板 = 0.832，仍 < 0.874**；簡單 min-max 拉伸僅 0.462。且 Spearman 與校準無關（0.871 < 0.932 恆成立）。故 §15.4 的門檻校準無法救，**不採 budget 為主幹**。
- **實作（人工核可）：** `crates/core/src/budget/mod.rs` 的 runtime difficulty 改為 `learned_diff.max(level_floor(escalated_level))`——learned 出難度，escalation（high_risk／policy／分歧）仍能向上拉、但**不會低於 learned**（回歸測試 `difficulty_backbone_is_learned_never_below` 守住；red/green 驗證過）。`budget_score` 只驅動 budget 區塊。
- **軸 B 決定：出貨 codex 6-head。** gold 上 codex Spearman 0.871 ≈ claude 0.876（差 0.005，噪訊級），但 ordinal 0.406 明顯優於 claude 0.266；且 codex 即 v2.2 learned 出貨者 → 一致。`weights.rs` 由 `cp data/budget.codex.jsonl data/budget.jsonl && trainer fit-budget` 生成。
- **逐維 `crosseval --dims`（2 labeler，fit-row/eval-col holdout）：** reasoning_depth、verification_difficulty 最穩（對角 0.88–0.90，跨 labeler 0.81–0.88）；constraint_density、error_cost 中等（0.76–0.82）；**context_integration 最弱**（連自評僅 0.55–0.57——文字特徵難預測「給了多少上下文」）；ambiguity 自評尚可但跨 labeler 掉至 ~0.43（主觀）。印證 6 維非等價：reasoning／verification 是可靠訊號，context_integration 是最弱維。
- **預設 router：維持 `learned`（不變）。** budget 為 opt-in 第三策略（`ROUTE_LLM_ROUTER=budget`），提供與 learned 同級路由 + 可解釋／tier／needs_tool／風險升級層。
- **level 門檻：維持 §5.1 起始值（4/8/12/17）。** budget 不任主幹，顯示 level 為 advisory；gold-fit 門檻會 in-sample 過擬合（天花板分析已示其極限），故不校準，留待日後若有獨立驗證集。
- **意義：** v3 的 gate 達成設計目的——以獨立人工 gold **否決**「6 維 composite 取代單一 learned 難度」的假設。budget 估計器在爭議難題上排序／分級不如 learned 且校準無法補救，故 v3 出貨為「learned 路由 + budget 決策層」：零路由退步、外加可解釋與風險升級。gemma 維度標註與 per-dimension 人工 gold 留待後續（不影響本判決）。
