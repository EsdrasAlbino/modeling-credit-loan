# Plan: Praso Credit Risk Prediction Models

## TL;DR
Build a **Model Routing system** for Praso credit risk: (1) **Application Model** — predicts default for new clients using pre-registration data (Serasa, iFood, Google Maps), (2) **Behavioral Model** — predicts default for active clients using application + purchase history, (3) **Model Router** — a meta-layer that scores each client on two dimensions (data richness + profile complexity) and dynamically routes them to the most appropriate model. Inspired by Nubank's scoring-based dispatch architecture where multiple scorers vote on which model handles each case best.

## Steps

### Phase 1: Environment Setup & Data Loading
1. Create project structure: `/notebooks/` (analysis) and `/data/` (input/output)
2. Create initial Jupyter notebook (`01_eda_and_exploration.ipynb`)
3. Install dependencies (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn)
4. Load both CSV files into notebook, validate data shape & types

### Phase 2: Exploratory Data Analysis (EDA)
*Depends on Phase 1*
5. **Application Data Exploration**:
   - Summary statistics for continuous variables (capital_social, idade_cnpj, serasa fields)
   - Categorical distributions (uf, segmento_cliente, natureza_juridica, fonte_cliente)
   - Check for nulls and patterns (e.g., ifood/google_maps missing indicates "not on platform")
   - Default rate (`inadimplente`) overall and by key segments

6. **Behavioral Data Exploration**:
   - Order value distributions (min, max, median, variance per client)
   - Payment delay patterns (atraso: 0 vs. delayed)
   - Order frequency per client
   - Link back to `inadimplente` at client level

7. **Key Visualizations**:
   - Default rate vs. idade_cnpj (expect higher default in newer businesses)
   - Default rate by segmento_cliente and uf
   - Default rate by serasa indicators (negativacoes, protestos)
   - Default rate by iFood/Google presence
   - Order value, delay frequency, and default correlation (behavioral)

8. **Findings Document**: Write observations as markdown cells — what's surprising, what patterns emerge

### Phase 3: Feature Engineering
*Depends on Phase 2*

**For Application Model (new clients)**:
9. Clean & encode application data:
   - Continuous vars: Handle nulls (median/mode), scale if needed
   - Categorical: One-hot encode (or target encoding for high cardinality)
   - Create derived features:
     - `has_serasa_negativacao` (boolean from serasa_contagem_negativacoes > 0)
     - `has_ifood` (boolean from ifood_contagem_avaliacoes != null)
     - `has_google_maps` (boolean from google_maps_avaliacao != null)
     - `idade_cnpj_midpoint` (extract midpoint from interval, e.g., 150-250 → 200)
     - CNAE breakdown (division/group/class/subclass if possible)
     - Creditor diversity (count distinct creditor types, count sectors represented)

**For Behavioral Model (recurring clients)**:
10. Aggregate behavioral data per client from orders:
    - Order value stats: min, max, mean, median, std per client
    - Delay stats: mean delay, max delay, % of orders with delay
    - Order frequency: total orders, orders per month
    - Recency: days since last order
    - Join aggregated features to application data on `id_cliente`

11. Create combined dataset for behavioral model (application features + behavioral aggregates)

### Phase 4: Application Model Development
*Depends on Phase 3 (steps 9)*

12. Split application data: 70% train, 30% test (stratified by `inadimplente`)
13. Baseline: Logistic Regression (interpretability)
    - Train on scaled features
    - Evaluate: ROC-AUC, precision, recall, F1
    - Extract feature importance (coefficients)

14. Advanced models (in parallel):
    - XGBoost classifier (best performance potential)
    - Random Forest (balance performance & interpretability)
    - Compare ROC-AUC, precision-recall curves

15. Hyperparameter tuning (GridSearchCV or RandomizedSearchCV on best model)
16. Final evaluation: Confusion matrix, ROC curve, feature importance plot
17. Save best model & create prediction function

### Phase 5: Behavioral Model Development
*Depends on Phase 3 (steps 10-11)*

18. Split behavioral data: 70% train, 30% test (stratified by `inadimplente`)
19. Repeat steps 13-17 for behavioral model using combined application + behavioral features
20. Compare feature importance: Which behavioral signals (order delays, value patterns) matter most?

### Phase 6: Model Router Development
*Depends on Phase 4 & 5 (steps 17-20)*

21. **Build Router Score A — Data Richness Score**:
    - Quantify behavioral data quality per client:
      - `n_orders`: number of historical orders
      - `behavioral_coverage_months`: time span of order history
      - `delay_variance`: stability of payment behavior
    - Combine into a single `data_richness_score` (e.g., weighted composite or Logistic Regression trained to predict behavioral model's uplift over application model)
    - Threshold: `data_richness_score >= 0.5` → route to Behavioral Model

22. **Build Router Score B — Profile Complexity Score**:
    - Measure how uncertain / borderline the Application Model is for a client:
      - Use Application Model's predicted probability distance from 0.5 (entropy)
      - `complexity_score = 1 - |p - 0.5| * 2` (1 = maximally uncertain, 0 = very clear cut)
    - Clients with high complexity AND low data richness → flag for manual review tier

23. **Build the Router (Meta-Model)**:
    - Inputs: `data_richness_score`, `complexity_score`, optional: `n_orders`, `inadimplente` history
    - Output: routing decision — `APPLICATION` | `BEHAVIORAL` | `MANUAL_REVIEW`
    - Implementation options (evaluate both):
      - **Rule-based router**: decision tree with explicit thresholds (interpretable, auditable)
      - **ML router**: Logistic Regression / lightweight XGBoost trained on which model historically performed better per client segment
    - Validate: Does routing improve overall system ROC-AUC vs. picking a single model for all clients?

24. **Routing evaluation**:
    - Compute system-level metrics: what % of clients are routed to each tier?
    - Compare: single-model baseline vs. router-dispatched system on held-out test set
    - Visualize routing decision boundaries using `data_richness_score` × `complexity_score` 2D plot

### Phase 7: Model Comparison & Selection
*Depends on Phase 4, 5 & 6 (steps 15-24)*

25. Compare ROC-AUC, precision, recall across:
    - Application vs. Behavioral model (standalone)
    - **Router-dispatched system** vs. both standalone models
    - Rule-based router vs. ML router
    - Create side-by-side comparison table including routing overhead

26. Visualize: ROC curves for all three pipelines (Application / Behavioral / Router-dispatched), routing decision boundary plot, per-tier accuracy breakdown

### Phase 8: Model Interpretability & Explainability
*Depends on Phase 7*

27. Build global interpretability artifacts:
    - Compute permutation importance for top models
    - Compute SHAP values (global) for feature impact ranking
    - Compare top drivers between application and behavioral models

28. Build local interpretability artifacts:
    - Explain individual predictions for representative clients (low/medium/high risk)
    - For each client, also explain **why the router sent them to that model** (routing rationale)
    - Validate whether explanations align with business intuition
    - Identify potential bias or proxy-risk features requiring governance

29. Produce explainability deliverables:
    - Plain-language explanation template for risk decisions (includes routing decision)
    - Model card section for each model + the router itself (assumptions, limitations, known failure modes)
    - Feature stability check across train/test and key segments (UF, segmento_cliente)

### Phase 9: Credit Policy Design & Application
*Depends on Phase 8*

30. Define decision thresholds **per routing tier**:
    - Application Model tier: approve if `score < 0.20` (tighter — less information available)
    - Behavioral Model tier: approve if `score < 0.30` (behavioral data reduces uncertainty)
    - Manual Review tier: human analyst reviews, router flags reason (high complexity, data gap)

31. Design credit policy document addressing:
    - **Model Routing flow**: entry point → scoring → dispatch → decision (diagram)
    - How to use application model for new clients (auto-approve, auto-deny, manual review tiers)
    - How to integrate behavioral model for recurring clients (risk reassessment frequency)
    - Risk-based limits (credit amount based on score and routing tier)
    - Handling edge cases (new business with no history, high-risk but high-value client)
    - Router governance: who can override routing decisions and under what conditions?

32. Scenario analysis: Simulate routing on full dataset — what % of approvals come from each tier? What's expected loss per tier? Compare vs. single-model policy.

## Relevant Files
- `dataset/credito_aplicacao_clientes_final.csv` — Application data (load as df_aplicacao)
- `dataset/credito_comportamental_pedidos_final.csv` — Order data (load as df_pedidos)
- `notebooks/01_eda_and_exploration.ipynb` — EDA & feature engineering
- `notebooks/02_application_model.ipynb` — Application Model training & evaluation
- `notebooks/03_behavioral_model.ipynb` — Behavioral Model training & evaluation
- `notebooks/04_model_router.ipynb` — Router scores (data richness + complexity) + routing logic
- `notebooks/05_interpretability.ipynb` — SHAP/permutation, local explanations, routing rationale
- `src/router.py` — Router class: `score_data_richness()`, `score_profile_complexity()`, `route(client_id)`
- `src/application_model.py` — Application Model pipeline
- `src/behavioral_model.py` — Behavioral Model pipeline

## Verification
1. **Data Loading**: Both CSVs load without errors; shape and dtypes match problem description
2. **EDA**: Visualizations show clear patterns (e.g., older businesses → lower default; payment delays correlate with default)
3. **Feature Engineering**: No data leakage; behavioral aggregations only use data before client default event
4. **Application Model**: ROC-AUC ≥ 0.65 on test set; top 5 features are interpretable
5. **Behavioral Model**: ROC-AUC higher than application model (due to behavioral signals); improvement is measurable
6. **Router — Score A**: `data_richness_score` correlates with behavioral model's predictive lift over application model
7. **Router — Score B**: `complexity_score` is highest for clients near the default boundary (probability ≈ 0.5)
8. **Router — System**: Router-dispatched system achieves higher overall ROC-AUC than best standalone model alone
9. **Routing Distribution**: No single tier handles >80% of clients (routing is actually doing work, not collapsing to one model)
10. **Interpretability**: Global + local explanations generated for each model AND for routing decisions
11. **Credit Policy**: Document describes routing flow, per-tier thresholds, and expected impact by tier
12. **Code Quality**: Notebook is reproducible; random seeds fixed; no hardcoded paths; `router.py` has unit tests for `route()`

## Decisions & Assumptions
- **Model Routing architecture** (Application + Behavioral + Router) rather than a single ensemble, allowing explicit, auditable dispatch decisions — each client's routing can be explained
- **Two router scores** (data richness + profile complexity) mirror the Nubank pattern: one scorer evaluates data availability, another evaluates case difficulty, the router combines them to dispatch
- **Rule-based router as baseline**: Start with decision-tree thresholds before training an ML router — the rule-based version is auditable and sets the performance floor
- **ML router is optional uplift**: Only adopt if it measurably beats rule-based routing on ROC-AUC with acceptable interpretability trade-off
- **Jupyter notebook primary deliverable** for interactivity and exploration alignment with academic context; `src/router.py` extracted for reuse
- **Stratified train-test split** to maintain default class balance
- **ROC-AUC as primary metric** at model level; at system level, also track routing distribution and per-tier default rate
- **Feature encoding**: One-hot encoding for categorical, median imputation for nulls (except designed boolean features from nulls)
- **No temporal leakage**: Behavioral model uses only order history available before default determination; router scores derived from same snapshot
- **Per-tier thresholds**: Each routing destination has its own approval threshold (tighter for Application tier, looser for Behavioral tier)
- **Explainability is mandatory for router too**: Routing decision must be explainable — "client routed to Behavioral Model because data_richness_score=0.73 (≥0.5) and 14 historical orders available"

## Further Considerations
1. **Behavioral Aggregation Window**: Should we use all historical orders, or only recent orders (e.g., last 6/12 months)? Recommendation: All history for now, but test recency impact later.
2. **Class Imbalance**: If default rate is low (<10%), consider SMOTE or class_weight in models. Test after initial EDA.
3. **Feature Selection**: With many features, consider correlation analysis, VIF (multicollinearity), or recursive feature elimination to simplify model & reduce overfitting.
4. **Router Threshold Calibration**: The `data_richness_score` cutoff (0.5) and `complexity_score` cutoff are initial guesses — calibrate using Precision-Recall on the routing validation set, not just model ROC-AUC.
5. **Cold Start for Behavioral Model**: A client with only 1-2 orders may have a high richness score numerically but low behavioral signal quality. Consider a minimum order count (e.g., ≥5 orders) as a hard gate before richness score is even computed.
6. **Router Drift Monitoring**: As the client portfolio grows, routing distribution may shift (more clients qualify for Behavioral tier). Plan periodic re-evaluation of router thresholds.
7. **LLM/SLM Extension Path**: If the project evolves beyond traditional ML, the router architecture is directly extensible — Score A and Score B can gate between ML models and a language model for unstructured data (e.g., client descriptions, support notes). The dispatch interface stays the same.
