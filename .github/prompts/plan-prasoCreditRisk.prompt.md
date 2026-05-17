# Plan: Praso Credit Risk Prediction Models

## TL;DR
Build two complementary credit risk models for Praso: (1) **Application Model** — predicts default for new clients using pre-registration data (Serasa, iFood, Google Maps), (2) **Behavioral Model** — predicts default for active clients by combining application features with aggregated purchase history. Develop via Jupyter notebook with EDA → feature engineering → two parallel modeling pipelines → interpretability analysis → actionable credit policy.

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

### Phase 6: Model Comparison & Selection
*Depends on Phase 4 & 5 (steps 15-20)*

21. Compare ROC-AUC, precision, recall across:
    - Application vs. Behavioral model (does behavioral data significantly improve predictions?)
    - Different algorithms (LR vs. XGBoost vs. RF)
    - Create side-by-side comparison table

22. Visualize: ROC curves for both models, feature importance for each

### Phase 7: Model Interpretability & Explainability
*Depends on Phase 6*

23. Build global interpretability artifacts:
    - Compute permutation importance for top models
    - Compute SHAP values (global) for feature impact ranking
    - Compare top drivers between application and behavioral models

24. Build local interpretability artifacts:
    - Explain individual predictions for representative clients (low/medium/high risk)
    - Validate whether explanations align with business intuition
    - Identify potential bias or proxy-risk features requiring governance

25. Produce explainability deliverables:
    - Plain-language explanation template for risk decisions
    - Model card section with assumptions, limitations, and known failure modes
    - Feature stability check across train/test and key segments (UF, segmento_cliente)

### Phase 8: Credit Policy Design & Application
*Depends on Phase 7*

26. Define decision thresholds:
    - At what default risk score do we approve/deny credit?
    - Trade-off: maximize customer acquisition vs. minimize defaults
    - Example thresholds: approve if score < 0.15 (15% default risk)

27. Design credit policy document addressing:
    - How to use application model for new clients (auto-approve, auto-deny, manual review tiers)
    - How to integrate behavioral model for recurring clients (risk reassessment frequency)
    - Risk-based limits (credit amount based on score)
    - Handling edge cases (new business with no history, high-risk but high-value client)

28. Scenario analysis: If we approve all clients with score < 0.30, what's expected loss? Approval rate?

## Relevant Files
- `dataset/credito_aplicacao_clientes_final.csv` — Application data (load as df_aplicacao)
- `dataset/credito_comportamental_pedidos_final.csv` — Order data (load as df_pedidos)
- `notebooks/01_eda_and_exploration.ipynb` — Primary notebook (all analysis & modeling)
- `notebooks/02_application_model.ipynb` (optional) — Separate notebook if needed
- `notebooks/03_behavioral_model.ipynb` (optional) — Separate notebook if needed
- `notebooks/04_interpretability.ipynb` (optional) — SHAP/permutation and local explanations

## Verification
1. **Data Loading**: Both CSVs load without errors; shape and dtypes match problem description
2. **EDA**: Visualizations show clear patterns (e.g., older businesses → lower default; payment delays correlate with default)
3. **Feature Engineering**: No data leakage; behavioral aggregations only use data before client default event
4. **Application Model**: ROC-AUC ≥ 0.65 on test set; top 5 features are interpretable
5. **Behavioral Model**: ROC-AUC higher than application model (due to behavioral signals); improvement is measurable
6. **Interpretability**: Global + local explanations are generated and reviewed for consistency and plausibility
7. **Credit Policy**: Document outlines clear decision rules with expected impact on default rate and approval rate
8. **Code Quality**: Notebook is reproducible; random seeds fixed; no hardcoded paths

## Decisions & Assumptions
- **Two separate models** (application + behavioral) rather than a single combined model, allowing reuse of application model for new clients
- **Jupyter notebook primary deliverable** for interactivity and exploration alignment with academic context
- **Stratified train-test split** to maintain default class balance
- **ROC-AUC as primary metric** per project brief; secondary metrics (precision, recall, F1) tracked for business context
- **Feature encoding**: One-hot encoding for categorical, median imputation for nulls (except designed boolean features from nulls)
- **No temporal leakage**: Behavioral model uses only order history available before default determination
- **Threshold selection is business decision**: Plan doesn't pick a threshold, but provides framework for decision
- **Explainability is mandatory**: Final recommendation requires both performance and interpretability evidence

## Further Considerations
1. **Behavioral Aggregation Window**: Should we use all historical orders, or only recent orders (e.g., last 6/12 months)? Recommendation: All history for now, but test recency impact later.
2. **Class Imbalance**: If default rate is low (<10%), consider SMOTE or class_weight in models. Test after initial EDA.
3. **Feature Selection**: With many features, consider correlation analysis, VIF (multicollinearity), or recursive feature elimination to simplify model & reduce overfitting.
