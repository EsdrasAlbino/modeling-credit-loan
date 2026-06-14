# Plano 03 — Modelos e Calibração de Probabilidade

## Contexto

Os modelos atuais são:
- **Modelo de Aplicação**: XGBoost tuned — ROC-AUC 0,772
- **Modelo Comportamental**: Random Forest tuned — ROC-AUC 0,770

Este plano cobre: substituição e comparação de algoritmos, tratamento de desbalanceamento,
ensemble, calibração de probabilidade e survival analysis.

---

## O que é Calibração de Probabilidade

### O problema

Quando o modelo retorna `p = 0,30`, isso deveria significar que 30% dos clientes com
esse score vão inadimplir. Na prática, XGBoost e Random Forest são bons em **ordenar**
clientes por risco, mas os **valores absolutos de probabilidade podem estar distorcidos**.

Exemplo real deste projeto:
- Modelo prevê `p = 0,30` para 500 clientes
- Se apenas 180 deles inadimpliram (36%), o modelo está **subestimando o risco**
- Se 120 inadimpliram (24%), está **superestimando**

Isso impacta diretamente a política de crédito: os thresholds `p < 0,25` e `p < 0,15`
só fazem sentido se `p` significar o que parece significar.

### Visualização — Curva de Calibração

```
Taxa real
  1.0 |               /
      |          /   ← modelo subestima risco (curva acima da diagonal)
  0.5 |     /
      |  / ← diagonal perfeita (calibrado)
  0.0 |/____________________
      0.0       0.5       1.0
              Score previsto
```

### Como corrigir

Após treinar o modelo base, aplica-se um ajuste de segunda etapa usando o
**conjunto de validação** (nunca o de treino):

| Método | Como funciona | Quando usar |
|--------|--------------|-------------|
| **Platt Scaling** | Treina uma regressão logística sobre os scores brutos | Dataset pequeno, distorção monotônica |
| **Isotonic Regression** | Ajuste não-paramétrico, mais flexível | Dataset maior (≥ 1.000 exemplos de validação) |

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# Calibrar sobre o conjunto de validação
calibrated_app = CalibratedClassifierCV(
    app_model, method='sigmoid', cv='prefit'
)
calibrated_app.fit(X_val, y_val)

# Visualizar antes e depois
fig, ax = plt.subplots(figsize=(7, 5))
for model, label in [(app_model, 'Sem calibração'), (calibrated_app, 'Com calibração')]:
    prob_pred = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, prob_pred, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker='o', label=label)

ax.plot([0, 1], [0, 1], 'k--', label='Calibração perfeita')
ax.set_xlabel('Score previsto')
ax.set_ylabel('Taxa real de inadimplência')
ax.legend()
ax.set_title('Curva de Calibração — Modelo de Aplicação')
plt.savefig('data/calibration_curve.png', dpi=150)
```

### Impacto na política de crédito

| Antes da calibração | Depois da calibração |
|--------------------|---------------------|
| `p = 0,25` pode significar 18% ou 32% de risco real | `p = 0,25` ≈ 25% de risco real |
| Threshold escolhido empiricamente | Threshold tem interpretação direta |
| Estimativa de perda esperada imprecisa | `perda = p × limite` é confiável |

---

## Fase 1 — Comparação de Algoritmos

### LightGBM

```python
# notebooks/04b_application_model_lgbm.ipynb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

lgbm_params = {
    'n_estimators':    [200, 500, 1000],
    'learning_rate':   [0.01, 0.05, 0.1],
    'max_depth':       [3, 5, 7, -1],
    'num_leaves':      [15, 31, 63],
    'subsample':       [0.6, 0.8, 1.0],
    'colsample_bytree':[0.6, 0.8, 1.0],
    'min_child_samples':[10, 20, 50],
}

lgbm = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(lgbm, lgbm_params, n_iter=80, cv=5,
                            scoring='roc_auc', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
```

Vantagens sobre XGBoost neste projeto:
- Treinamento mais rápido (histograma-based)
- Melhor performance em datasets com muitas features categóricas
- `categorical_feature` nativo — não requer encoding manual

### CatBoost

```python
# notebooks/04c_application_model_catboost.ipynb
from catboost import CatBoostClassifier

cat_features = ['cnae_divisao', 'segmento_cliente']  # passados como strings

model_cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
)
model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
```

Vantagem principal: **zero encoding manual** para categóricas — passa a coluna original.

### Tabela de Comparação (a preencher após execução)

| Modelo | AUC | AP | KS | Treino (s) | Calibrado? |
|--------|-----|----|----|-----------|------------|
| XGBoost (atual) | 0,772 | 0,637 | — | — | Não |
| LightGBM | — | — | — | — | — |
| CatBoost | — | — | — | — | — |
| XGBoost calibrado | — | — | — | — | Sim |

---

## Fase 2 — Tratamento de Desbalanceamento

Taxa atual: 31,3% de inadimplentes — moderada, mas pode ser tratada.

### Opção A — `scale_pos_weight` (XGBoost) / `class_weight` (RF)

```python
# XGBoost
ratio = (y_train == 0).sum() / (y_train == 1).sum()  # ~2.2
xgb = XGBClassifier(scale_pos_weight=ratio, ...)

# Random Forest
rf = RandomForestClassifier(class_weight='balanced', ...)
```

Mais simples, sem alterar os dados. Ponto de partida recomendado.

### Opção B — SMOTE (Synthetic Minority Oversampling)

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('clf',   XGBClassifier(**best_params)),
])
pipeline.fit(X_train, y_train)
```

> Instalar: `pip install imbalanced-learn`

Cuidado: aplicar SMOTE **apenas no conjunto de treino**, nunca no de validação/teste.

### Opção C — Focal Loss

Penaliza mais os exemplos difíceis (clientes no limiar de decisão):

```python
import xgboost as xgb
import numpy as np

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        p     = 1 / (1 + np.exp(-y_pred))
        pt    = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
        fl    = -alpha_t * (1 - pt)**gamma * np.log(pt + 1e-8)
        dfl   = fl * (gamma * np.log(pt + 1e-8) * pt - (1 - pt))
        d2fl  = fl * gamma * (1 - pt) * pt
        return dfl, d2fl
    return loss

model = xgb.train(
    params,
    dtrain,
    obj=focal_loss(gamma=2.0, alpha=0.25),
)
```

---

## Fase 3 — Ensemble (Stacking)

Combinar as predições do XGBoost e do RF com um meta-modelo leve:

```python
# notebooks/05c_stacking_ensemble.ipynb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('xgb', best_xgb_pipeline),
    ('rf',  best_rf_pipeline),
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=0.1),
    cv=5,
    stack_method='predict_proba',
    passthrough=False,
)
stacking.fit(X_train, y_train)
```

O meta-modelo aprende **quando confiar em cada modelo** — pode capturar que XGBoost
é melhor em perfis com muitas features Serasa e RF é melhor para features comportamentais.

---

## Fase 4 — Survival Analysis (avançado)

Em vez de prever *se* o cliente vai inadimplir, prever *quando*.

**Quando faz sentido**: se o dataset tiver a data do primeiro atraso (não apenas o flag).

```python
# pip install scikit-survival
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Structarr

# y precisa ser array estruturado: (evento: bool, tempo: float)
y_surv = Structarr(
    [('inadimplente', bool), ('dias_ate_primeiro_atraso', float)],
    shape=(len(y),)
)
y_surv['inadimplente']           = y.astype(bool)
y_surv['dias_ate_primeiro_atraso'] = df['dias_ate_default'].values

rsf = RandomSurvivalForest(n_estimators=200, random_state=42)
rsf.fit(X_train, y_surv_train)

# Retorna risco acumulado por tempo — pode definir threshold por prazo de crédito
risk_scores = rsf.predict(X_test)
```

---

## Notebooks a Criar

```
notebooks/
  04b_application_model_lgbm.ipynb      ← LightGBM no Modelo de Aplicação
  04c_application_model_catboost.ipynb  ← CatBoost no Modelo de Aplicação
  04d_calibration.ipynb                 ← Curva de calibração + Platt/Isotonic
  05b_behavioral_model_lgbm.ipynb       ← LightGBM no Modelo Comportamental
  05c_stacking_ensemble.ipynb           ← Meta-modelo combinando XGBoost + RF
  05d_survival_analysis.ipynb           ← Survival Analysis (se data disponível)
```

---

## Cronograma

| Semana | Tarefa | Prioridade |
|--------|--------|-----------|
| 1 | Calibração dos modelos atuais (Platt Scaling) | Alta — impacto imediato na política |
| 1 | Curva de calibração antes/depois | Alta |
| 2 | LightGBM no Modelo de Aplicação | Alta |
| 2 | `class_weight='balanced'` em todos os modelos | Média |
| 3 | CatBoost — comparação | Média |
| 3 | SMOTE — testar impacto no AUC | Média |
| 4 | Stacking XGBoost + RF | Média |
| 5+ | Survival Analysis (depende de dados de data) | Baixa |

---

## Critérios de Sucesso

| Métrica | Atual | Meta |
|---------|-------|------|
| AUC — Modelo de Aplicação | 0,772 | ≥ 0,790 |
| AUC — Modelo Comportamental | 0,770 | ≥ 0,790 |
| AUC — Sistema (router) | 0,850 | ≥ 0,870 |
| Calibração (ECE — Expected Calibration Error) | Não medido | ≤ 0,03 |
| Default rate real dos aprovados com `p < 0,25` | — | Δ ≤ 3pp vs. `p` previsto |
