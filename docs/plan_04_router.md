# Plano 04 — Model Router

## Contexto

O router atual usa regras fixas baseadas em Score A e Score B com thresholds
definidos manualmente (A ≥ 0,5 → BEHAVIORAL; B ≥ 0,6 → MANUAL_REVIEW).

Resultados atuais:
- AUC do sistema (rule-based): **0,850**
- Distribuição: BEHAVIORAL=664 (22%), APPLICATION=998 (33%), MANUAL_REVIEW=1.338 (45%)
- Thresholds fixos: `threshold_A = 0,50` | `threshold_B = 0,60`

---

## Fase 1 — Otimização dos Thresholds Atuais

Os thresholds atuais foram definidos por regra. Otimizá-los via busca em grid
diretamente no AUC do sistema é a intervenção de menor esforço e maior retorno imediato.

```python
# notebooks/04e_router_threshold_optimization.ipynb
import numpy as np
from itertools import product
from sklearn.metrics import roc_auc_score

def system_auc(df, y_all, p_app, p_behav, threshold_a, threshold_b):
    # Score A e B
    score_a = score_data_richness(df)
    score_b = score_profile_complexity(df, app_model)

    # Aplicar thresholds
    tier = pd.Series('APPLICATION', index=df.index)
    tier[score_a >= threshold_a] = 'BEHAVIORAL'
    tier[(score_a < threshold_a) & (score_b >= threshold_b)] = 'MANUAL_REVIEW'

    # Predições combinadas
    p_sys = p_app.copy()
    p_sys[tier == 'BEHAVIORAL'] = p_behav[tier == 'BEHAVIORAL']

    return roc_auc_score(y_all, p_sys)


# Grid search
results = []
for ta, tb in product(np.arange(0.30, 0.80, 0.05), np.arange(0.30, 0.80, 0.05)):
    auc = system_auc(df, y_all, p_app, p_behav, ta, tb)
    results.append({'threshold_a': ta, 'threshold_b': tb, 'auc': auc})

df_results = pd.DataFrame(results).sort_values('auc', ascending=False)
print(df_results.head(10))

best = df_results.iloc[0]
print(f"\nMelhor AUC: {best['auc']:.4f}")
print(f"threshold_A = {best['threshold_a']:.2f}")
print(f"threshold_B = {best['threshold_b']:.2f}")
```

**Visualização do grid**:
```python
pivot = df_results.pivot(index='threshold_a', columns='threshold_b', values='auc')
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn')
plt.title('AUC do Sistema por threshold_A × threshold_B')
plt.savefig('data/07_router_threshold_grid.png', dpi=150)
```

---

## Fase 2 — Tiers Granulares

Dividir BEHAVIORAL em dois sub-tiers para aplicar thresholds de crédito diferenciados:

| Tier | Condição | Score A | Threshold sugerido |
|------|----------|---------|-------------------|
| `BEHAVIORAL_ATIVO` | Pedidos frequentes | Score A ≥ 0,75 | p < 0,30 |
| `BEHAVIORAL_MODERADO` | Poucos pedidos, mas tem histórico | 0,50 ≤ Score A < 0,75 | p < 0,22 |
| `APPLICATION` | Sem dados comportamentais, perfil claro | Score A < 0,50, Score B < 0,60 | p < 0,15 |
| `MANUAL_REVIEW` | Sem dados, perfil incerto | Score A < 0,50, Score B ≥ 0,60 | Analista |

```python
def route_granular(score_a, score_b, threshold_b=0.6):
    tier = pd.Series('APPLICATION', index=score_a.index)
    tier[score_a >= 0.75] = 'BEHAVIORAL_ATIVO'
    tier[(score_a >= 0.50) & (score_a < 0.75)] = 'BEHAVIORAL_MODERADO'
    tier[(score_a < 0.50) & (score_b >= threshold_b)] = 'MANUAL_REVIEW'
    return tier
```

**Atualizar `src/router.py`** para suportar `n_tiers=4` como parâmetro.

---

## Fase 3 — ML Router Treinado para Maximizar AUC

O ML Router atual aprende a **replicar as regras**. A versão v2 é treinada diretamente
para maximizar o AUC do sistema.

### Abordagem: geração de dataset de roteamento ótimo

```python
# notebooks/04f_router_ml_v2.ipynb
from sklearn.tree import DecisionTreeClassifier

# 1. Para cada cliente, testar qual tier gera a melhor predição
#    (menor erro em relação ao label real)
optimal_tiers = []
for idx in range(len(df)):
    errors = {}
    # BEHAVIORAL: usar p_behav se disponível, senão não aplicável
    if df.iloc[idx]['orders_count'] > 0:
        errors['BEHAVIORAL']    = abs(p_behav[idx] - y_all.iloc[idx])
    errors['APPLICATION']   = abs(p_app[idx] - y_all.iloc[idx])
    errors['MANUAL_REVIEW'] = abs(p_app[idx] - y_all.iloc[idx]) * 1.1  # penaliza
    optimal_tiers.append(min(errors, key=errors.get))

df_router = df.copy()
df_router['optimal_tier'] = optimal_tiers

# 2. Treinar classificador que aprende o tier ótimo
features_router = ['score_A', 'score_B', 'orders_per_month',
                   'pct_orders_delayed', 'delay_mean', 'p_app']
X_router = df_router[features_router].fillna(0)
y_router = df_router['optimal_tier']

router_v2 = DecisionTreeClassifier(max_depth=5, random_state=42)
router_v2.fit(X_router, y_router)
```

### Comparação v1 vs. v2

| Router | AUC do sistema | Distribuição MANUAL_REVIEW |
|--------|---------------|--------------------------|
| Rule-based (atual) | 0,850 | 44,6% |
| ML v1 (replica regras) | 0,848 | — |
| ML v2 (maximiza AUC) | A medir | A medir |

---

## Fase 4 — Champion/Challenger

Antes de substituir o router em produção, validar o novo em paralelo.

### Estrutura

```
Tráfego de clientes novos
        │
        ├── 80% → Router atual (champion) → decisão vigente
        └── 20% → Router v2 (challenger)  → decisão registrada mas não aplicada
```

### Implementação

```python
import random

def route_with_champion_challenger(client_data, champion, challenger,
                                   challenger_fraction=0.20, seed=42):
    random.seed(seed)
    use_challenger = random.random() < challenger_fraction

    champion_tier    = champion.route(client_data)
    challenger_tier  = challenger.route(client_data)

    # Logar ambas as decisões
    log_routing_decision(
        client_id       = client_data['id_cliente'],
        champion_tier   = champion_tier,
        challenger_tier = challenger_tier,
        is_challenger   = use_challenger,
        timestamp       = datetime.now(),
    )

    return challenger_tier if use_challenger else champion_tier
```

### Critério de promoção

Após 30 dias de coleta:
1. Calcular AUC do challenger no subconjunto de 20%
2. Se `AUC_challenger > AUC_champion + 0,005` → promover challenger
3. Documentar decisão com data e métricas no `models/router_versions.json`

---

## Atualização de `src/router.py`

| Adição | Método |
|--------|--------|
| Suporte a 4 tiers | `route_granular()` |
| Otimização de thresholds | `optimize_thresholds(X, y, p_app, p_behav)` |
| Champion/Challenger | `ModelRouter.route_with_challenger(df, challenger)` |
| Logging de decisões | `ModelRouter.log_decision(client_id, tier, score_a, score_b, p)` |

---

## Notebooks a Criar

```
notebooks/
  04e_router_threshold_optimization.ipynb  ← grid search de threshold_A × threshold_B
  04f_router_ml_v2.ipynb                   ← ML router treinado para maximizar AUC
  04g_router_champion_challenger.ipynb     ← simulação de A/B testing
```

---

## Cronograma

| Semana | Tarefa | Impacto esperado |
|--------|--------|-----------------|
| 1 | Grid search de thresholds (Fase 1) | +0,5–2pp de AUC |
| 2 | Tiers granulares BEHAVIORAL_ATIVO / MODERADO (Fase 2) | Thresholds de crédito mais precisos |
| 3–4 | ML Router v2 treinado para AUC (Fase 3) | +1–3pp vs. rule-based |
| 5+ | Champion/Challenger em produção (Fase 4) | Validação real |

---

## Critérios de Sucesso

| Métrica | Atual | Meta |
|---------|-------|------|
| AUC do sistema | 0,850 | ≥ 0,865 |
| % clientes em MANUAL_REVIEW | 44,6% | ≤ 35% |
| % clientes em BEHAVIORAL | 22,1% | ≥ 30% (com mais dados) |
| Thresholds documentados | Fixos | Versionados com data e AUC de validação |
