---
name: tasks-incrementar-relatorio-v1
description: Tarefas atômicas para incrementar o Relatório AM v1 com as informações faltantes identificadas na spec, em conformidade com CIN0144 (Tema 2)
metadata:
  type: tasks
  feature: incrementar-relatorio-v1
  created: 2026-06-17
  updated: 2026-06-17
  status: ready
---

# Tasks — Incrementar Relatório AM v1

> Consultar [spec.md](spec.md) para os requisitos completos (CIN0144 + Tema 2 + lacunas v1).
> Arquivo alvo: `docs/Relatorio AM_v1.md` → `docs/Relatorio AM_v2.md`

---

## Fase 1 — Contexto e EDA (Base do relatório)

### TASK-01 — Expandir Introdução com Contexto de Negócio
**Requisito:** REQ-02
**Onde:** Seção `I. INTRODUÇÃO`, após o segundo parágrafo
**O Que Fazer:**
- Adicionar parágrafo explicando o ciclo de caixa do pequeno varejista
- Incluir conceitos PME, PMR, PMP de forma sucinta
- Contextualizar que a Praso oferece análise instantânea no cadastro como diferencial
**Fonte:** `docs/Descrição problema Praso.md` — primeiro parágrafo
**Done When:** Introdução menciona ciclo de caixa e prazo de avaliação de crédito tradicional (2-14 dias)
**Gate:** Texto revisado, sem reproduzir verbatim a fonte (adaptar academicamente)
**Dependência:** Nenhuma

---

### TASK-02 — Expandir Análise Exploratória (EDA Bivariada)
**Requisito:** REQ-03
**Onde:** Seção `II-A. Análise Exploratória dos Dados` — expandir após o parágrafo atual
**O Que Fazer:**
- Adicionar subseção de análise bivariada com inadimplência
- Mencionar as principais descobertas:
  1. Empresas mais antigas (>2.400 dias) têm inadimplência significativamente menor
  2. Clientes com sócios negativados têm risco superior à média
  3. Maior quantidade de protestos → maior risco
  4. Presença digital (iFood/Google Maps) correlaciona com menor inadimplência
- Citar que gráficos de distribuição foram gerados (ex: taxa de inadimplência × intervalos de `idade_cnpj`)
**Fonte:** `notebooks/01_eda_and_exploration.ipynb` (consultar para confirmar achados)
**Done When:** Seção lista pelo menos 4 achados bivariados com direção da relação
**Dependência:** Nenhuma

---

### TASK-02b — Adicionar Correlações e Outliers (CIN / T2-01)
**Requisito:** REQ-15, T2-01, CIN-01
**Onde:** Seção `II-A. Análise Exploratória dos Dados` — após TASK-02
**O Que Fazer:**
- Adicionar parágrafo sobre análise de correlação entre features numéricas (mencionar heatmap se gerado)
- Descrever outliers identificados e como foram tratados (winsorização, exclusão ou mantidos)
- Consultar `notebooks/01_eda_and_exploration.ipynb` para confirmar análises existentes
**Fonte:** `notebooks/01_eda_and_exploration.ipynb`
**Done When:** Seção menciona correlações e outliers com tratamento documentado
**Dependência:** TASK-02

---

### TASK-02c — Justificar Estratégia de Missing Values (T2-02)
**Requisito:** REQ-16, T2-02
**Onde:** Seção `II-B. Pré-processamento dos Dados` — expandir parágrafo sobre nulos
**O Que Fazer:**
- Explicitar que nulos em iFood/Google Maps = ausência de presença digital (não erro de coleta)
- Descrever estratégia para demais colunas com missing (flags, imputação ou exclusão)
- Justificar cada escolha com raciocínio de negócio
**Fonte:** `src/feature_engineering.py`, `notebooks/01_eda_and_exploration.ipynb`
**Done When:** Pré-processamento contém justificativa explícita para missing values (requisito T2-02)
**Dependência:** TASK-02

---

### TASK-02d — Discutir Impacto do Desbalanceamento (T2-03)
**Requisito:** REQ-17, T2-03, CIN-01
**Onde:** Seção `II-A. Análise Exploratória dos Dados` — após parágrafo de distribuição de classes
**O Que Fazer:**
- Quantificar proporção 31,3% / 68,7%
- Explicar por que acurácia isolada pode ser enganosa neste cenário
- Mencionar que AUC-ROC, F1 e estratificação foram escolhidos como resposta ao desbalanceamento
**Fonte:** Relatório v1 + métricas em `models/`
**Done When:** Parágrafo discute impacto do desbalanceamento nas métricas de avaliação
**Dependência:** TASK-02

---

## Fase 2 — Pré-processamento e Feature Engineering

### TASK-03 — Detalhar Tratamento dos Credores Serasa
**Requisito:** REQ-04
**Onde:** Seção `II-B. Pré-processamento dos Dados`, após o parágrafo sobre encoding
**O Que Fazer:**
- Adicionar parágrafo descrevendo como `serasa_credores` (lista de até 5 credores) foi tratada:
  - Contagem de credores únicos (`n_credores`)
  - Contagem de setores distintos entre os credores (`n_setores_credores`)
  - Interação entre negativações e número de credores
**Fonte:** `src/feature_engineering.py`
**Done When:** Pré-processamento cita o tratamento da variável de credores como lista
**Dependência:** TASK-02

---

### TASK-04 — Adicionar Seção de Features Comportamentais
**Requisito:** REQ-05
**Onde:** Seção `II-D. Feature Engineering` — adicionar subseção ao final
**O Que Fazer:**
- Adicionar subseção "Features Comportamentais" listando as 18 variáveis derivadas dos pedidos
- Agrupar por categoria: métricas de valor, métricas de atraso, métricas de frequência, features log-transformadas
- Breve justificativa para as features de delay (delay_spike_ratio, has_any_delay)
**Fonte:** `models/05_behavioral_metrics_20260614_105248.json` → campo `_behavioral_features`
**Done When:** Seção lista as 18 features agrupadas logicamente
**Dependência:** TASK-03

---

## Fase 3 — Modelagem

### TASK-05 — Adicionar Detalhes de Tuning de Hiperparâmetros
**Requisito:** REQ-06
**Onde:** Seção `III. MODELAGEM`, após as subseções de cada algoritmo
**O Que Fazer:**
- Adicionar subseção `III-D. Otimização de Hiperparâmetros`
- Descrever busca via RandomizedSearchCV / GridSearchCV com validação cruzada
- Apresentar os melhores hiperparâmetros encontrados para cada modelo:
  - XGBoost (App): subsample=0.7, n_estimators=100, max_depth=3, lr=0.05, colsample=0.8
  - Random Forest (Comportamental): n_estimators=100, min_samples_leaf=2, max_features=log2
- Mencionar que AUC de validação foi a métrica de otimização
**Fonte:** `models/04_application_metrics_*.json` → `_best_params`; `models/05_behavioral_metrics_*.json` → `_best_params`
**Done When:** Subseção lista os hiperparâmetros finais de cada modelo com contexto de como foram encontrados
**Dependência:** Nenhuma

---

### TASK-05b — Reamostragem: Implementar ou Documentar Gap (T2-04, T2-05) ⚠️ CRÍTICO
**Requisito:** REQ-18, T2-04, T2-05
**Onde:** Seção `III. MODELAGEM` — nova subseção `III-F. Tratamento de Desbalanceamento`
**O Que Fazer:**

**Opção A (preferencial):** Implementar no notebook de treino e documentar:
1. Aplicar SMOTE ou Random Undersampling **apenas no treino**
2. Treinar pelo menos um modelo com e sem reamostragem
3. Tabela comparativa de métricas (AUC, F1, Precision, Recall)

**Opção B (se não implementar):** Documentar no relatório:
1. Declarar que reamostragem não foi aplicada
2. Justificar alternativas usadas (`class_weight`, estratificação, threshold tuning)
3. Registrar como limitação na conclusão

**Fonte:** `docs/plan_03_modelos_calibracao.md`, `notebooks/03_*` (se existir)
**Done When:** Relatório contém comparação com/sem reamostragem OU justificativa explícita da alternativa
**Dependência:** TASK-05
**Nota:** Gap crítico para conformidade com Tema 2 — priorizar antes da entrega

---

### TASK-06 — Adicionar Seção de Arquitetura do Sistema de Roteamento
**Requisito:** REQ-07
**Onde:** Seção `III. MODELAGEM`, após TASK-05 — nova subseção `III-E. Sistema de Roteamento`
**O Que Fazer:**
- Explicar a arquitetura de 3 tiers do sistema híbrido
- Descrever Score_A (riqueza de dados comportamentais) e Score_B (complexidade/risco cadastral)
- Apresentar a tabela de tiers com condições, modelo usado e quantidade de clientes
- Explicar os thresholds: threshold_A = 0,50 | threshold_B = 0,60
- Mencionar que foi também treinado um ML Router (Decision Tree) comparativamente
**Fonte:** `src/router.py`, `models/06_router_metrics_*.json`, `notebooks/04_model_router.ipynb`
**Done When:** Seção explica os 3 tiers com condições de roteamento e quantidades de clientes
**Dependência:** TASK-05

---

## Fase 4 — Resultados

### TASK-07 — Expandir Tabela de Resultados (Métricas Completas)
**Requisito:** REQ-08, REQ-11
**Onde:** Seção `IV. ANÁLISE E COMPARAÇÃO DOS RESULTADOS` — substituir/expandir texto atual
**O Que Fazer:**
- Adicionar Tabela 1: Métricas do Modelo de Aplicação (todos os modelos comparados)
  - Colunas: Modelo | ROC-AUC | Accuracy | Precision | Recall | F1
  - Linhas: LR, RF, XGBoost, XGBoost Tuned (destacar o melhor)
- Adicionar Tabela 2: Métricas do Modelo Comportamental
  - Incluir linha de Baseline (App-only) com AUC 0,623 como referência
  - Mostrar lift de +14,67pp do RF Tuned sobre o baseline
- Adicionar parágrafo contextualizando cada tabela
**Fonte:** `models/04_application_metrics_20260611_174247.json`, `models/05_behavioral_metrics_20260614_105248.json`
**Done When:** Duas tabelas presentes, com valores exatos dos JSONs e linha de baseline no modelo comportamental
**Dependência:** TASK-06

---

### TASK-08 — Adicionar Comparação de Routers e Distribuição por Tier
**Requisito:** REQ-09, REQ-10
**Onde:** Seção `IV. ANÁLISE E COMPARAÇÃO DOS RESULTADOS`, após Tabela 2 (TASK-07)
**O Que Fazer:**
- Adicionar Tabela 3: Comparação de sistemas de roteamento
  - Baseline (só App) AUC 0,804 → Rule-based Router AUC 0,850 (+4,56pp) → ML Router AUC 0,848 (+4,32pp)
- Adicionar Tabela 4: Distribuição de clientes por tier
  - BEHAVIORAL: 664 (22,1%) | APPLICATION: 998 (33,2%) | MANUAL_REVIEW: 1.338 (44,6%)
- Parágrafo explicando por que o router rule-based supera o ML router levemente
**Fonte:** `models/06_router_metrics_20260614_112028.json`
**Done When:** Tabelas 3 e 4 presentes com valores exatos do JSON de router
**Dependência:** TASK-07

---

## Fase 5 — Interpretabilidade e Política de Crédito

### TASK-09 — Adicionar Seção de Interpretabilidade
**Requisito:** REQ-12
**Onde:** Nova seção `V. INTERPRETABILIDADE DOS MODELOS` (antes da conclusão atual)
**O Que Fazer:**
1. Abrir o notebook `notebooks/06_interpretability.ipynb` e identificar:
   - Top features por importância no Modelo de Aplicação
   - Top features por importância no Modelo Comportamental
2. Escrever subseção para cada modelo listando as top 5-10 features
3. Confirmar que `idade_cnpj`, `serasa_negativacoes`, `delay_mean`, `pct_orders_delayed` aparecem no topo
4. Adicionar breve análise de sentido (direção esperada de cada feature)
**Fonte:** `notebooks/06_interpretability.ipynb`
**Done When:** Seção V lista top features de cada modelo com análise de sentido coerente com negócio
**Dependência:** TASK-08
**Nota:** Se o notebook não tiver outputs salvos, inferir a partir dos dados da EDA e dos planos

---

### TASK-10 — Adicionar Seção de Política de Crédito
**Requisito:** REQ-13
**Onde:** Nova seção `VI. POLÍTICA DE CRÉDITO` (após interpretabilidade)
**O Que Fazer:**
1. Abrir `notebooks/07_credit_policy.ipynb` e extrair os thresholds definidos
2. Escrever a seção com:
   - Explicação de que o threshold é decisão de negócio (trade-off risco × volume)
   - Tabela de thresholds sugeridos por tier:
     - BEHAVIORAL: aprovar se p < 0,30
     - APPLICATION: aprovar se p < 0,15
     - MANUAL_REVIEW: análise humana
   - Discussão sobre como o ROC-AUC mede capacidade de ordenação independente do threshold
3. Mencionar que a Praso pode ajustar os thresholds conforme metas de inadimplência tolerada
**Fonte:** `notebooks/07_credit_policy.ipynb`
**Done When:** Seção VI presente com tabela de thresholds e explicação do trade-off
**Dependência:** TASK-09

---

## Fase 6 — Revisão Final

### TASK-11 — Atualizar Abstract e Conclusão
**Requisito:** REQ-01, REQ-14
**Onde:** Abstract e Seção `VII. CONCLUSÃO E DISCUSSÃO` (renumerar após novas seções)
**O Que Fazer:**
1. **Abstract**: Adicionar menção ao sistema de roteamento; confirmar métricas (0,850 rule-based)
2. **Conclusão**: Expandir com achados específicos:
   - Lift de +14,67pp do modelo comportamental sobre baseline app-only
   - Gain de +4,56pp do router sobre baseline (app sozinho)
   - Confirmação de que `idade_cnpj` e indicadores Serasa são os preditores mais fortes
   - Referenciar que 44,6% dos clientes ainda vão para revisão manual (oportunidade futura)
3. Renumerar seções após inclusão das novas (V e VI)
**Fonte:** Todas as tasks anteriores
**Done When:** Abstract e Conclusão têm métricas reais, sem afirmações genéricas
**Dependência:** TASK-10 (todas as tasks anteriores devem estar completas)

---

## Fase 7 — Conformidade CIN0144

### TASK-12 — Verificar Tempo de Inferência (CIN-07)
**Requisito:** REQ-20, CIN-07
**Onde:** Seção `IV. ANÁLISE E COMPARAÇÃO DOS RESULTADOS` — parágrafo ou nota de rodapé
**O Que Fazer:**
- Verificar se notebooks de treino medem tempo de predição por modelo
- Se disponível: adicionar tabela ou menção ao tempo médio de inferência
- Se não disponível: registrar como limitação (não inventar valores)
**Fonte:** `notebooks/03_*`, `notebooks/04_*`
**Done When:** Tempo de inferência documentado ou limitação declarada
**Dependência:** TASK-07

---

### TASK-13 — Adequar ao Limite de 8 Páginas (CIN-10)
**Requisito:** REQ-19, CIN-10
**Onde:** Exportação final do relatório
**O Que Fazer:**
1. Copiar template institucional do relatório (link no CIN0144)
2. Transferir conteúdo do `Relatorio AM_v2.md` para o template
3. Priorizar seções obrigatórias (CIN-13 a CIN-17); condensar texto redundante
4. Verificar que PDF final ≤ 8 páginas
**Fonte:** Template institucional CIn-UFPE
**Done When:** PDF exportado com ≤ 8 páginas cobrindo estrutura mínima CIN0144
**Dependência:** TASK-11

---

### TASK-14 — Checklist de Conformidade CIN0144
**Requisito:** CIN-01 a CIN-17, T2-01 a T2-06, RQ-06, RQ-07
**Onde:** Revisão final antes da entrega
**O Que Fazer:**
- Percorrer checklist da spec (tabelas CIN e T2)
- Confirmar que cada requisito está endereçado no relatório ou justificado como limitação
- Marcar status na tabela de traceability em `spec.md`
**Done When:** Todos os itens CIN/T2 verificados ou justificados
**Dependência:** TASK-13

---

## Resumo de Dependências

```
TASK-01 ───────────────────────────────────────────────────────── TASK-11 → TASK-13 → TASK-14
TASK-02 → TASK-02b ─┐
       → TASK-02c ──┼→ TASK-03 → TASK-04
       → TASK-02d ──┘
TASK-05 → TASK-05b → TASK-06 → TASK-07 → TASK-08 → TASK-09 → TASK-10 ─┘
TASK-07 → TASK-12
```

## Critérios de Aceitação Global

- [ ] Todas as métricas numéricas rastreáveis para arquivos JSON em `models/`
- [ ] Nenhuma métrica inventada (calibração, LightGBM, etc. — não implementados)
- [ ] Sistema de roteamento explicado com 3 tiers e thresholds documentados
- [ ] Features comportamentais listadas (18 features)
- [ ] Tabelas de comparação de modelos completas (≥3 métricas por modelo — T2-06)
- [ ] Seção de política de crédito presente
- [ ] Seção de interpretabilidade presente
- [ ] Estrutura mínima CIN0144 coberta (Intro, Dados/FE, Modelagem, Resultados, Conclusão)
- [ ] Missing values justificados (T2-02)
- [ ] Desbalanceamento discutido (T2-03)
- [ ] Reamostragem comparada ou alternativa justificada (T2-04/05) ⚠️
- [ ] PDF final ≤ 8 páginas (CIN-10)
- [ ] Relatório v2 salvo como `docs/Relatorio AM_v2.md`

## Ordem de Execução Sugerida

1. TASK-01, TASK-02, TASK-02b, TASK-02c, TASK-02d em paralelo (EDA e contexto)
2. TASK-03 → TASK-04 (sequencial)
3. TASK-05 → **TASK-05b** (crítico — reamostragem) → TASK-06 (paralelo com TASK-03/04)
4. TASK-07 → TASK-08 → TASK-12 (resultados)
5. TASK-09 → TASK-10 (interpretabilidade e política)
6. TASK-11 (abstract e conclusão)
7. TASK-13 → TASK-14 (exportação PDF e checklist CIN)
