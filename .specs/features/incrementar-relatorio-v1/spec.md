---
name: incrementar-relatorio-v1
description: Identificar e documentar informações faltantes no Relatório AM v1 para produzir a versão final incrementada, em conformidade com CIN0144 (Tema 2)
metadata:
  type: feature
  scope: large
  created: 2026-06-17
  updated: 2026-06-17
  source: CIN0144 - Especificação de Projeto.pdf
---

# Spec — Incrementar Relatório AM v1

## Contexto

O relatório atual (`docs/Relatorio AM_v1.md`) está incompleto em relação ao que foi
implementado no projeto. Existem notebooks, métricas em JSON, código de roteamento e
análises de interpretabilidade que **não estão documentados** no relatório.

O objetivo é produzir uma versão v2 do relatório que reflita fielmente tudo que foi
construído, com métricas reais, arquitetura do sistema híbrido, política de crédito e
análise de importância de variáveis — **e que atenda aos requisitos institucionais da
disciplina CIN0144 (Aprendizado de Máquina e Ciência de Dados), Tema 2**.

**Tema escolhido:** Tema 2 — Predição de Aprovação de Acesso a Recursos  
**Área:** Classificação Binária Supervisionada (acesso concedido/negado → inadimplência/adimplência)

---

## Requisitos Institucionais — CIN0144

### Atividades obrigatórias do projeto

| ID | Atividade | Status no Projeto | Ação no Relatório |
|----|-----------|-------------------|-------------------|
| CIN-01 | Análise exploratória (missing, duplicatas, outliers, desbalanceamento) | Parcial — EDA existe, faltam duplicatas/outliers explícitos | Documentar o que foi feito; cobrir lacunas se implementadas |
| CIN-02 | Pré-processamento (encoding categórico, correlações, normalização) | Implementado | Detalhar no relatório (REQ-04) |
| CIN-03 | Separação treino/validação/teste | Implementado (70/15/15 estratificado) | Já no v1; confirmar no v2 |
| CIN-04 | Feature engineering (seleção e extração) | Implementado | Detalhar (REQ-05) |
| CIN-05 | ≥3 algoritmos distintos com justificativa | Implementado (LR, RF, XGBoost) | Já no v1; expandir justificativas |
| CIN-06 | Treinamento e tunagem de hiperparâmetros | Implementado | Documentar (REQ-06) |
| CIN-07 | Métricas adequadas (acurácia, precisão, recall, F1, AUC-ROC, tempo inferência) | Parcial — métricas existem, tempo de inferência não documentado | Adicionar tabela completa (REQ-08); incluir tempo se disponível |
| CIN-08 | Comparação entre modelos treinados | Parcial | Expandir (REQ-08, REQ-09) |
| CIN-09 | Conclusão com limitações, melhorias e implicações práticas | Parcial — genérica | Expandir (REQ-14) |

### Entregáveis obrigatórios

| ID | Entregável | Requisito | Status |
|----|-----------|-----------|--------|
| CIN-10 | Relatório em PDF (máx. 8 páginas) | Template institucional; cópia própria do template | Pendente — v2 deve ser exportado respeitando limite |
| CIN-11 | Apresentação (15 min) | Slides no formato CIn-UFPE | Fora do escopo desta feature (apenas relatório) |
| CIN-12 | Código no GitHub | Código completo, comentado, com instruções de execução | Verificar README; não é escopo do relatório |

### Estrutura mínima do relatório (CIN0144)

| ID | Seção exigida | Mapeamento no Relatório v2 |
|----|--------------|---------------------------|
| CIN-13 | **Introdução** — contextualização, relevância prática, objetivos | Seção I (+ REQ-02) |
| CIN-14 | **Análise de dados e FE** — descrição do dataset, EDA, pré-processamento, features | Seção II (+ REQ-03, REQ-04, REQ-05) |
| CIN-15 | **Modelagem** — algoritmos justificados, tunagem de hiperparâmetros e thresholds | Seção III (+ REQ-06, REQ-07, REQ-13) |
| CIN-16 | **Resultados** — gráficos e tabelas comparando modelos | Seção IV (+ REQ-08, REQ-09, REQ-10) |
| CIN-17 | **Conclusão** — descobertas, vantagens, limitações, insights, trabalhos futuros | Seção VII (+ REQ-14) |

### Prazos e restrições

| ID | Requisito | Valor |
|----|-----------|-------|
| CIN-18 | Equipe | 4 integrantes |
| CIN-19 | Indicação do tema (Classroom) | até 19/10/2025 |
| CIN-20 | Entrega final (relatório, slides, código) | até 16/12/2025 |

---

## Requisitos do Tema 2 — Predição de Aprovação de Acesso a Recursos

| ID | Requisito específico | Status no Projeto | Ação no Relatório |
|----|---------------------|-------------------|-------------------|
| T2-01 | EDA detalhada: correlações, distribuições e outliers | Parcial — distribuições e bivariada parcialmente cobertas | Expandir (REQ-03); adicionar correlações e outliers se analisados |
| T2-02 | Estratégia para dados faltantes com justificativa | Implementado (flags de ausência, imputação) | Documentar justificativa explicitamente (REQ-04) |
| T2-03 | Análise de balanceamento entre classes | Parcial — mencionado (31,3% inadimplentes) | Expandir com impacto nas métricas |
| T2-04 | ≥1 técnica de reamostragem (SMOTE ou Random Undersampling) | **Não implementado** — apenas planejado em `docs/plan_03_modelos_calibracao.md` | **Gap crítico:** implementar OU documentar limitação e justificar uso de `class_weight` |
| T2-05 | Comparar resultados com e sem reamostragem | **Não implementado** | Depende de T2-04 |
| T2-06 | ≥3 métricas para classificação binária desbalanceada (Acurácia, Precisão, Recall, F1, AUC-ROC) | Implementado | Tabelas completas (REQ-08) |

### Critérios de aceitação — Tema 2

1. WHEN o relatório for revisado THEN SHALL conter justificativa explícita para tratamento de missing values (T2-02)
2. WHEN a seção de balanceamento for lida THEN SHALL explicar o impacto do desbalanceamento (31,3% vs 68,7%) nas métricas escolhidas (T2-03)
3. WHEN a modelagem for descrita THEN SHALL comparar resultados com e sem técnica de reamostragem, OU documentar por que não foi aplicada e qual alternativa foi usada (T2-04, T2-05)
4. WHEN os resultados forem apresentados THEN SHALL incluir no mínimo 3 métricas adequadas a cenário desbalanceado por modelo (T2-06)

---

## Diagnóstico — O Que Está Faltando

### Lacunas identificadas por seção

| ID | Seção do Relatório | O Que Falta | Fonte de Dados |
|----|-------------------|-------------|----------------|
| REQ-01 | Abstract | Atualizar com métricas reais e mencionar o router | metrics JSONs |
| REQ-02 | I. Introdução | Adicionar contexto do ciclo de caixa (PME/PMR/PMP) | `Descrição problema Praso.md` |
| REQ-03 | II-A. EDA | Adicionar análise bivariada com inadimplência; mencionar gráficos chave | notebooks/01 |
| REQ-04 | II-B. Pré-processamento | Detalhar tratamento dos credores (lista → variáveis) e encoding CNAE | `src/feature_engineering.py` |
| REQ-05 | II-D. Feature Engineering | Listar as 18 features comportamentais usadas no modelo final | `models/05_behavioral_metrics_*.json` |
| REQ-06 | III. Modelagem | Adicionar seção de tuning de hiperparâmetros com busca e melhores params | metrics JSONs |
| REQ-07 | III. Modelagem | Adicionar seção explicando o sistema de roteamento (3 tiers) | `src/router.py`, `notebooks/04` |
| REQ-08 | IV. Resultados | Adicionar tabela completa de métricas (ROC-AUC, Precision, Recall, F1) | todos os metrics JSONs |
| REQ-09 | IV. Resultados | Comparar Router Rule-based vs ML Router vs Baseline | `models/06_router_metrics_*.json` |
| REQ-10 | IV. Resultados | Adicionar distribuição de clientes por tier do router | `models/06_router_metrics_*.json` |
| REQ-11 | IV. Resultados | Baseline do modelo comportamental vs modelo de aplicação puro | `models/05_behavioral_metrics_*.json` |
| REQ-12 | Nova seção V | Interpretabilidade — feature importance e análise SHAP (se disponível) | `notebooks/06_interpretability.ipynb` |
| REQ-13 | Nova seção VI | Política de crédito — thresholds, tiers, critérios de aprovação | `notebooks/07_credit_policy.ipynb` |
| REQ-14 | VI. Conclusão | Expandir com achados específicos e métricas reais | todo o acima |
| REQ-15 | II-A. EDA | Análise de correlações entre features e detecção de outliers | notebooks/01 |
| REQ-16 | II-A. EDA | Justificativa explícita da estratégia de missing values (T2-02) | notebooks/01, `src/feature_engineering.py` |
| REQ-17 | II-A. EDA | Discussão do impacto do desbalanceamento de classes nas métricas (T2-03) | relatório v1 + métricas |
| REQ-18 | III. Modelagem | Comparação com/sem reamostragem (SMOTE ou undersampling) ou justificativa da alternativa (T2-04/05) | notebooks/03 ou `plan_03` |
| REQ-19 | Formato | Adequar conteúdo ao limite de 8 páginas no template PDF (CIN-10) | template institucional |
| REQ-20 | IV. Resultados | Incluir tempo de inferência por modelo, se disponível (CIN-07) | notebooks de treino |

---

## Requisitos Funcionais

### REQ-01 — Abstract Atualizado
- Mencionar o router (regra + ML) como componente do sistema híbrido
- Confirmar ROC-AUC 0,850 para o sistema completo (rule-based router)
- Citar número de features comportamentais utilizadas

### REQ-02 — Contexto de Negócio (Introdução)
- Explicar o ciclo de caixa do pequeno varejista (PME, PMR, PMP)
- Contextualizar por que crédito instantâneo é uma vantagem competitiva da Praso
- Mencionar que a análise de crédito é feita no momento do cadastro

### REQ-03 — EDA Bivariada
- Análise de inadimplência por `idade_cnpj` — confirmação da tendência de queda
- Análise de `serasa_socio_tem_negativacao` vs inadimplência
- Taxa de inadimplência por quantidade de protestos
- Menção às análises de presença digital (iFood / Google Maps) vs inadimplência

### REQ-04 — Pré-processamento Detalhado
- Explicar como a coluna `serasa_credores` (lista de strings) foi transformada:
  - contagem de credores únicos
  - contagem de setores distintos
  - flag de credor do setor "Distribuição"
- Detalhar os níveis do CNAE extraídos (divisão, grupo, classe)

### REQ-05 — Features Comportamentais
Listar as 18 features usadas no modelo comportamental final:
```
orders_count, valor_min, valor_max, valor_mean, valor_median, valor_std,
delay_mean, delay_max, pct_orders_delayed, recency_days, orders_per_month,
log_valor_mean, log_valor_max, log_valor_min, log_valor_median, log_valor_std,
has_any_delay, delay_spike_ratio
```

### REQ-06 — Tuning de Hiperparâmetros

**Modelo de Aplicação (XGBoost Tuned):**
```
subsample: 0.7 | n_estimators: 100 | max_depth: 3
learning_rate: 0.05 | colsample_bytree: 0.8
```

**Modelo Comportamental (Random Forest Tuned):**
```
n_estimators: 100 | min_samples_leaf: 2
max_features: log2 | max_depth: None
```

### REQ-07 — Arquitetura do Sistema de Roteamento (3 Tiers)

O sistema classifica cada cliente em um de três tiers:

| Tier | Condição | Modelo Usado | Qtd Clientes |
|------|----------|-------------|-------------|
| BEHAVIORAL | Score_A ≥ 0,50 (tem histórico rico) | Modelo Comportamental | 664 (22,1%) |
| APPLICATION | Score_A < 0,50 e Score_B < 0,60 | Modelo de Aplicação | 998 (33,2%) |
| MANUAL_REVIEW | Score_A < 0,50 e Score_B ≥ 0,60 | Revisão humana | 1.338 (44,6%) |

Thresholds: `threshold_A = 0,50` | `threshold_B = 0,60`

### REQ-08 — Tabela Completa de Métricas

**Modelo de Aplicação (conjunto de teste):**

| Modelo | ROC-AUC | Accuracy | Precision | Recall | F1 |
|--------|---------|----------|-----------|--------|----|
| Logistic Regression | 0,749 | 0,684 | 0,497 | 0,667 | 0,570 |
| Random Forest | 0,738 | 0,723 | 0,579 | 0,429 | 0,493 |
| XGBoost | 0,733 | 0,704 | 0,528 | 0,543 | 0,535 |
| **XGBoost Tuned** | **0,772** | **0,701** | **0,517** | **0,723** | **0,603** |

**Modelo Comportamental (conjunto de teste, n=664):**

| Modelo | ROC-AUC | Accuracy | Precision | Recall | F1 |
|--------|---------|----------|-----------|--------|----|
| Baseline (App-only) | 0,623 | 0,620 | 0,270 | 0,476 | 0,345 |
| Logistic Regression | 0,733 | 0,695 | 0,383 | 0,738 | 0,504 |
| XGBoost | 0,760 | 0,790 | 0,500 | 0,476 | 0,488 |
| Random Forest | 0,775 | 0,795 | 0,556 | 0,119 | 0,196 |
| **Random Forest Tuned** | **0,770** | **0,795** | **0,514** | **0,429** | **0,468** |

### REQ-09 — Comparação de Routers

| Sistema | ROC-AUC | Δ vs Baseline |
|---------|---------|--------------|
| Baseline (apenas Modelo de Aplicação) | 0,804 | — |
| **Router Rule-based (atual)** | **0,850** | **+4,56pp** |
| ML Router (Decision Tree) | 0,848 | +4,32pp |

> O router baseado em regras supera levemente o ML router, indicando que as regras de roteamento atuais são bem calibradas.

### REQ-10 — Distribuição por Tier
- BEHAVIORAL: 664 clientes (22,1%) — usam modelo comportamental
- APPLICATION: 998 clientes (33,2%) — usam modelo de aplicação
- MANUAL_REVIEW: 1.338 clientes (44,6%) — enviados para análise humana

### REQ-11 — Baseline Comportamental
- O modelo comportamental (RF Tuned, AUC 0,770) supera em **+14,67pp** o baseline
  que usa apenas o modelo de aplicação para clientes com histórico (AUC 0,623)
- Isso confirma o valor incremental do histórico transacional para clientes recorrentes

### REQ-12 — Interpretabilidade
- Listar as variáveis mais importantes do Modelo de Aplicação (via SHAP ou feature_importances_)
- Listar as variáveis mais importantes do Modelo Comportamental
- Análise de sentido: confirmar que `idade_cnpj`, `serasa_negativacoes`, `delay_mean` e
  `pct_orders_delayed` estão entre os top features

### REQ-13 — Política de Crédito
- Explicar os thresholds sugeridos por tier:
  - BEHAVIORAL: aprovar se `p < 0,30`
  - APPLICATION: aprovar se `p < 0,15` (threshold mais conservador — sem histórico)
  - MANUAL_REVIEW: análise humana com critérios adicionais
- Mencionar que a escolha do threshold é uma decisão de negócio (trade-off risco × aquisição)
- Calcular: para cada threshold, qual seria a taxa de aprovação e a inadimplência esperada

### REQ-14 — Conclusão Expandida
- Referenciar métricas reais ao invés de afirmações genéricas
- Mencionar o ganho de +14,67pp do modelo comportamental sobre o baseline de aplicação
- Mencionar que o router rule-based atingiu AUC 0,850 vs baseline de 0,804 (+4,56pp)
- Reafirmar limitações com dados concretos (664/3000 = 22,1% têm dados comportamentais)
- Incluir implicações práticas para concessão de crédito na Praso (CIN-09, CIN-17)
- Apontar trabalhos futuros (ex.: reamostragem, survival analysis, mais dados comportamentais)

### REQ-15 — Correlações e Outliers (T2-01)
- Apresentar análise de correlação entre features numéricas relevantes
- Mencionar tratamento ou análise de outliers identificados na EDA
- Referenciar visualizações geradas nos notebooks (heatmap, boxplots)

### REQ-16 — Justificativa de Missing Values (T2-02)
- Explicar por que nulos em iFood/Google Maps foram tratados como "ausência de presença digital" (não imputação cega)
- Descrever estratégia para demais colunas com missing (imputação, exclusão ou flag)
- Justificar a escolha com base no significado de negócio de cada atributo

### REQ-17 — Impacto do Desbalanceamento (T2-03)
- Quantificar a proporção de classes (31,3% inadimplentes / 68,7% adimplentes)
- Discutir por que AUC-ROC e F1 são mais informativos que acurácia pura neste cenário
- Mencionar uso de estratificação na divisão treino/validação/teste

### REQ-18 — Reamostragem ou Alternativa (T2-04, T2-05)
**Cenário A (preferencial — se implementado):**
- Aplicar SMOTE ou Random Undersampling no treino
- Comparar métricas dos modelos com e sem reamostragem (tabela comparativa)
- Discutir trade-offs (recall vs precision)

**Cenário B (se não implementado):**
- Documentar explicitamente que reamostragem não foi aplicada
- Justificar uso de alternativas: `class_weight='balanced'`, estratificação, ou threshold tuning
- Registrar como limitação e trabalho futuro

### REQ-19 — Limite de 8 Páginas (CIN-10)
- Consolidar conteúdo para caber no template PDF institucional (máx. 8 páginas)
- Priorizar tabelas e figuras essenciais; mover detalhes extensos para apêndice do repositório se necessário
- Exportar a partir do template oficial (cópia própria da equipe)

### REQ-20 — Tempo de Inferência (CIN-07)
- Reportar tempo médio de predição por modelo (se medido nos notebooks)
- Discutir relevância para decisão de crédito em tempo real no cadastro

---

## Requisitos de Qualidade

- **RQ-01**: Todas as métricas no relatório devem ser rastreáveis para os arquivos JSON em `models/`
- **RQ-02**: Nenhuma métrica pode ser inventada — usar apenas valores dos arquivos de métricas
- **RQ-03**: O relatório deve seguir formato acadêmico (IEEE/ABNT-like) com seções numeradas
- **RQ-04**: Tabelas devem usar Markdown para compatibilidade com exportação
- **RQ-05**: Notebooks devem ser consultados para confirmar o comportamento descrito
- **RQ-06**: Relatório final deve cobrir a estrutura mínima exigida por CIN0144 (CIN-13 a CIN-17)
- **RQ-07**: Requisitos do Tema 2 (T2-01 a T2-06) devem estar endereçados ou explicitamente justificados como limitação
- **RQ-08**: PDF final não deve exceder 8 páginas (CIN-10)

---

## Requirement Traceability

| Requirement ID | Story/Área | Fase | Status |
| -------------- | ---------- | ---- | ------ |
| CIN-01 a CIN-09 | Atividades do projeto | Relatório | Pending |
| CIN-10 | Entregável PDF | Exportação | Pending |
| CIN-13 a CIN-17 | Estrutura do relatório | Relatório | Partial |
| T2-01 | REQ-03, REQ-15 | EDA | Pending |
| T2-02 | REQ-16 | Pré-processamento | Pending |
| T2-03 | REQ-17 | EDA | Pending |
| T2-04/05 | REQ-18 | Modelagem | **Gap** |
| T2-06 | REQ-08 | Resultados | Partial |
| REQ-01 a REQ-14 | Lacunas v1 → v2 | Relatório | Pending |
| REQ-19 | Formato PDF | Exportação | Pending |
| REQ-20 | Tempo inferência | Resultados | Pending |

**Coverage:** 35 requisitos totais (CIN + T2 + REQ), 0 verificados, 1 gap crítico (T2-04/05 reamostragem)

---

## O Que NÃO Fazer

- Não inventar métricas de calibração — não foram executadas no projeto atual
- Não adicionar survival analysis — está nos planos mas não implementado
- Não adicionar LightGBM/CatBoost — não foram treinados (planos futuros)
- Não reescrever seções que já estão corretas
- Não omitir o gap de reamostragem (T2-04) — documentar honestamente se não implementado
- Não exceder 8 páginas no PDF final sem priorizar conteúdo obrigatório do CIN0144
