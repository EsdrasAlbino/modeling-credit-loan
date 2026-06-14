# Plano 02 — Engenharia de Features

## Contexto

O pipeline atual usa features diretas com transformações básicas (midpoint de intervalos,
extração de CNAE, log de valores de pedido). Há espaço significativo para criar
representações mais informativas sem coletar novos dados.

Estado atual:
- Modelo de Aplicação: 41 features
- Modelo Comportamental: 53 features
- AUC de referência: App=0,772 | Behavioral=0,770

---

## Fase 1 — Sem novos dados (implementar agora)

### 1.1 Target Encoding para CNAE

**Problema atual**: `cnae_divisao` é tratada como string categórica.
Modelos baseados em árvore podem não capturar bem a relação ordinal entre setores.

**Solução**: substituir pela **taxa média de inadimplência por divisão CNAE**,
calculada exclusivamente no conjunto de treino para evitar data leakage.

```python
# src/feature_engineering.py
from category_encoders import TargetEncoder

def add_cnae_target_encoding(df_train, df_val, df_test, target_col='inadimplente'):
    enc = TargetEncoder(cols=['cnae_divisao'], smoothing=10)
    df_train['cnae_default_rate'] = enc.fit_transform(
        df_train['cnae_divisao'], df_train[target_col]
    )
    df_val['cnae_default_rate']  = enc.transform(df_val['cnae_divisao'])
    df_test['cnae_default_rate'] = enc.transform(df_test['cnae_divisao'])
    return df_train, df_val, df_test, enc
```

> Instalar: `pip install category_encoders`

**Impacto esperado**: +1–2pp de AUC no Modelo de Aplicação.

---

### 1.2 Features de Interação

Combinações de features que capturam relações que o modelo sozinho não aprende facilmente:

| Feature nova | Fórmula | Intuição |
|-------------|---------|---------|
| `cnpj_x_credores` | `idade_cnpj_mid × serasa_n_setores` | Empresa nova com muitos credores é muito pior que cada variável separada |
| `capital_por_atividade` | `capital_social_mid / (orders_per_month + 1)` | Capital declarado vs. atividade real — discrepância é sinal de risco |
| `atraso_por_valor` | `delay_mean / (log_valor_mean + 1)` | Atraso ponderado pelo tamanho do pedido |
| `negativacao_x_credores` | `serasa_socio_tem_negativacao × serasa_n_setores` | Sócio negativo com múltiplos credores: risco combinado |

```python
def add_interaction_features(df):
    df['cnpj_x_credores']     = df['idade_cnpj_mid'] * df['serasa_n_setores']
    df['capital_por_atividade'] = df['capital_social_mid'] / (df['orders_per_month'].fillna(0) + 1)
    if 'delay_mean' in df.columns and 'log_valor_mean' in df.columns:
        df['atraso_por_valor'] = df['delay_mean'] / (df['log_valor_mean'] + 1)
    df['negativacao_x_credores'] = (
        df['serasa_socio_tem_negativacao'] * df['serasa_n_setores']
    )
    return df
```

---

### 1.3 Weight of Evidence (WoE)

**O que é**: transforma cada categoria de uma variável pelo log-odds de inadimplência
naquele grupo. Variáveis com WoE alto concentram maus pagadores.

**Fórmula**:
```
WoE(i) = ln( % bons na categoria i / % maus na categoria i )
```

**Quando usar**: variáveis binárias Serasa (negativação, protestos) e categóricas com
poucos valores. Melhora interpretabilidade e performance em dados desbalanceados.

```python
from category_encoders import WOEEncoder

woe_cols = [
    'serasa_socio_tem_negativacao',
    'cnae_divisao',
    'segmento_cliente',
]

enc_woe = WOEEncoder(cols=woe_cols, regularization=1.0)
X_train_woe = enc_woe.fit_transform(X_train[woe_cols], y_train)
X_val_woe   = enc_woe.transform(X_val[woe_cols])
```

---

### 1.4 Features Temporais por Janela (Behavioral)

**Problema atual**: as features comportamentais são agregadas sobre todo o histórico.
Um cliente que atrasou muito no passado mas melhorou nos últimos 3 meses parece igual
a um que sempre atrasou.

**Solução**: agregar por janelas de tempo (requer a data de cada pedido).

```python
def add_temporal_window_features(df_pedidos, reference_date):
    df_pedidos['data_pedido'] = pd.to_datetime(df_pedidos['data_pedido'])
    windows = {'30d': 30, '60d': 60, '90d': 90}
    results = {}
    for name, days in windows.items():
        cutoff = reference_date - pd.Timedelta(days=days)
        recent = df_pedidos[df_pedidos['data_pedido'] >= cutoff]
        agg = recent.groupby('id_cliente').agg(
            orders_count   =('id_pedido',      'count'),
            delay_mean     =('dias_atraso',     'mean'),
            pct_delayed    =('atrasou',         'mean'),
            valor_mean     =('valor_pedido',    'mean'),
        ).add_suffix(f'_{name}')
        results[name] = agg
    return pd.concat(results.values(), axis=1)
```

**Features geradas** (por janela × 4 métricas × 3 janelas = 12 features novas):
- `orders_count_30d`, `delay_mean_30d`, `pct_delayed_30d`, `valor_mean_30d`
- Idem para 60d e 90d

**Feature de tendência**:
```python
df['delay_trend'] = df['delay_mean_30d'] - df['delay_mean_90d']
# positivo = piorando; negativo = melhorando
```

---

### 1.5 Seleção de Features

Após criar novas features, remover as que adicionam ruído.

**Método 1 — Boruta** (recomendado para este projeto):
```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', random_state=42)
boruta.fit(X_train.values, y_train.values)

selected_features = X_train.columns[boruta.support_].tolist()
```

**Método 2 — Recursive Feature Elimination (RFE)**:
```python
from sklearn.feature_selection import RFECV

selector = RFECV(estimator=XGBClassifier(), cv=5, scoring='roc_auc', min_features_to_select=10)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.support_].tolist()
```

---

## Fase 2 — Com dados externos

| Feature | Fonte | Derivação |
|---------|-------|-----------|
| `situacao_cadastral_rf` | Receita Federal | Flag: ativa/suspensa/inapta |
| `data_abertura_oficial` | Receita Federal | Recomputar `idade_cnpj` com data oficial vs. declarada |
| `n_socios` | Receita Federal | Número de sócios no quadro societário |
| `inadimplencia_setorial` | Banco Central | Taxa de inadimplência histórica do setor CNAE |
| `pib_municipio` | IBGE | Proxy de saúde econômica da região |

---

## Notebooks a Criar

```
notebooks/
  03b_feature_engineering_v2.ipynb   ← target encoding + interações + WoE
  03c_temporal_features.ipynb        ← janelas temporais (se data disponível)
  03d_feature_selection.ipynb        ← Boruta / RFE
```

---

## Cronograma

| Semana | Tarefa | Arquivo |
|--------|--------|---------|
| 1 | Target encoding CNAE | `src/feature_engineering.py` |
| 1 | Features de interação | `src/feature_engineering.py` |
| 2 | WoE nas variáveis Serasa | `src/feature_engineering.py` |
| 2 | Retreinar modelos e comparar AUC | `notebooks/03b_feature_engineering_v2.ipynb` |
| 3 | Features temporais por janela (se data disponível) | `notebooks/03c_temporal_features.ipynb` |
| 4 | Seleção com Boruta | `notebooks/03d_feature_selection.ipynb` |

---

## Critérios de Sucesso

| Métrica | Atual | Meta |
|---------|-------|------|
| AUC — Modelo de Aplicação | 0,772 | ≥ 0,790 |
| AUC — Modelo Comportamental | 0,770 | ≥ 0,790 |
| Número de features | 41 / 53 | Reduzir em ≥ 20% após seleção |
| Feature mais importante | `idade_cnpj_mid` | Verificar se `cnae_default_rate` entra no top-3 |
