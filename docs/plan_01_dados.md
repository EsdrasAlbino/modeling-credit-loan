# Plano 01 — Expansão de Dados

## Contexto

O dataset atual possui 3.000 clientes, dos quais apenas 664 (22,1%) têm histórico
comportamental de pedidos. O restante é avaliado apenas com dados cadastrais e Serasa.
Expandir e enriquecer as fontes de dados é a alavanca de maior impacto a longo prazo.

---

## Fontes Disponíveis

### Públicas e Gratuitas

| Fonte | Dados disponíveis | Como acessar |
|-------|-------------------|--------------|
| **Receita Federal** | Situação cadastral, data de abertura, capital social declarado, quadro societário, CNAE, porte | API REST: `https://receitaws.com.br/v1/cnpj/{cnpj}` — sem autenticação |
| **Banco Central (API aberta)** | Taxas de inadimplência por setor, porte e região | `https://olinda.bcb.gov.br/olinda/servico/SCR` |
| **IBGE** | PIB municipal, índices por setor CNAE | `https://servicodados.ibge.gov.br/api/docs` |

### Pagas (Bureaus de Crédito)

| Fonte | Dados adicionais vs. versão atual | Observação |
|-------|-----------------------------------|------------|
| **Serasa Experian (completo)** | Score PJ, histórico de negativações com datas, protestos, falências, grupo econômico | Projeto já usa subset. Versão completa é paga |
| **Boa Vista SCPC / SPC Brasil** | Cobertura complementar ao Serasa em algumas regiões | Alternativa ou complemento |
| **Quod** | Dados do SCR (endividamento bancário total do CNPJ) | Bureau criado pelos 5 maiores bancos |

### Internas da Plataforma Praso

| Fonte | Features geráveis |
|-------|-------------------|
| **Logs de navegação** | Tempo de sessão, páginas visitadas antes de solicitar crédito, padrão de busca |
| **Histórico de suporte** | Número de chamados, tipo de reclamação, chargeback, NPS |
| **Dados de logística** | Taxa de cancelamento, prazo médio de entrega, devoluções, nota do vendedor |
| **Recorrência na plataforma** | Tempo como cliente ativo, frequência de uso, sazonalidade |

### Open Banking

Com autorização explícita do cliente (Resolução BCB 4.649):

| Dado acessível | Feature gerada |
|----------------|----------------|
| Extrato bancário 12 meses | Saldo médio, volatilidade, sazonalidade de receita |
| Recebíveis futuros | Capacidade de pagamento prospectiva |
| Compromissos ativos | Exposição total a crédito |

> Open Banking é a fonte de **maior impacto esperado** — fluxo de caixa real é o
> melhor preditor de inadimplência em PMEs, mas requer parceria com fintech ou banco.

---

## Expansão do Tier Comportamental

A estratégia mais simples para melhorar o sistema sem dados externos:
aumentar o número de clientes com histórico de pedidos.

**Objetivo**: crescer de 22% → 40%+ de clientes no tier BEHAVIORAL.

**Tática**:
1. Oferecer crédito inicial menor (R$ 2.000–5.000) sem exigir histórico
2. Após 2–3 pedidos pagos em dia, o cliente migra automaticamente para avaliação BEHAVIORAL
3. O modelo comportamental reavalia e pode ampliar o limite

---

## Reject Inference

Clientes **negados** nunca entram no treino atual — o modelo aprende só com aprovados.
Isso causa **viés de seleção**: o modelo subestima risco em perfis que raramente aprova.

### Técnicas

| Técnica | Como funciona | Quando usar |
|---------|--------------|-------------|
| **Parceling** | Divide rejeitados aleatoriamente em bons/maus com proporção estimada | Simples, ponto de partida |
| **Fuzzy Augmentation** | Adiciona rejeitados com peso proporcional à probabilidade de serem maus pagadores | Mais robusto |
| **Extrapolation** | Treina o modelo nos aprovados e extrapola predições para rejeitados, usando-as como proxy de label | Requer validação cuidadosa |

### Implementação sugerida

```python
# Separar aprovados (têm label real) e rejeitados (label desconhecido)
df_approved = df[df['foi_aprovado'] == 1]
df_rejected = df[df['foi_aprovado'] == 0]

# Estimar probabilidade de default nos rejeitados com o modelo atual
p_rejected = model.predict_proba(df_rejected)[:, 1]

# Atribuir peso: rejeitados com p alto contribuem mais como "maus"
df_rejected['sample_weight'] = p_rejected
df_rejected['inadimplente']  = (p_rejected > 0.5).astype(int)  # label sintético

# Combinar e retreinar com sample_weight
df_combined = pd.concat([df_approved, df_rejected])
model.fit(X_combined, y_combined, sample_weight=df_combined['sample_weight'])
```

---

## Cronograma Sugerido

| Semana | Ação | Esforço | Impacto |
|--------|------|---------|---------|
| 1 | Integrar API Receita Federal — enriquecer CNPJ de todos os 3.000 clientes | Baixo | Médio |
| 2 | Cruzar CNAE com taxa de inadimplência setorial do Banco Central | Baixo | Médio |
| 3–4 | Extrair dados internos da plataforma (logs, suporte, logística) | Médio | Alto |
| 5–6 | Implementar reject inference com parceling | Médio | Médio-alto |
| 7+ | Negociar acesso ao Serasa completo ou Quod | Alto | Alto |
| 10+ | Avaliar Open Banking como produto (parceria fintech) | Muito alto | Muito alto |

---

## Critérios de Sucesso

- Cobertura BEHAVIORAL: crescer de 22% para ≥ 35%
- AUC do sistema: crescer de 0,850 para ≥ 0,880
- Redução do tier MANUAL_REVIEW: de 44,6% para ≤ 30% (mais clientes com dados suficientes para decisão automática)
