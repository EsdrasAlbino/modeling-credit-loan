# **SISTEMA HÍBRIDO DE CREDIT SCORING PARA PEQUENOS VAREJISTAS UTILIZANDO MACHINE LEARNING: UM ESTUDO DE CASO NA PRASO**

## **Abstract**

A concessão de crédito para pequenos varejistas representa um desafio devido à escassez de informações financeiras estruturadas e ao risco de inadimplência. A Praso, startup brasileira especializada em distribuição para pequenos estabelecimentos comerciais, utiliza um sistema de análise de crédito baseado em dados cadastrais, informações de bureaus de crédito e dados comportamentais dos clientes. Este trabalho propõe a construção de modelos de Machine Learning para previsão de inadimplência utilizando dados de aproximadamente 3.000 clientes. Foram realizadas etapas de análise exploratória, pré-processamento, engenharia de atributos e treinamento de modelos supervisionados. Os resultados demonstram que a utilização combinada de dados cadastrais e comportamentais permite melhorar significativamente a capacidade de identificação de clientes de risco, alcançando ROC-AUC de 0,850 no sistema integrado.

# **I. INTRODUÇÃO**

O acesso ao crédito é um dos principais fatores que influenciam a sustentabilidade financeira de pequenos varejistas. Entretanto, a concessão de crédito envolve riscos associados à possibilidade de inadimplência dos clientes, exigindo mecanismos eficientes de avaliação de risco.

A Praso atua no segmento de distribuição para pequenos estabelecimentos comerciais e oferece crédito para seus clientes realizarem compras por meio da plataforma. Para isso, utiliza informações públicas, dados de crédito e histórico de relacionamento para estimar a probabilidade de inadimplência de cada cliente.

A utilização de técnicas de Machine Learning permite automatizar esse processo, reduzindo o tempo de análise, aumentando a escalabilidade e melhorando a qualidade das decisões de crédito.

O objetivo deste trabalho é desenvolver e avaliar modelos de classificação capazes de prever a probabilidade de inadimplência de clientes da plataforma Praso, utilizando tanto informações cadastrais quanto dados comportamentais oriundos do histórico de pedidos.

# **II. ANÁLISE DE DADOS E FEATURE ENGINEERING**

## **A. Análise Exploratória dos Dados**

O conjunto de dados disponibilizado pela Praso é composto por duas bases principais: uma base de clientes contendo informações cadastrais e uma base de pedidos contendo informações transacionais.

A base de clientes possui aproximadamente 3.000 registros e contém atributos relacionados à localização, segmento de atuação, características jurídicas, indicadores de crédito da Serasa e presença digital em plataformas como iFood e Google Maps. A variável alvo do problema é a inadimplência, representada por uma variável binária.

A análise inicial mostrou que 31,3% dos clientes apresentaram histórico de inadimplência, enquanto 68,7% permaneceram adimplentes. Essa distribuição caracteriza um desbalanceamento moderado entre as classes.

Observou-se também que apenas 664 clientes (22,1%) possuíam histórico comportamental de pedidos, enquanto 2.336 clientes (77,9%) não possuíam informações transacionais suficientes para utilização de um modelo comportamental.

Entre as variáveis disponíveis, foram identificadas como categóricas: UF, município, segmento do cliente, natureza jurídica, fonte de aquisição, código CNAE, faixa de preço do iFood e segmentos dos credores. As variáveis numéricas incluem capital social, idade do CNPJ, quantidade de negativações, quantidade de protestos, avaliações do Google Maps e métricas derivadas dos pedidos.

A análise univariada indicou que empresas mais antigas apresentavam menor taxa de inadimplência. Da mesma forma, clientes com sócios negativados ou maior quantidade de protestos apresentavam risco significativamente superior à média da carteira.

## **B. Pré-processamento dos Dados**

Inicialmente foi realizada a verificação de valores faltantes. Observou-se que atributos relacionados ao iFood e Google Maps apresentavam grande quantidade de valores nulos. Entretanto, esses valores possuem significado de negócio, indicando que o estabelecimento não está presente na plataforma correspondente.

Para preservar essa informação, foram criadas variáveis binárias indicando a existência ou ausência do dado original.

As variáveis categóricas foram transformadas utilizando técnicas de encoding adequadas. Para atributos de alta cardinalidade, como CNAE, foi utilizada uma abordagem baseada em agrupamento hierárquico e target encoding.

As variáveis numéricas foram padronizadas utilizando StandardScaler quando necessário, especialmente para modelos sensíveis à escala dos atributos.

## **C. Divisão dos Dados**

Os dados foram divididos em conjuntos de treino, validação e teste utilizando amostragem estratificada para preservar a proporção de inadimplentes em todos os subconjuntos.

A divisão adotada foi:

* Treino: 70%  
* Validação: 15%  
* Teste: 15%

Essa estratégia permitiu realizar ajuste de hiperparâmetros sem comprometer a avaliação final dos modelos.

## **D. Feature Engineering**

Uma das principais contribuições deste trabalho foi a criação de atributos derivados capazes de representar melhor o comportamento de risco dos clientes.

Inicialmente, variáveis originalmente representadas por intervalos foram convertidas para seus pontos médios. Por exemplo, uma faixa de idade do CNPJ representada por (1000,1500\] foi transformada no valor contínuo 1250\.

Para o atributo CNAE foram extraídos níveis hierárquicos como divisão, grupo e classe econômica. Além disso, foi aplicado Target Encoding para capturar a taxa histórica de inadimplência associada a cada segmento econômico.

A partir das informações da Serasa foram criadas variáveis derivadas representando quantidade de credores, quantidade de setores distintos entre os credores e interações entre negativações e credores.

Para os clientes com histórico transacional foram calculadas métricas agregadas como quantidade total de pedidos, valor médio dos pedidos, atraso médio, atraso máximo e percentual de pedidos pagos em atraso.

# **III. MODELAGEM**

Foram avaliados três algoritmos supervisionados amplamente utilizados em problemas de credit scoring.

## **A. Regressão Logística**

A Regressão Logística foi utilizada como modelo baseline devido à sua simplicidade e elevada interpretabilidade. Esse algoritmo modela diretamente a probabilidade de ocorrência do evento de interesse através de uma função logística.

## **B. Random Forest**

O Random Forest é um método ensemble baseado na construção de múltiplas árvores de decisão. Sua principal vantagem é a capacidade de capturar relações não lineares entre os atributos e a variável alvo.

Além disso, o algoritmo apresenta boa robustez a ruídos e variáveis irrelevantes.

## **C. XGBoost**

O XGBoost foi utilizado como principal modelo deste trabalho devido ao seu excelente desempenho em problemas tabulares.

O algoritmo utiliza a técnica de Gradient Boosting, construindo sucessivamente árvores capazes de corrigir os erros das árvores anteriores. Seu desempenho superior em competições de Machine Learning justifica sua utilização em aplicações de crédito.

# **IV. ANÁLISE E COMPARAÇÃO DOS RESULTADOS**

A principal métrica utilizada para avaliação dos modelos foi a ROC-AUC, pois essa métrica mede a capacidade do modelo de ordenar corretamente clientes de maior e menor risco independentemente do threshold adotado.

O modelo de aplicação, responsável por avaliar novos clientes sem histórico de compras, alcançou ROC-AUC de 0,772 utilizando informações cadastrais, dados da Serasa e presença digital.

O modelo comportamental, construído a partir do histórico de pedidos e pagamentos dos clientes, alcançou ROC-AUC de 0,770.

Embora os resultados individuais sejam semelhantes, a utilização conjunta dos modelos através de uma estratégia híbrida permitiu alcançar ROC-AUC de 0,850, representando um ganho expressivo de desempenho.

Os resultados demonstram que informações comportamentais agregam valor significativo ao processo de avaliação de risco, reduzindo a incerteza sobre o comportamento futuro dos clientes.

A análise das importâncias das variáveis indicou que idade do CNPJ, indicadores da Serasa e métricas relacionadas ao atraso de pagamentos figuraram entre os atributos mais relevantes para a previsão de inadimplência.

# **V. CONCLUSÃO E DISCUSSÃO**

Este trabalho apresentou o desenvolvimento de um sistema híbrido de credit scoring aplicado ao contexto da plataforma Praso. A solução proposta combinou informações cadastrais e comportamentais para estimar a probabilidade de inadimplência de pequenos varejistas.

Os resultados obtidos demonstram que modelos de Machine Learning são capazes de apoiar de forma eficiente o processo de concessão de crédito, reduzindo riscos e aumentando a capacidade de tomada de decisão automatizada.

Entre os principais achados destaca-se a importância da idade do CNPJ, dos indicadores de crédito da Serasa e do histórico de pagamentos como fatores preditivos relevantes para a inadimplência.

Como limitações do estudo, destaca-se a baixa disponibilidade de dados comportamentais, uma vez que apenas 22,1% dos clientes possuíam histórico de pedidos suficiente para utilização do modelo comportamental.

Como trabalhos futuros sugere-se a integração de novas fontes de dados, incluindo informações da Receita Federal, indicadores setoriais do Banco Central e dados oriundos de Open Banking. Também podem ser exploradas técnicas de reject inference, calibração avançada de probabilidades e modelos de survival analysis para previsão temporal de inadimplência.

