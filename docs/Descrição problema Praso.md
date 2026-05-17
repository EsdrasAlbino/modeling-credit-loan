Uma das principais dores do pequeno varejista é o ciclo de caixa. O ciclo de caixa é o tempo médio que uma empresa leva para transformar os seus gastos em receita (ou as suas compras em vendas). Quanto menor o ciclo de caixa, melhor, pois o caixa (dinheiro) fica menos tempo preso no estoque ou a espera de ser recebido como pagamento. Um ciclo de caixa negativo implica que a empresa só precisa pagar os seus fornecedores após receber todo o dinheiro de suas vendas. O ciclo de caixa é calculado como (PME \+ PMR) \- PMP, em que:

PME (prazo médio de estocagem) é o tempo médio que os produtos ficam armazenados antes de serem vendidos;  
PMR (prazo médio de recebimento) é o tempo médio que a empresa leva para receber o pagamento dos seus clientes;  
PMP (prazo médio de pagamento) é o tempo médio que a empresa tem para pagar os seus fornecedores.  
No mercado atual, a grande maioria dos varejistas compram à vista em atacados ou na CEASA (ou seja, PMP \= 0\) e alguns têm acesso a prazo de pagamento com fornecedores, ou seja, podem pagar alguns dias após receberem as mercadorias que compraram. Em média, um fornecedor leva 2-14 dias para avaliar o “crédito” de um varejista, ou seja, decidir se vai oferecer prazo de pagamento ou não. Essa avaliação é feita porque uma vez que um comprador recebe as mercadorias, ele pode simplesmente não pagar, seja ele mal-intencionado ou não, e gerar grandes prejuízos financeiros para o fornecedor. Esses fornecedores tradicionais também exigem que os varejistas comprem um volume mínimo de mercadorias, o que aumenta o PME.  
A Praso se propôs a resolver a dor de ciclo de caixa de pequenos empreendedores. Além de não exigir um volume mínimo para a compra, a Praso desburocratizou o crédito de varejistas com uma análise instantâneas no momento em do cadastro no aplicativo. Para fazer isso, a Praso desenvolveu um modelo de crédito proprietário, que a partir do CNPJ de um estabelecimento puxa informações de diversas APIs e calcula um score de risco para um determinado cliente.

Score de crédito  
O score de risco de um varejista é uma estimativa da probabilidade de esse cliente ficar “inadimplente”, ou seja, não pagar pelas mercadorias que recebeu (ex. score de 0.1 corresponde a 10% de chance de inadimplência). A escolha de a partir de qual score de risco conceder ou não crédito é uma escolha estratégica do negócio que leva em conta metas de lucro e de aquisição de novos clientes. Portanto, o importante para o modelo é ordenar bem os clientes, isso é, garantir que o score de risco de um cliente que ficou inadimplente é maior que o score de um cliente que não ficou inadimplente. Para medir a qualidade do modelo, usamos o ROC-AUC.

Dados  
Aplicação \- Clientes  
credito\_aplicacao\_clientes\_final.csv  
Descrição das colunas:

id\_cliente: Identificador único do cliente (útil para fazer um JOIN na tabela de pedidos).  
uf: Unidade da Federação (ex. PE).  
municipio: Município (ex. RECIFE).  
segmento\_cliente: Segmento do estabelecimento (ex. Padaria). Os dados estão anonimizados entre Segmento 1-21.  
natureza\_juridica: A natureza jurídica é um regime jurídico que define como uma empresa é reconhecida pela lei, e como ela será organizada, gerida e tratada pelo sistema jurídico (ex. 213-5 \- Empresário (Individual)).  
fonte\_cliente: Por qual fonte o cliente foi adquirido (ex. indicação de um amigo ou time de vendas). Os dados estão anonimizados entre Fonte 1-5.  
cnae\_codigo: Código da atividade econômica principal, identificado por um número (ex. 56.11-2-01). O código foi anonimizado porém a hierarquia e estrutura de informações foi mantida em todos os níveis do CNAE.  
capital\_social: O valor investido pelos sócios de uma empresa para que ela possa funcionar e se manter.  
idade\_cnpj: Dias desde a abertura do CNPJ na Secretaria da Fazenda até o cadastro na Praso.  
serasa\_contagem\_negativacoes: Número total de negativações. Negativação é um registro formal da inadimplência de uma pessoa física ou jurídica em sistemas de proteção ao crédito, como a Serasa.  
serasa\_contagem\_protestos: Número total de protestos. Protesto é um registro formal da inadimplência de uma pessoa física ou jurídica em um cartório.  
serasa\_credores: Até 5 empresas diferentes para as quais o cliente deve. Se o cliente dever para mais de 5 empresas diferentes, as 5 mais recentes serão exibidas. Os dados estão anonimizados, identificando apenas o segmento de cada um dos credores (ex. Comércio). No caso do cliente não dever para outra empresa, o campo é nulo.  
serasa\_socio\_tem\_negativacao: Booleana se algum sócio do estabelecimento tem (1) ou não tem (0) negativação na Serasa.  
ifood\_contagem\_avaliacoes: Número total de avaliações no iFood. Em caso do estabelecimento não estar no iFood, o campo é nulo.  
ifood\_faixa\_preco: Faixa de preço no iFood (ex. $$$). Em caso do estabelecimento não estar no iFood, o campo é nulo.  
google\_maps\_avaliacao: Avaliação/nota no Google Maps (0-5). Em caso do estabelecimento não estar no Google Maps, o campo é nulo.  
google\_maps\_contagem\_avaliacoes: Número total de avaliações no Google Maps. Em caso do estabelecimento não estar no Google Maps, o campo é nulo.  
google\_maps\_tem\_website: Booleana se tem (1) ou não tem (0) site associado no Google Maps. Em caso do estabelecimento não estar no Google Maps, o campo é nulo.  
inadimplente: Se o cliente ficou inadimplente (1) ou não (0).  
Comportamental \- Pedidos  
credito\_comportamental\_pedidos\_final.csv  
Descrição das colunas:

id\_pedido: Identificador único do pedido.  
id\_cliente: Identificador único do cliente (útil para fazer um JOIN na tabela de clientes).  
valor: Valor das mercadorias pedidas pelo cliente, excluindo cupons de desconto.  
atraso: Número de dias que o cliente atrasou para pagar um pedido em comparação ao prazo de pagamento original. Se o valor é 0, isso significa que o cliente não atrasou.  
data\_entrega: Data em que o pedido foi entregue para o cliente pela logística da Praso.  
Desafio  
Exploração  
Antes de começar a desenvolver os modelo preditivos, entenda os dados mais a fundo\! Reflita sobre o que cada dado significa e o que você espera encontrar ao analisá-los. Por exemplo, você espera que a inadimplência de clientes negativados na Serasa seja maior ou menor do que a média? É recomendado gerar alguns gráficos para analisar as variáveis listadas VS a inadimplência, por exemplo:

Gráfico 1\. Taxa de inadimplência por intervalo de idade do CNPJ. É possível observar uma tendência clara de redução de inadimplência em estabelecimentos mais antigos, especialmente com mais 2.400 dias (6.5 anos).

Modelagem  
Modelo de aplicação / Avaliação de clientes novos: Entender a probabilidade de um novo cliente (que acabou de se cadastrar na Praso) ficar inadimplente. As variáveis que esse modelo pode levar em conta são apenas as variáveis às quais temos acesso antes de um cliente se cadastrar (ex. dados públicos da Receita Federal, dados de crédito da Serasa e dados de plataformas de venda, como o iFood). A utilidade desse modelo de aplicação é decidir para quais clientes conceder crédito, de forma a minimizar o prejuízo financeiro consequente da inadimplência e maximizar o número de clientes comprando.  
Modelo comportamental / Avaliação de clientes recorrentes: Entender a probabilidade de um cliente ativo (que fez compra nos últimos 30 dias) ficar inadimplente. As variáveis que esse modelo pode levar em conta são tanto as variáveis às quais temos acesso antes de um cliente se cadastrar como o histórico de compras desse cliente (ex. quantos pedidos ele já fez, em quantos ele atrasou, qual foi o atraso médio etc.). Para construir o modelo, é necessário gerar variáveis agregadas a partir do histórico de compras dos clientes e uní-las aos dados de clientes, nos quais é possível identificar quem ficou inadimplente e quem não.  
Aplicação  
Agora que você tem um modelo para avaliar o risco dos clientes, como você faria para construir uma política de crédito na Praso? O que você levaria em conta, além do risco, na hora de decidir se aprova ou não um cliente? E como você incluiria o componente comportamental na sua política?  
O objetivo aqui não é construir um modelo ou uma política na prática e com números, mas sim discutir por escrito como você aplicaria o que você construiu no cenário prático da Praso\!

Sugestões  
Em variáveis contínuas, extraia métricas como mínima, máxima, mediana, variância, moda etc. Histogramas podem ser úteis para visualização dos dados\! Já em variáveis categóricas, tente agrupar os dados por valores. Em ambos os casos, compare-os sempre com a inadimplência. Por exemplo, analise se a inadimplência de padarias é maior ou menor que a inadimplência de hotéis.  
Lembre-se de que algumas variáveis contínuas (ex. idade do CNPJ) foram transformadas em intervalos (ex. (150-250\]) para a proteção e a segurança dos dados dos clientes da Praso. Na hora de analisá-las e utilizá-las no modelo, pode ser interessante transformá-las em variáveis contínuas, usando o ponto médio do intervalo como valor.  
Explore maneiras de gerar novas variáveis a partir das atuais. Algumas sugestões:A partir de colunas que têm dados nulos é possível gerar variáveis booleanas indicando se o dado existe ou não. No caso de ifood\_contagem\_avaliacoes, a variável indicaria se o cliente está ou não no iFood.  
Considere agrupar variáveis categóricas em novas variáveis, por exemplo, quebrar municípios em capitais, regiões metropolitanas e interior ou por número de habitantes.  
A coluna cnae\_codigo (Classificação Nacional das Atividades Econômicas) é uma variável interessante para o modelo, mas que precisa de um certo tratamento para ser bem aproveitada. É possível quebrar um código CNAE em divisões, grupos, classes e subclasses. Por exemplo, o CNAE 56.11-2-04 pode ser quebrado na divisão 56, no grupo 56.11, na classe 56.11-2 e na subclasse 56.11-2-04.  
A coluna credores (empresas para quem o cliente deve) também é interessante para o modelo, porém apresenta um desafio por ser uma lista de credores. Por exemplo, um cliente pode dever para “Alimentos e Bebidas, Alimentos e Bebidas, Alimentos e Bebidas” (3 empresas diferentes de Alimentos e Bebidas), enquanto outro pode dever para “Alimentos e Bebidas, Distribuição” (2 empresas, uma de Alimentos e Bebidas e outra de Distribuição).  
Para o modelo comportamental, agregue dados para formar novas variáveis a nível cliente. Algumas sugestões:Considerando a variável de valor do pedido, calcule o valor mínimo, médio, mediano e máximo de cada cliente. Avalie cada uma dessas variáveis VS a inadimplência.  
Considerando a variável de atraso dos pedidos, além de extrair métricas básicas, calcule a proporção de pedidos em que o cliente atrasa para pagar (0-100%).  
Tome cuidado com overfitting dos dados, especialmente ao trabalhar com variáveis categóricas que têm baixa representatividade (ex. municípios com poucos clientes).  
Teste modelos diferentes. Nem sempre um xgboost vai ser melhor do que uma regressão logística tradicional\!  
Divirta-se\! Você está trabalhando com dados reais de uma startup brasileira que começou aqui em Pernambuco. Se você curtir esse tipo de desafio, você pode curtir trabalhar nela também\!  
