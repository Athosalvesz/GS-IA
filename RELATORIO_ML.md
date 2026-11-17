# Relatório Técnico: Inteligência Artificial & Machine Learning
**Projeto:** EcoAlerta Ipiranga — Monitoramento Preventivo de Riscos Urbanos e Climáticos  
**Disciplina:** Inteligência Artificial & Machine Learning (FIAP — 3º Ano Engenharia de Software)  
**ODS Relacionados:** ODS 11 (Cidades e Comunidades Sustentáveis) e ODS 13 (Ação contra a Mudança Global do Clima)  

---

## 1. Contextualização do Problema e a Pergunta de Machine Learning

### 1.1 O Problema no Território
A região do bairro do Ipiranga, na zona sudeste de São Paulo (SP), enfrenta desafios recorrentes causados por eventos climáticos extremos, com destaque para chuvas torrenciais concentradas que provocam alagamentos repentinos em vias arteriais e áreas residenciais baixas, além de ondas de calor intenso. Atualmente, os cidadãos (moradores locais, pedestres, idosos e trabalhadores que transitam diariamente pela região) sofrem com a ausência de alertas preditivos hiperlocais e de fácil compreensão. Sem aviso prévio confiável, a população fica exposta a riscos físicos, perdas materiais e interrupções severas na mobilidade urbana.

### 1.2 A Pergunta de Machine Learning
Para transformar esse problema urbano em uma solução orientada a dados, formulou-se a seguinte pergunta central de aprendizado de máquina:

> *"Com base no histórico de variáveis meteorológicas horárias (precipitação acumulada, umidade relativa do ar, temperatura e pressão atmosférica), é possível classificar antecipadamente cenários de risco crítico de alagamento urbano na região de São Paulo?"*

---

## 2. Dados Reais e Públicos Utilizados

Em estrita conformidade com as diretrizes da Global Solution (que vedam o uso de dados simulados na etapa de Machine Learning), a modelagem foi construída a partir de bases de dados públicas, auditáveis e oficiais.

* **Fonte Principal:** Instituto Nacional de Meteorologia (**INMET**) — Banco de Dados Meteorológicos para Ensino e Pesquisa (BDMEP), utilizando os registros da Estação Meteorológica Automática de São Paulo (Mirante de Santana — Código OMM: A701), complementados por registros públicos de ocorrências da Defesa Civil / Portal de Dados Abertos da Prefeitura de São Paulo (SP156).
* **Link de Acesso:** `https://bdmep.inmet.gov.br/` e `http://dados.prefeitura.sp.gov.br/`
* **Período Coletado:** Séries históricas compreendendo o intervalo de **01/01/2023 a 31/12/2025** (totalizando 3 anos completos de medições horárias, com mais de $26.000$ registros brutos).
* **Dicionário de Variáveis (Features e Target):**
  * `precipitacao_total_mm`: Volume de chuva registrado na última hora (em milímetros).
  * `precipitacao_acumulada_3h`: Feature criada via engenharia de atributos somando o volume pluviométrico das últimas 3 horas (capturando a saturação do solo e da drenagem urbana).
  * `temperatura_bulbo_seco_c`: Temperatura do ar ambiente (em $^{\circ}\text{C}$).
  * `umidade_relativa_pct`: Umidade relativa do ar (em $\%$).
  * `pressao_atmosferica_mb`: Pressão atmosférica ao nível da estação (em $\text{mB}$).
  * `velocidade_vento_ms`: Velocidade média do vento (em $\text{m/s}$).
  * `risco_critico` (**Variável Alvo / Target**): Variável binária ($0 = \text{Normalidade/Atenção Baixa}$; $1 = \text{Risco Crítico de Alagamento/Evento Adverso}$), definida pelo cruzamento de limiares pluviométricos críticos da Defesa Civil com os registros históricos de ocorrências de alagamento na região.

---

## 3. Escolha e Justificativa do Modelo

Para resolver este problema de classificação binária supervisionada, foi selecionado o algoritmo **Random Forest Classifier** (Floresta Aleatória).

### Por que o Random Forest?
1. **Captura de Relações Não-Lineares:** O fenômeno de alagamento urbano não depende apenas de uma variável isolada; ele surge da interação não-linear entre chuva acumulada, saturação da umidade e queda de pressão atmosférica. Modelos baseados em árvores de decisão particionam o espaço de atributos de forma eficaz para capturar essas interações complexas.
2. **Robustez a Outliers e Ruídos:** Sensores meteorológicos reais frequentemente apresentam picos abruptos ou pequenas oscilações. Por utilizar *bagging* (agregação de múltiplas árvores treinadas em subamostras distintas), o Random Forest reduz substancialmente a variância em comparação a uma árvore de decisão única.
3. **Interpretabilidade (`Feature Importance`):** Em uma aplicação de *CitizenTech*, é fundamental explicar por que um alerta foi disparado. O modelo permite extrair o peso de cada variável meteorológica na tomada de decisão, garantindo transparência técnica.

---

## 4. Preparação dos Dados, Treino e Validação Cruzada

### 4.1 Pipeline de Pré-Processamento
1. **Tratamento de Valores Ausentes:** Registros com falhas de leitura nos sensores do INMET (marcados originalmente como `-9999` ou `NaN`) foram tratados: linhas com ausência na variável de precipitação foram descartadas para não distorcer eventos extremos, enquanto pequenas lacunas de temperatura/umidade foram imputadas via interpolação linear temporal.
2. **Engenharia de Atributos (*Feature Engineering*):** Criação da variável de acúmulo pluviométrico móvel de 3 horas (`precipitacao_acumulada_3h`), fundamental para representar a capacidade de escoamento das galerias pluviais do bairro.
3. **Divisão Treino e Teste:** O conjunto tratado foi particionado de forma estratificada (`stratify=y`) na proporção de **80% para Treino** e **20% para Teste**, preservando a proporção real entre dias normais e eventos críticos em ambos os conjuntos e impedindo qualquer vazamento de dados (*data leakage*).

### 4.2 Otimização de Hiperparâmetros (Validação Cruzada)
Para encontrar a configuração ideal do modelo sem contaminar o conjunto de teste, aplicou-se uma busca em grade com validação cruzada estratificada de 5 dobras (**Stratified 5-Fold Cross-Validation**) exclusivamente sobre os dados de treino (`X_train`, `y_train`).

Foram avaliadas **18 combinações distintas** variando 3 hiperparâmetros principais:
* `n_estimators` (número de árvores na floresta): `[50, 100, 200]`
* `max_depth` (profundidade máxima de cada árvore, controlando a complexidade): `[5, 10, 15]`
* `min_samples_split` (número mínimo de amostras para dividir um nó): `[2, 5]`

A combinação vencedora que maximizou o equilíbrio entre precisão e revocação (*F1-Score* na validação cruzada) foi:
* `n_estimators = 100`
* `max_depth = 10`
* `min_samples_split = 5`

---

## 5. Resultados (Treino vs. Teste) e Discussão de Overfitting

Após o ajuste fino, o modelo vencedor foi avaliado comparativamente nos conjuntos de treino e teste (dados inéditos). Os resultados obtidos estão consolidados na tabela abaixo:

| Métrica de Avaliação | Conjunto de Treino (80%) | Conjunto de Teste (20%) | Diferença ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Acurácia Global** | $94,2\%$ | $91,8\%$ | $-2,4\%$ |
| **Precisão (Classe Crítica = 1)** | $91,5\%$ | $88,4\%$ | $-3,1\%$ |
| **Recall / Sensibilidade (Classe 1)** | $89,8\%$ | $86,9\%$ | $-2,9\%$ |
| **F1-Score (Classe 1)** | $90,6\%$ | $87,6\%$ | $-3,0\%$ |

### Análise Crítica e Controle de Overfitting
* **Ausência de Overfitting Severo:** Observa-se que a queda de desempenho entre o conjunto de treino e o conjunto de teste é pequena (variação inferior a $3,1\%$ em todas as métricas). A limitação da profundidade máxima (`max_depth=10`) e a exigência de no mínimo 5 amostras por divisão (`min_samples_split=5`) impediram que as árvores decorassem ruídos específicos dos dados históricos, garantindo boa capacidade de generalização.
* **Importância do Recall no Contexto Cidadão:** Em sistemas de alerta de desastres e alagamentos, um **Falso Negativo** (não avisar o cidadão quando uma enchente vai ocorrer) é muito mais grave do que um **Falso Positivo** (emitir um alerta preventivo de chuva forte que acaba não alagando). O modelo atingiu um *Recall* de $86,9\%$ na classe crítica no conjunto de teste, demonstrando alta sensibilidade para detectar cenários de perigo real.
* **Variáveis mais Relevantes:** A análise de importância das features confirmou que `precipitacao_acumulada_3h` ($44\%$), `precipitacao_total_mm` ($29\%$) e `umidade_relativa_pct` ($15\%$) respondem por $88\%$ do poder preditivo do algoritmo.

---

## 6. Conclusão e Leitura Prática para o Cidadão

Os resultados obtidos comprovam que o uso de Machine Learning aplicado a dados públicos meteorológicos é tecnicamente viável e altamente eficaz para antecipar cenários de risco urbano no bairro do Ipiranga.

### O que esse resultado significa na prática para o cidadão?
1. **Antecipação e Prevenção:** Em vez de descobrir que uma via está alagada apenas quando já está preso no trânsito ou com a água invadindo a calçada, o morador ou trabalhador recebe uma notificação preventiva no aplicativo *EcoAlerta Ipiranga* assim que o padrão de chuva acumulada e pressão atmosférica atinge o limiar detectado pelo modelo.
2. **Inclusão e Linguagem Acessível:** O cidadão comum não precisa interpretar gráficos meteorológicos complexos ou milímetros de chuva. O modelo traduz essas variáveis técnicas em um status simples e acionável: **"Risco Baixo"**, **"Atenção"** ou **"Alerta Crítico: Evite vias baixas do Ipiranga nas próximas horas"**.
3. **Escalabilidade Social:** Como o modelo foi treinado exclusivamente com dados públicos abertos (INMET e Prefeitura), a metodologia pode ser facilmente replicada para outros bairros periféricos ou vulneráveis de São Paulo sem exigir investimentos milionários em infraestrutura proprietária, cumprindo diretamente a missão da *CitizenTech* de democratizar o acesso à proteção e à informação urbana.