# Machine Learning e Visão Computacional

## Projeto Avaliativo – Módulo 1

### Predição de Inadimplência utilizando Machine Learning

---

# Sobre o projeto

Este projeto foi desenvolvido como parte da disciplina **Machine Learning e Visão Computacional**.

O objetivo é construir um pipeline completo de Ciência de Dados capaz de prever se um cliente irá se tornar inadimplente em um empréstimo bancário, utilizando técnicas de preparação de dados, engenharia de atributos, aprendizado de máquina e avaliação de modelos.

Durante o desenvolvimento foram aplicadas todas as etapas de um projeto real de Data Science, desde a análise exploratória até a recomendação final do modelo mais adequado para utilização em produção.

---

# Problema de negócio

Instituições financeiras concedem milhares de empréstimos diariamente.

Uma decisão incorreta pode gerar prejuízos financeiros significativos.

O objetivo deste projeto é desenvolver um modelo capaz de prever se um cliente será inadimplente (`loan_status = 1`) ou permanecerá adimplente (`loan_status = 0`).

Os modelos avaliados foram:

* K-Nearest Neighbors (KNN)
* Árvore de Decisão (Decision Tree)

Ao final, ambos foram comparados considerando não apenas métricas estatísticas, mas também o impacto financeiro dos erros de classificação.

---

# Tecnologias utilizadas

* Python 3
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)

---

# Estrutura do projeto

```text
├── data.csv
├── df_tratado.csv
├── projeto.ipynb
├── README.md
└── requirements.txt (opcional)
```

---

# Dicionário de Dados

| Coluna                     | Descrição                                                                   |
| -------------------------- | --------------------------------------------------------------------------- |
| person_age                 | Idade do cliente                                                            |
| person_income              | Renda anual                                                                 |
| person_home_ownership      | Situação da moradia                                                         |
| person_emp_length          | Tempo de emprego (anos)                                                     |
| loan_intent                | Finalidade do empréstimo                                                    |
| loan_grade                 | Classificação de risco atribuída pelo banco                                 |
| loan_amnt                  | Valor do empréstimo                                                         |
| loan_int_rate              | Taxa de juros (%)                                                           |
| loan_percent_income        | Percentual da renda comprometido pelo empréstimo                            |
| cb_person_default_on_file  | Histórico anterior de inadimplência                                         |
| cb_person_cred_hist_length | Tempo de histórico de crédito                                               |
| loan_status                | Variável alvo (0 = adimplente / 1 = inadimplente)                           |
| financial_pressure         | Variável criada neste projeto representando a pressão financeira do cliente |

---

# Pipeline desenvolvido

## 1. Análise Exploratória (EDA)

Foram realizadas análises estatísticas para compreender a base de dados:

* quantidade de registros;
* tipos das variáveis;
* valores ausentes;
* registros duplicados;
* estatísticas descritivas;
* histogramas;
* boxplots;
* gráfico de distribuição da variável alvo;
* mapa de correlação.

---

## 2. Limpeza dos dados

Nesta etapa foram realizadas:

* remoção de registros duplicados;
* identificação dos valores ausentes;
* substituição dos valores nulos pela mediana.

A mediana foi escolhida porque as distribuições apresentavam assimetria e presença de outliers, tornando-a mais robusta que a média.

---

## 3. Engenharia de Atributos (Feature Engineering)

O enunciado propunha criar uma variável baseada apenas na razão entre valor do empréstimo e renda.

Entretanto, essa informação já existia originalmente na base através da variável **loan_percent_income**, tornando a nova variável praticamente redundante.

Por esse motivo foi criada a variável:

```text
financial_pressure
```

calculada como:

```text
loan_int_rate × (loan_amnt / person_income)
```

Essa variável busca representar simultaneamente:

* valor do empréstimo;
* comprometimento da renda;
* custo financeiro do crédito.

Dessa forma, agrega uma informação diferente da existente originalmente.

---

## 4. Preparação dos dados

Foram realizadas as seguintes etapas:

* Encoding das variáveis categóricas;
* Divisão entre variáveis preditoras (X) e variável alvo (y);
* Separação em treino (80%) e teste (20%);
* utilização de Stratified Split;
* balanceamento apenas da base de treino utilizando SMOTE;
* padronização apenas das variáveis contínuas para utilização no modelo KNN.

A Árvore de Decisão foi treinada sem escalonamento, pois esse algoritmo não depende da escala das variáveis.

---

## 5. Modelagem

Foram treinados dois algoritmos:

### KNN

Valores testados:

* K = 3
* K = 5
* K = 7
* K = 9
* K = 11

---

### Árvore de Decisão

Profundidades avaliadas:

* 3
* 5
* 7
* 9
* 11
* None

---

## 6. Diagnóstico de Overfitting

O desempenho foi acompanhado simultaneamente nas bases de treino e teste.

A comparação permitiu identificar:

* aumento excessivo da acurácia de treino;
* redução da capacidade de generalização;
* escolha dos hiperparâmetros com melhor equilíbrio.

---

## 7. Avaliação

Os modelos foram avaliados utilizando:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Matriz de Confusão

---

# Resumo Executivo

A Árvore de Decisão apresentou maior acurácia geral e melhor precisão quando comparada ao KNN.

Entretanto, considerando o contexto de negócio, o modelo KNN apresentou maior capacidade de identificar clientes inadimplentes (maior Recall para a classe positiva).

No cenário proposto, um Falso Negativo representa conceder crédito a um cliente que poderá não quitar sua dívida, gerando perdas financeiras diretas.

Já um Falso Positivo não implica necessariamente na recusa automática do empréstimo, podendo apenas resultar em uma análise complementar, exigência de garantias adicionais ou aplicação de taxas de juros mais elevadas.

Por esse motivo, apesar da menor acurácia global, o modelo KNN foi considerado a melhor alternativa para auxiliar o processo de concessão de crédito.

---

# Como executar

## 1. Clone o repositório

```bash
git clone https://github.com/antonioaugustoalt/ML_Risco_Inadimplencia.git
```

---

## 2. Acesse a pasta

```bash
cd NOME_DO_REPOSITORIO
```

---

## 3. Instale as dependências

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install imbalanced-learn
```

ou

```bash
pip install -r requirements.txt
```

---

## 4. Execute o notebook

Abra o arquivo `.ipynb` utilizando:

* Jupyter Notebook
* JupyterLab
* VS Code

e execute as células em sequência.

---

# Autor

Antonio Augusto

Projeto desenvolvido para fins acadêmicos na disciplina de **Machine Learning e Visão Computacional**.
