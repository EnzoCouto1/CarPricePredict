# 🚗 Estimativa de Preços de Veículos Usados

> **Disciplina:** C318 - Fundamentos de Machine Learning  
> **Instituto Nacional de Telecomunicações - Inatel** > **Modelo:** Regressão Linear Múltipla

---

## 📋 Sobre o Projeto
Este projeto visa desenvolver um modelo de Machine Learning capaz de prever o preço de venda de carros usados com base em características como ano de fabricação, quilometragem, tipo de combustível e vendedor.

O objetivo é auxiliar revendedores e proprietários a precificarem seus veículos de forma justa e competitiva, utilizando dados históricos para reduzir a subjetividade da avaliação humana.

## ❓ Perguntas de Negócio
O projeto busca responder às seguintes questões:
1. Quais características (ex: Ano, Combustível, Transmissão) mais influenciam o preço final do veículo?
2. É possível prever o preço de revenda com uma margem de erro aceitável utilizando um modelo linear simples?
3. Qual o impacto da idade do carro na sua desvalorização imediata?

## 🛠 Tecnologias Utilizadas
* **Linguagem:** Python 3
* **Bibliotecas:**
    * `Pandas`: Manipulação e análise de dados.
    * `Seaborn` / `Matplotlib`: Visualização de dados (Heatmaps, Scatter plots).
    * `Scikit-Learn`: Criação do modelo de Regressão, pré-processamento e métricas de avaliação.

## 📂 Dataset
Foi utilizado o **"Vehicle dataset from Cardekho"**, disponível publicamente no Kaggle.
* **Fonte:** [Kaggle Link](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho)
* **Principais Colunas:**
    * `Selling_Price`: Preço de venda (Target).
    * `Present_Price`: Preço de tabela atual.
    * `Kms_Driven`: Quilometragem rodada.
    * `Fuel_Type`: Tipo de combustível (Petrol, Diesel, CNG).
    * `Seller_Type`: Tipo de vendedor (Individual ou Dealer).
    * `Transmission`: Câmbio (Manual ou Automatic).

## 🚀 Etapas do Desenvolvimento

### 1. Análise Exploratória de Dados (EDA)
Realizamos a visualização dos dados para entender correlações:
* **Mapa de Calor:** Identificou forte correlação positiva entre o *Preço de Tabela* e o *Preço de Venda*.
* **Scatter Plot:** Confirmou a tendência linear de desvalorização conforme o aumento da idade do veículo.
* **Boxplot:** Mostrou que revendedoras (*Dealers*) tendem a praticar preços mais elevados que vendedores individuais.

### 2. Pré-processamento
Para preparar os dados para o modelo:
* **Feature Engineering:** Criação da variável `Car_Age` (Idade do Carro) subtraindo o ano de fabricação do ano atual.
* **Limpeza:** Remoção da coluna `Car_Name` (alta cardinalidade) e `Year` (redundante após criação da idade).
* **Encoding:** Aplicação de *One-Hot Encoding* para transformar variáveis categóricas (`Fuel_Type`, `Transmission`) em numéricas.

### 3. Modelagem
Utilizamos o algoritmo de **Regressão Linear Múltipla**.
* **Divisão dos dados:** 80% para Treino e 20% para Teste.
* **Motivação:** O problema apresenta características lineares fortes e buscamos um modelo interpretável (Navalha de Occam).

## 📊 Resultados e Métricas

O modelo obteve uma performance satisfatória para o escopo do projeto:

| Métrica | Resultado | Descrição |
| :--- | :--- | :--- |
| **R² Score** | **0.85** | O modelo consegue explicar 85% da variação dos preços dos carros. |
| **MAE** | **1.22** | Erro Médio Absoluto das previsões. |

### Principais Insights (Coeficientes)
A análise dos pesos do modelo revelou que:
1.  **Preço de Tabela (`Present_Price`):** É o maior impulsionador do valor de revenda.
2.  **Idade (`Car_Age`):** É o principal fator de desvalorização (coeficiente negativo).
3.  **Venda por Loja:** Veículos vendidos por concessionárias têm uma valorização automática em comparação a vendas particulares.

## 📦 Como executar
1.  Clone este repositório.
2.  Instale as dependências:
    ```bash
    pip install pandas seaborn matplotlib scikit-learn
    ```
3.  Certifique-se de que o arquivo `car data.csv` está na raiz.
4.  Execute o script principal:
    ```bash
    python modelo_regressao.py
    ```