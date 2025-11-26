# 🚗 Estimativa de Preços de Veículos Usados

> **Disciplina:** C318 - Fundamentos de Machine Learning
> **Instituto Nacional de Telecomunicações - Inatel**
> **Modelo Final:** Random Forest Regressor (R² = 0.96)

---

## 📋 Sobre o Projeto
Este projeto visa desenvolver um modelo de Machine Learning capaz de prever o preço de venda de carros usados com alta precisão.

O objetivo é auxiliar revendedores e proprietários a precificarem seus veículos de forma justa e competitiva, utilizando dados históricos para reduzir a subjetividade da avaliação humana.

## ❓ Perguntas de Negócio
1. Quais características (ex: Ano, Combustível, Transmissão) mais influenciam o preço final do veículo?
2. É possível prever o preço de revenda com uma margem de erro aceitável?
3. Qual algoritmo performa melhor para este cenário: modelos lineares ou baseados em árvore?

## 🛠 Tecnologias Utilizadas
* **Linguagem:** Python 3
* **Bibliotecas:** `Pandas`, `Seaborn`, `Scikit-Learn`.

## 📂 Dataset
Foi utilizado o **"Vehicle dataset from Cardekho"**, disponível no Kaggle.
* **Variáveis Chave:** `Selling_Price` (Target), `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`.

## 🚀 Metodologia

### 1. Pré-processamento
* **Feature Engineering:** Criação da variável `Car_Age` (2024 - Ano Fabricação).
* **Limpeza:** Remoção de colunas redundantes.
* **Encoding:** Transformação de variáveis categóricas (One-Hot Encoding).

### 2. Modelagem (Estratégia Challenger)
Adotamos uma abordagem comparativa para garantir a melhor performance:
1.  **Baseline (Linha de Base):** Regressão Linear Múltipla.
2.  **Challenger (Desafiante):** Random Forest Regressor (Ensemble Method).

## 📊 Resultados e Comparação

O modelo **Random Forest** superou significativamente a Regressão Linear, demonstrando que a relação entre as variáveis e o preço não é puramente linear.

| Métrica | Regressão Linear | Random Forest (Campeão) | Melhoria |
| :--- | :--- | :--- | :--- |
| **R² Score** | 0.8490 | **0.9600** | +13% |
| **Erro (MAE)** | 1.21 Lakhs | **0.63 Lakhs** | -48% (Erro reduzido pela metade) |

### Interpretação dos Resultados
* **R² de 0.96:** O modelo final consegue explicar 96% da variação de preços. Isso é considerado um resultado excepcional para precificação de ativos.
* **Importância das Variáveis:** O Random Forest identificou que o **Preço de Tabela (Present_Price)** é o fator dominante, seguido pela **Idade do Carro**.

## 📦 Como executar
1.  Instale as dependências: `pip install pandas seaborn matplotlib scikit-learn`
2.  Execute o script de comparação:
    ```bash
    python comparacao_modelos.py
    ```

---
