import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando novamente (caso tenha fechado)
df = pd.read_csv('car data.csv')

# Configuração visual
sns.set_theme(style="whitegrid")

# GRÁFICO 1: Correlação (O que tem a ver com o quê?)
colunas_numericas = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
sns.heatmap(colunas_numericas.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de Correlação - O que influencia o preço?")
plt.show()

# GRÁFICO 2: Preço de Venda x Ano
# A tendência deve ser: quanto mais novo (ano maior), maior o preço.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Year', y='Selling_Price', hue='Fuel_Type', s=100)
plt.title("Preço de Venda por Ano e Combustível")
plt.xlabel("Ano do Carro")
plt.ylabel("Preço de Venda (em Lakhs)")
plt.show()

# GRÁFICO 3: Quem vende mais caro? (Vendedor Individual vs Loja)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Seller_Type', y='Selling_Price')
plt.title("Distribuição de Preço por Tipo de Vendedor")
plt.show()