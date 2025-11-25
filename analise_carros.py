import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregamento dos Dados
# Certifique-se de que o nome do arquivo aqui é igual ao que você baixou
try:
    df = pd.read_csv('car data.csv')
    print("✅ Dados carregados com sucesso!")
except FileNotFoundError:
    print("❌ Erro: Arquivo não encontrado. Verifique se o nome está correto e se está na mesma pasta.")

# 2. Espiada Inicial (Primeiras 5 linhas)
# Isso ajuda a ver se as colunas foram lidas corretamente
print("\n--- Primeiras 5 linhas do Dataset ---")
print(df.head())

# 3. Informações Técnicas
print("\n--- Informações do Dataset ---")
print(df.info())

# 4. Estatísticas Descritivas
# Mostra média, mínimo, máximo e desvio padrão dos preços e anos
print("\n--- Estatísticas Básicas ---")
print(df.describe())

# 5. Verificando valores nulos
print("\n--- Quantidade de valores nulos por coluna ---")
print(df.isnull().sum())