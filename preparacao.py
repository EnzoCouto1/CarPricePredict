import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar os dados
df = pd.read_csv('car data.csv')

# 2. Feature Engineering: Criar a "Idade do Carro"
# Vamos assumir que estamos em 2024. Se o carro é de 2014, ele tem 10 anos.
df['Car_Age'] = 2024 - df['Year']

# 3. Limpeza: Remover colunas que não vamos usar
# Removemos 'Year' (pois já temos a idade) e 'Car_Name' (muitos nomes atrapalham agora)
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

# 4. Encoding: Transformar texto em números
# O drop_first=True ajuda a evitar repetições desnecessárias (Dummy Variable Trap)
df = pd.get_dummies(df, drop_first=True)

print("--- Como os dados ficaram após a transformação ---")
print(df.head())

# 5. Separar X (Características) e y (O Alvo/Preço)
X = df.drop('Selling_Price', axis=1)  # Tudo, menos o preço
y = df['Selling_Price']               # Só o preço

# 6. Divisão Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDados prontos!")
print(f"Tamanho do Treino: {X_train.shape[0]} carros")
print(f"Tamanho do Teste: {X_test.shape[0]} carros")