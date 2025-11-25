import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --- 1. PREPARAÇÃO (Igual ao passo anterior) ---
df = pd.read_csv('car data.csv')
df['Car_Age'] = 2024 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. TREINAMENTO DO MODELO ---
# Instanciamos o algoritmo
modelo = LinearRegression()

# A mágica acontece aqui: A IA aprende a relação entre X (características) e y (preço)
modelo.fit(X_train, y_train)

print("✅ Modelo treinado com sucesso!")

# --- 3. PREVISÃO ---
# Pedimos para o modelo prever os preços dos carros de teste (que ele nunca viu)
predicoes = modelo.predict(X_test)

# --- 4. AVALIAÇÃO (Métricas para o seu Relatório) ---
score_r2 = r2_score(y_test, predicoes)
mae = mean_absolute_error(y_test, predicoes)

print(f"\n--- Resultados da Avaliação ---")
print(f"R² Score: {score_r2:.2f} (Quanto mais perto de 1.0, melhor)")
print(f"Erro Médio Absoluto (MAE): {mae:.2f}")

# --- 5. GRÁFICO: Real vs Predito ---
# Se o modelo fosse perfeito, todos os pontos estariam na linha pontilhada vermelha
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicoes)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Preço Real')
plt.ylabel('Preço que o Modelo Previu')
plt.title('Performance do Modelo: Preço Real vs Predito')
plt.show()


# --- 6. INTERPRETAÇÃO (Para o Relatório) ---
# Vamos ver quais colunas têm mais "peso" na decisão do preço
coeficientes = pd.DataFrame(modelo.coef_, X.columns, columns=['Coeficiente'])
coeficientes = coeficientes.sort_values(by='Coeficiente', ascending=False)

print("\n--- O que mais valoriza/desvaloriza o carro? ---")
print(coeficientes)