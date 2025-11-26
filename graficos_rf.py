import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- PREPARAÇÃO ---
df = pd.read_csv('car data.csv')
df['Car_Age'] = 2024 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TREINAMENTO (Random Forest) ---
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
predicoes = rf_reg.predict(X_test)

# --- GRÁFICO 1: Real vs Previsto ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predicoes, color='green', alpha=0.6)
# Linha de perfeição
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto pelo Random Forest')
plt.title('Performance Random Forest (R² = 0.96)')
plt.show()

# --- GRÁFICO 2: O que é mais importante? ---
# O Random Forest nos diz quais colunas ele mais usou para decidir o preço
importancias = pd.Series(rf_reg.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importancias.values, y=importancias.index, palette='viridis')
plt.title('Importância das Variáveis (O segredo do preço)')
plt.xlabel('Grau de Importância')
plt.show()