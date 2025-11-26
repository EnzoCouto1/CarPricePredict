import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. PREPARAÇÃO (Igualzinho antes)
df = pd.read_csv('car data.csv')
df['Car_Age'] = 2024 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODELO 1: Regressão Linear (O Simples/Baseline) ---
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
pred_lin = lin_reg.predict(X_test)
r2_lin = r2_score(y_test, pred_lin)
mae_lin = mean_absolute_error(y_test, pred_lin)

# --- MODELO 2: Random Forest (O Complexo/Desafiante) ---
# Random Forest cria centenas de "árvores de decisão" e tira a média.
# Geralmente é muito bom para capturar padrões não-lineares.
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
pred_rf = rf_reg.predict(X_test)
r2_rf = r2_score(y_test, pred_rf)
mae_rf = mean_absolute_error(y_test, pred_rf)

# --- RESULTADO FINAL: A Batalha ---
print("="*40)
print("     RELATÓRIO DE COMPARAÇÃO")
print("="*40)
print(f"{'Métrica':<15} | {'Regressão Linear':<15} | {'Random Forest':<15}")
print("-" * 55)
print(f"{'R² Score':<15} | {r2_lin:.4f}          | {r2_rf:.4f}")
print(f"{'Erro MAE':<15} | {mae_lin:.4f}          | {mae_rf:.4f}")
print("-" * 55)

if r2_rf > r2_lin:
    print("\n🏆 VENCEDOR: Random Forest (Capturou melhor os padrões complexos)")
else:
    print("\n🏆 VENCEDOR: Regressão Linear (O problema é simples e linear)")