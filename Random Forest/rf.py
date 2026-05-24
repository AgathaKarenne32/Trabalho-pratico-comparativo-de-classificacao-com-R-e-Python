import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# 1. Carregando a base original
df = pd.read_csv('UCI_Credit_Card.csv')

# Removendo a coluna ID
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Separando X e y
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# 2. Divisão treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Treinamento do Random Forest
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)

# 4. Previsões e Avaliação
previsoes = modelo_rf.predict(X_test)

print("=== Resultados Random Forest (Python) ===")
print(f"Acurácia: {accuracy_score(y_test, previsoes):.4f}")
print(f"Precisão: {precision_score(y_test, previsoes):.4f}")
print(f"Revocação: {recall_score(y_test, previsoes):.4f}")
print(f"F1-Score: {f1_score(y_test, previsoes):.4f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, previsoes))