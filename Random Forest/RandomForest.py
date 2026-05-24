import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Carregamento dos dados à prova de falhas
pasta_atual = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(pasta_atual, 'UCI_Credit_Card.csv')

df = pd.read_csv(caminho_csv)

if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# Divisão 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Treino do Modelo Random Forest
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)

# 3. Previsões e Métricas
previsoes = modelo_rf.predict(X_test)
matriz = confusion_matrix(y_test, previsoes)

print("=== MÉTRICAS RANDOM FOREST (PYTHON) ===")
print(f"Acurácia:  {accuracy_score(y_test, previsoes):.4f}")
print(f"Precisão:  {precision_score(y_test, previsoes):.4f}")
print(f"Revocação: {recall_score(y_test, previsoes):.4f}")
print(f"F1-Score:  {f1_score(y_test, previsoes):.4f}")

# 4. Gerar e Guardar a Matriz de Confusão
plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão: Random Forest (Python)')
plt.xlabel('Previsão do Modelo')
plt.ylabel('Realidade')
plt.tight_layout()
plt.savefig(os.path.join(pasta_atual, "matriz_confusao_rf_python.png"), dpi=300)
plt.close()

# 5. Gerar e Guardar a Importância das Variáveis (Excelente para o relatório!)
importancias = modelo_rf.feature_importances_
df_imp = pd.DataFrame({'Variável': X.columns, 'Importância': importancias}).sort_values(by='Importância', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importância', y='Variável', data=df_imp, palette='viridis')
plt.title('Top 10 Variáveis Mais Importantes (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(pasta_atual, "importancia_rf_python.png"), dpi=300)
plt.close()

print("\n✅ Imagens 'matriz_confusao_rf_python.png' e 'importancia_rf_python.png' guardadas com sucesso!")