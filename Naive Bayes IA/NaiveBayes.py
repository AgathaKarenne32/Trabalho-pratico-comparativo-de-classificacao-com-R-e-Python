import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Carregando e separando os dados
# Lendo o arquivo que está na mesma pasta do script
pasta_atual = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(pasta_atual, 'UCI_Credit_Card.csv')

df = pd.read_csv(caminho_csv) 

if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# Divisão 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Treinamento do modelo Naive Bayes
modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)

# 3. Previsões e Métricas
previsoes = modelo_nb.predict(X_test)

acuracia = accuracy_score(y_test, previsoes)
precisao = precision_score(y_test, previsoes)
revocacao = recall_score(y_test, previsoes)
f1 = f1_score(y_test, previsoes)
matriz = confusion_matrix(y_test, previsoes)

print("=== MÉTRICAS NAIVE BAYES (PYTHON) ===")
print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Revocação: {revocacao:.4f}")
print(f"F1-Score:  {f1:.4f}")

# 4. Gerando e Salvando a Imagem da Matriz de Confusão
plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão: Naive Bayes (Python)')
plt.xlabel('Previsão do Modelo')
plt.ylabel('Realidade')
plt.tight_layout()

# Salva a imagem na mesma pasta
plt.savefig("matriz_confusao_nb_python.png", dpi=300)
plt.close()
print("\n✅ Imagem 'matriz_confusao_nb_python.png' salva com sucesso!")