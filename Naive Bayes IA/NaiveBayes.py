import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# 1. Carregando a base de dados 
# (Baixe do Kaggle e coloque o arquivo CSV na mesma pasta do seu script)
df = pd.read_csv('UCI_Credit_Card.csv')

# 2. Pré-processamento básico
# A base costuma vir com uma coluna 'ID' que não ajuda na previsão, vamos retirá-la
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Separando atributos explicativos (X) e a variável alvo (y)
# No Kaggle, a variável de calote se chama 'default.payment.next.month'
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

# 3. Divisão entre Treino (70%) e Teste (30%) 
# O random_state=42 fixa a semente para garantir a reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treinamento do modelo Naive Bayes
modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)

# 5. Previsões
previsoes = modelo_nb.predict(X_test)

# 6. Avaliação e Métricas
print("=== Resultados Naive Bayes (Python) ===")
print(f"Acurácia: {accuracy_score(y_test, previsoes):.4f}")
print(f"Precisão: {precision_score(y_test, previsoes):.4f}")
print(f"Revocação: {recall_score(y_test, previsoes):.4f}")
print(f"F1-Score: {f1_score(y_test, previsoes):.4f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, previsoes))