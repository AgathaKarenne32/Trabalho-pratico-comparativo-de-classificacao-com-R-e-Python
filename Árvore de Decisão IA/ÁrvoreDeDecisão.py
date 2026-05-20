import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

SEED = 42
np.random.seed(SEED)

df = pd.read_csv('...data/UCI_Credit_Card.csv')

print("=" * 60)
print("INSPEÇÃO INICIAL")
print("=" * 60)
print(f"Dimensões: {df.shape}")
print(f"\nPrimeiras linhas:\n{df.head(3)}")
print(f"\nDistribuição da variável alvo:")
print(df["default.payment.next.month"].value_counts())
print(f"\nProporção (%):")
print((df["default.payment.next.month"].value_counts(normalize=True) * 100).round(2))


print("\n" + "=" * 60)
print("PRÉ-PROCESSAMENTO")
print("=" * 60)

df.drop(columns=["ID"], inplace=True)

print(f"Valores ausentes: {df.isnull().sum().sum()}")

df.rename(columns={"default.payment.next.month": "default"}, inplace=True)

X = df.drop(columns=["default"])
y = df["default"]

print(f"\nAtributos: {X.shape[1]} variáveis")
print(f"Observações: {len(y)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

print(f"\nTreino : {X_train.shape[0]} registros")
print(f"Teste  : {X_test.shape[0]} registros")


print("\n" + "=" * 60)
print("TREINAMENTO — ÁRVORE DE DECISÃO")
print("=" * 60)

dt = DecisionTreeClassifier(
    criterion        = "gini",   
    max_depth        = 5,        
    min_samples_leaf = 50,       
    min_samples_split= 100,      
    random_state     = SEED
)

dt.fit(X_train, y_train)

print(f"Profundidade real da árvore: {dt.get_depth()}")
print(f"Número de folhas           : {dt.get_n_leaves()}")


print("\n" + "=" * 60)
print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("=" * 60)

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n  Acurácia:  {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precisão:  {prec:.4f}")
print(f"  Revocação: {rec:.4f}")
print(f"  F1-score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc:.4f}")

print(f"\n  Matriz de Confusão:")
print(f"  {cm}")

print(f"\n  Relatório Completo:")
print(classification_report(y_test, y_pred,
      target_names=["Não Inadimplente", "Inadimplente"]))

cv_f1 = cross_val_score(dt, X, y, cv=5, scoring="f1")
cv_auc = cross_val_score(dt, X, y, cv=5, scoring="roc_auc")
print(f"  F1  médio 5-fold CV: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"  AUC médio 5-fold CV: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")


print("\n" + "=" * 60)
print("IMPORTÂNCIA DAS VARIÁVEIS (Gini)")
print("=" * 60)

importancias = (
    pd.Series(dt.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)
print(importancias[importancias > 0].to_string())



fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Não Inadimplente", "Inadimplente"],
            yticklabels=["Não Inadimplente", "Inadimplente"])
ax.set_title("Matriz de Confusão — Árvore de Decisão (Python)")
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusao_dt_python.png", dpi=150)
plt.close()
print("\n✔ matriz_confusao_dt_python.png salvo.")

top10 = importancias[importancias > 0].head(10)
fig, ax = plt.subplots(figsize=(8, 5))
top10.sort_values().plot(kind="barh", color="steelblue", ax=ax)
ax.set_title("Top Variáveis — Árvore de Decisão (Python)")
ax.set_xlabel("Importância (Gini)")
plt.tight_layout()
plt.savefig("importancia_dt_python.png", dpi=150)
plt.close()
print("✔ importancia_dt_python.png salvo.")

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"Árvore de Decisão (AUC = {auc:.3f})", color="steelblue")
ax.plot([0, 1], [0, 1], "k--", label="Aleatório")
ax.set_xlabel("Taxa de Falso Positivo")
ax.set_ylabel("Taxa de Verdadeiro Positivo (Revocação)")
ax.set_title("Curva ROC — Árvore de Decisão (Python)")
ax.legend()
plt.tight_layout()
plt.savefig("roc_dt_python.png", dpi=150)
plt.close()
print("✔ roc_dt_python.png salvo.")

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt, feature_names=X.columns.tolist(),
          class_names=["Não Inadim.", "Inadim."],
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Árvore de Decisão (profundidade=5) — UCI Credit Card", fontsize=13)
plt.tight_layout()
plt.savefig("arvore_decisao_python.png", dpi=120)
plt.close()
print("✔ arvore_decisao_python.png salvo.")

print("\n" + "=" * 60)
print("REGRAS DA ÁRVORE (texto)")
print("=" * 60)
print(export_text(dt, feature_names=list(X.columns), max_depth=3))

print("\n✔ Execução concluída com sucesso.")