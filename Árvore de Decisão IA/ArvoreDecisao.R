# ── Instalar pacotes (execute uma vez) ───────────────────────
# install.packages(c("rpart", "rpart.plot", "caret", "pROC", "ggplot2", "dplyr"))

library(rpart)       
library(rpart.plot)  
library(caret)       
library(pROC)        
library(ggplot2)
library(dplyr)

set.seed(42)


cat("======================================================\n")
cat("INSPEÇÃO INICIAL\n")
cat("======================================================\n")

df <- read.csv('...data/UCI_Credit_Card.csv', stringsAsFactors = FALSE)

cat("Dimensões:", dim(df), "\n")
cat("\nColunas:\n"); print(names(df))
cat("\nDistribuição da variável alvo:\n")
print(table(df$default.payment.next.month))
cat("\nProporção (%):\n")
print(round(prop.table(table(df$default.payment.next.month)) * 100, 2))


cat("\n======================================================\n")
cat("PRÉ-PROCESSAMENTO\n")
cat("======================================================\n")

df$ID <- NULL

names(df)[names(df) == "default.payment.next.month"] <- "default"

df$default <- factor(df$default,
                     levels = c(0, 1),
                     labels = c("NaoInadim", "Inadim"))

cat("Valores ausentes:", sum(is.na(df)), "\n")
cat("Observações finais:", nrow(df), "\n")
cat("Atributos:", ncol(df) - 1, "\n")


idx    <- createDataPartition(df$default, p = 0.80, list = FALSE)
treino <- df[ idx, ]
teste  <- df[-idx, ]

cat(sprintf("\nTreino : %d registros\n", nrow(treino)))
cat(sprintf("Teste  : %d registros\n",  nrow(teste)))


cat("\n======================================================\n")
cat("TREINAMENTO — ÁRVORE DE DECISÃO\n")
cat("======================================================\n")

arvore <- rpart(
  default ~ .,
  data    = treino,
  method  = "class",
  parms   = list(split = "gini"),       
  control = rpart.control(
    cp        = 0.001,   
    maxdepth  = 5,       
    minsplit  = 100,     
    minbucket = 50       
  )
)

cp_otimo <- arvore$cptable[which.min(arvore$cptable[, "xerror"]), "CP"]
cat(sprintf("CP ótimo (menor xerror): %.6f\n", cp_otimo))

arvore_podada <- prune(arvore, cp = cp_otimo)

cat(sprintf("Número de nós terminais (folhas): %d\n",
            sum(arvore_podada$frame$var == "<leaf>")))


cat("\n======================================================\n")
cat("AVALIAÇÃO NO CONJUNTO DE TESTE\n")
cat("======================================================\n")

y_pred <- predict(arvore_podada, teste, type = "class")
y_prob <- predict(arvore_podada, teste, type = "prob")[, "Inadim"]
y_real <- teste$default

cm  <- confusionMatrix(y_pred, y_real, positive = "Inadim")
acc  <- cm$overall["Accuracy"]
prec <- cm$byClass["Precision"]
rec  <- cm$byClass["Recall"]
f1   <- cm$byClass["F1"]

roc_obj <- roc(y_real, y_prob, quiet = TRUE)
auc_val <- as.numeric(auc(roc_obj))

cat(sprintf("\n  Acurácia:  %.4f (%.2f%%)\n", acc, acc * 100))
cat(sprintf("  Precisão:  %.4f\n", prec))
cat(sprintf("  Revocação: %.4f\n", rec))
cat(sprintf("  F1-score:  %.4f\n", f1))
cat(sprintf("  AUC-ROC:   %.4f\n", auc_val))
cat("\n  Matriz de Confusão:\n")
print(cm$table)
cat("\n  Relatório completo:\n")
print(cm)

ctrl   <- trainControl(method = "cv", number = 5,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)
cv_mod <- train(default ~ ., data = df, method = "rpart",
                trControl = ctrl, metric = "ROC",
                tuneLength = 10)
cat(sprintf("\n  AUC médio (5-fold CV): %.4f\n", max(cv_mod$results$ROC)))


cat("\n======================================================\n")
cat("IMPORTÂNCIA DAS VARIÁVEIS\n")
cat("======================================================\n")

imp <- arvore_podada$variable.importance
if (length(imp) > 0) {
  imp_df <- data.frame(
    variavel    = names(imp),
    importancia = as.numeric(imp)
  ) %>% arrange(desc(importancia))
  print(imp_df)
} else {
  cat("Nenhuma variável com importância registrada.\n")
}


png("arvore_decisao_R.png", width = 1400, height = 700, res = 120)
rpart.plot(
  arvore_podada,
  type    = 4,       
  extra   = 104,     
  fallen.leaves = TRUE,
  main    = "Árvore de Decisão (podada) — UCI Credit Card (R)",
  cex     = 0.75
)
dev.off()
cat("\n✔ arvore_decisao_R.png salvo.\n")

if (length(imp) > 0) {
  top_imp <- head(imp_df, 10)
  top_imp$variavel <- factor(top_imp$variavel,
                             levels = top_imp$variavel[order(top_imp$importancia)])
  p <- ggplot(top_imp, aes(x = variavel, y = importancia)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = "Top Variáveis — Árvore de Decisão (R)",
         x = NULL, y = "Importância") +
    theme_minimal(base_size = 13)
  ggsave("importancia_dt_R.png", p, width = 8, height = 5, dpi = 150)
  cat("✔ importancia_dt_R.png salvo.\n")
}

png("roc_dt_R.png", width = 700, height = 600, res = 120)
plot(roc_obj,
     main = sprintf("Curva ROC — Árvore de Decisão (R)  AUC = %.3f", auc_val),
     col  = "steelblue", lwd = 2,
     xlab = "Taxa de Falso Positivo",
     ylab = "Taxa de Verdadeiro Positivo (Revocação)")
abline(0, 1, lty = 2, col = "gray60")
dev.off()
cat("✔ roc_dt_R.png salvo.\n")

png("cp_table_R.png", width = 700, height = 500, res = 120)
plotcp(arvore, main = "Complexidade vs. Erro — Árvore de Decisão (R)")
dev.off()
cat("✔ cp_table_R.png salvo.\n")

cat("\n✔ Execução concluída com sucesso.\n")