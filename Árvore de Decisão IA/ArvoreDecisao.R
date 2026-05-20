library(rpart)       

set.seed(42)

cat("======================================================\n")
cat("INSPEÇÃO INICIAL\n")
cat("======================================================\n")

df <- read.csv('../data/UCI_Credit_Card.csv', stringsAsFactors = FALSE)

cat("Dimensões:", dim(df), "\n")
cat("\nDistribuição da variável alvo:\n")
print(table(df$default.payment.next.month))
cat("\nProporção (%):\n")
print(round(prop.table(table(df$default.payment.next.month)) * 100, 2))


cat("\n======================================================\n")
cat("PRÉ-PROCESSAMENTO\n")
cat("======================================================\n")

df$ID <- NULL

names(df)[names(df) == "default.payment.next.month"] <- "default"

df$default <- factor(df$default, levels = c(0, 1), labels = c("NaoInadim", "Inadim"))

cat("Valores ausentes:", sum(is.na(df)), "\n")
cat("Observações finais:", nrow(df), "\n")
cat("Atributos:", ncol(df) - 1, "\n")

tamanho_treino <- floor(0.80 * nrow(df))
indices_treino <- sample(seq_len(nrow(df)), size = tamanho_treino)

treino <- df[ indices_treino, ]
teste  <- df[-indices_treino, ]

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
cat("AVALIAÇÃO NO CONJUNTO DE TESTE (MÉTRICAS OBRIGATÓRIAS)\n")
cat("======================================================\n")

y_pred <- predict(arvore_podada, teste, type = "class")
y_real <- teste$default

matriz_confusao <- table(Previsso = y_pred, Real = y_real)

cat("\nMatriz de Confusão:\n")
print(matriz_confusao)

VP <- matriz_confusao["Inadim", "Inadim"]
VN <- matriz_confusao["NaoInadim", "NaoInadim"]
FP <- matriz_confusao["Inadim", "NaoInadim"]
FN <- matriz_confusao["NaoInadim", "Inadim"]

acuracia   <- (VP + VN) / sum(matriz_confusao)
precisao   <- VP / (VP + FP)
revocacao  <- VP / (VP + FN)
f1_score   <- 2 * (precisao * revocacao) / (precisao + revocacao)

cat("\n--- Relatório Comparativo Final ---\n")
cat(sprintf("Acurácia  : %.4f (%.2f%%)\n", acuracia, acuracia * 100))
cat(sprintf("Precisão  : %.4f\n", precisao))
cat(sprintf("Revocação : %.4f\n", revocacao))
cat(sprintf("F1-Score  : %.4f\n", f1_score))


cat("\n======================================================\n")
cat("IMPORTÂNCIA DAS VARIÁVEIS (NATIVO)\n")
cat("======================================================\n")

importancia <- arvore_podada$variable.importance
if (length(importancia) > 0) {
  imp_df <- data.frame(
    Variavel = names(importancia),
    Importancia = as.numeric(importancia)
  )
  imp_df <- imp_df[order(-imp_df$Importancia), ]
  print(head(imp_df, 10))
} else {
  cat("Nenhuma variável com importância registrada.\n")
}

cat("\n✔ Execução concluída com sucesso sem dependências externas.\n")
