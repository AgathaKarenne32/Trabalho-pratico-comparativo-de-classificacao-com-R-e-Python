library(caret)
library(e1071)

# 1. Carregando e Preparando os Dados
# Lendo o arquivo que está na mesma pasta do script
df <- read.csv("UCI_Credit_Card.csv") 

if("ID" %in% colnames(df)) { df$ID <- NULL }

colnames(df)[colnames(df) == "default.payment.next.month"] <- "inadimplente"
df$inadimplente <- as.factor(df$inadimplente)

# Divisão 70/30
set.seed(42)
indice_treino <- createDataPartition(df$inadimplente, p = 0.7, list = FALSE)
treino <- df[indice_treino, ]
teste <- df[-indice_treino, ]

# 2. Treinamento do Modelo Naive Bayes
modelo_nb <- naiveBayes(inadimplente ~ ., data = treino)

# 3. Previsões e Matriz
previsoes <- predict(modelo_nb, newdata = teste)
resultados <- confusionMatrix(previsoes, teste$inadimplente, positive = "1", mode = "prec_recall")

print("=== MÉTRICAS NAIVE BAYES (R) ===")
print(resultados$overall["Accuracy"])
print(resultados$byClass[c("Precision", "Recall", "F1")])

# 4. Gerando e Salvando a Imagem da Matriz de Confusão
png("matriz_confusao_nb_r.png", width = 800, height = 800, res = 150)
fourfoldplot(resultados$table, color = c("#CC6666", "#99CC99"), 
             conf.level = 0, margin = 1, main = "Matriz de Confusão: Naive Bayes (R)")
dev.off()

cat("\n✅ Imagem 'matriz_confusao_nb_r.png' salva com sucesso na mesma pasta!\n")