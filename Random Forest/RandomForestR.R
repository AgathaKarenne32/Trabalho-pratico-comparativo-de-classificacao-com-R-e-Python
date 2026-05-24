library(caret)
library(randomForest)

# 1. Carregamento dos Dados (lê da mesma pasta)
df <- read.csv("UCI_Credit_Card.csv")

if("ID" %in% colnames(df)) { df$ID <- NULL }

colnames(df)[colnames(df) == "default.payment.next.month"] <- "inadimplente"
df$inadimplente <- as.factor(df$inadimplente)

# Divisão 70/30
set.seed(42)
indice_treino <- createDataPartition(df$inadimplente, p = 0.7, list = FALSE)
treino <- df[indice_treino, ]
teste <- df[-indice_treino, ]

# 2. Treino do Modelo Random Forest 
# (importance = TRUE é obrigatório para gerar o gráfico de variáveis)
modelo_rf <- randomForest(inadimplente ~ ., data = treino, importance = TRUE)

# 3. Previsões e Métricas
previsoes <- predict(modelo_rf, newdata = teste)
resultados <- confusionMatrix(previsoes, teste$inadimplente, positive = "1", mode = "prec_recall")

print("=== MÉTRICAS RANDOM FOREST (R) ===")
print(resultados$overall["Accuracy"])
print(resultados$byClass[c("Precision", "Recall", "F1")])

# 4. Gerar e Salvar a Matriz de Confusão
png("matriz_confusao_rf_r.png", width = 800, height = 800, res = 150)
fourfoldplot(resultados$table, color = c("#CC6666", "#99CC99"), 
             conf.level = 0, margin = 1, main = "Matriz de Confusão: Random Forest (R)")
dev.off()

# 5. Gerar e Salvar a Importância das Variáveis
png("importancia_rf_r.png", width = 800, height = 600, res = 120)
varImpPlot(modelo_rf, main="Importância das Variáveis (Random Forest - R)", col="blue", pch=19)
dev.off()

cat("\n✅ Imagens 'matriz_confusao_rf_r.png' e 'importancia_rf_r.png' salvas com sucesso na sua pasta!\n")