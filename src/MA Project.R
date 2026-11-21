###################################### MA PROJECT ###############################################

# import libraries
require(tidyverse)
library(flexclust)
require(factoextra)
library(readr)
require(cluster)
require(dplyr) 
require(reshape2)
require(MASS)
library(vegan)
require(plyr)
library(gridExtra)
library(class)
library(caret)

################################ EXPLORATORY ANALYSIS ##########################################################

# import data
data1 = read.csv("hp_deprivation_index_scores.csv")
head(data1)
str(data1) 

# convert population variable to numeric
data1$TOTPOP22 = as.numeric(gsub(",", "", data1$TOTPOP22))
class(data1$TOTPOP22)
str(data1) 

# create dataset with numerical values only
data2 = data1[3:15]   

# standardisation
summary(data1[c(4:15)])
data3 = scale(data1[4:15], center = T, scale = T) 

# check for missing value
which(is.na(data1))

# identify rows with outliers (any variable has |z| > 3)
X = data3[sapply(data3, is.numeric)]
z_scores = scale(X)
outlier_rows = which(apply(z_scores, 1, function(row) any(abs(row) > 3)))
print(outlier_rows)

# find percentage of outliers
length(outlier_rows)
nrow(data3)
round(length(outlier_rows) / nrow(data3) * 100, 2)

# visualise outliers with boxplots

# bounded variables
datax = data1[c(4:14)] 
windows(width = 14, height = 12)
boxplot(as.data.frame(datax),notch=T, col = "lightblue",
        main = "Bounded Variables", ylab = "Percentage",
        cex = 1.5, cex.lab = 1.5, cex.axis = 1, cex.main = 1.5)

# population variable
boxplot(data1$TOTPOP22,notch=T, 
        main = "Total Population", ylab = "Population")

# persons per room variable
boxplot(data1$PEROOM22,notch=T, 
        main = "Persons per Room", ylab = "Persons")

# deprivation score and class
plot(data1$Index22_ED_std_rel_wt, data1$TOTPOP22, main = "Deprivation Classification", 
     ylab = "Population", xlab = "Deprivation Score",
     col = c("blue", "red", "green", "yellow", "black", "purple")[data1$Index22_ED_rel_wt_cat], 
     cex = 1.5, cex.lab = 1.5, cex.axis = 1.5, cex.main = 1.5, pch = 16)
legend("topleft", 
       legend = c("E. Disadvantaged","V. Disadvantaged","Disadvantaged","Mar. Below Av.","Mar. Above Av.","Affluent"),  
       col = c("blue", "red", "green", "yellow", "black", "purple"), 
       pch = 16,  bty = "n", y.intersp = 0.8, cex = 1.5) 

# create correlation heatmap 
cor_data3 = round(cor(data3), 2)
melted_cormat = melt(cor_data3)

ggplot(data = melted_cormat, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value), size = 3) +
  scale_fill_gradient2(low = "blue", high = "red",
                       limit = c(-1,1), name = "Correlation") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_text(size=4),
        panel.background = element_blank(),
        legend.position = "bottom") +
  ggtitle("Correlation Heatmap ")

# unique categories in classification scale variable
unique(data1$Index22_ED_rel_wt_lab)

# no. of electoral divisions in each classification level
table(data1$Index22_ED_rel_wt_lab)

# total population per classification level
totals = data1 %>% group_by(Index22_ED_rel_wt_lab) %>%
  summarise(total_population = sum(TOTPOP22, na.rm = T))
totals

sum(data1$TOTPOP22)

# largest and smallest populations
largest = head(sort(data1[[3]], decreasing = T), 3)
smallest = head(sort(data1[[3]], decreasing = F), 3)

data1[which(data1[[3]] %in% largest),2]
largest
data1[which(data1[[3]] %in% smallest),2]
smallest

######################################### PCA ##################################################

#####

# check impact of outliers
Y = data3

# winsorised (1st and 99th percentiles)
Y_win = Y
for(i in 1:ncol(Y)) {
  p1 = quantile(Y[, i], 0.01)
  p99 = quantile(Y[, i], 0.99)
  Y_win[Y[, i] < p1, i] = p1
  Y_win[Y[, i] > p99, i] = p99}
Y_win

# truncated (|z| < 3)
Y_trunc = Y
for(i in 1:ncol(Y)) {
  Y_trunc[Y[, i] > 3, i] = 3
  Y_trunc[Y[, i] < -3, i] = -3}
Y_trunc

# PCA
pca_orig = prcomp(Y)
pca_win = prcomp(Y_win)
pca_trunc = prcomp(Y_trunc)

# correlations with original PC1
round(cor(pca_orig$x[,1], pca_win$x[,1]), 4)
round(cor(pca_orig$x[,1], pca_trunc$x[,1]), 4)

#####

# perform PCA
pca = prcomp(data3) 
pca

round(pca$sdev, 2)
round(pca$rotation, 2) 

# scree plot
fviz_screeplot(pca)

# variables that contribute most to each PC
p1 = fviz_contrib(pca, choice = "var", axes = 1, top = 12)
p2 = fviz_contrib(pca, choice = "var", axes = 2, top = 12)
p3 = fviz_contrib(pca, choice = "var", axes = 3, top = 12)
grid.arrange(p1, p2, p3, nrow = 1, ncol = 3)

# map PCs onto observations
projections = predict(pca)   
head(projections, n = 3)
tail(projections, n = 3)

# check correlation of PCs with deprivation score
data1x = cbind.data.frame(data1$Index22_ED_rel_wt_cat, projections[,1:3])
data1$Index22_ED_rel_wt_cat
cor_data1x = round(cor(data1x), 2)
cor_data1x

# scatterplot matrix for first three PCs
pairs(data1x, main = "Scatterplot of PCs vs Deprivation Score", pch = 16, cex = 0.8,
      labels = c("Deprivation score", "PC1", "PC2", "PC3"),
      col = c("blue", "red", "green", "yellow", "black", "purple")[data1$Index22_ED_rel_wt_cat])

###################################### MDS ##############################################

# dissimilarity matrix
dis = dist(data3, method="man")

# create matrix
dis_mat = as.matrix(dis) # should this include the population of the division

# first three and last three divisions
data1[c(1:3), 2]
data1[c(3415:3417), 2]

# dissimilarity between first three and last three observations
round(dis_mat[c(1:3, 3415:3417), c(1:3, 3415:3417)],2)

# set seed
set.seed(132)

# classical
loc1 = cmdscale(dis, k = 1, eig = T)
loc2 = cmdscale(dis, k = 2, eig = T)
loc3 = cmdscale(dis, k = 3, eig = T)
loc4 = cmdscale(dis, k = 4, eig = T)

GOF_vector1 = c(loc1$GOF[1], loc2$GOF[1], loc3$GOF[1], loc4$GOF[1])
GOF_vector1

# metric least squares 
sam1 = sammon(dis, k = 1)
sam2 = sammon(dis, k = 2)
sam3 = sammon(dis, k = 3)
sam4 = sammon(dis, k = 4)

GOF_vector2 = c(sam1$stress, sam2$stress, sam3$stress, sam4$stress)
GOF_vector2

# Kruskal's
kru1 = isoMDS(dis, k = 1)
kru2 = isoMDS(dis, k = 2)
kru3 = isoMDS(dis, k = 3)
kru4 = isoMDS(dis, k = 4)

GOF_vector3 = c(kru1$stress, kru2$stress, kru3$stress, kru4$stress)
GOF_vector3

# elbow plot
par(mfrow=c(1, 3))
plot(1:4, 1 - GOF_vector1, type = 'l', col = "red",
     xlab = 'Dimensions', ylab = 'Variance', main = 'Classical',
     cex.lab = 1.8, cex.main = 2.0, cex.axis = 1.5)
plot(1:4, GOF_vector2, type = 'l', col = "red",
     xlab = 'Dimensions', ylab = 'Stress', main = 'Metric Least Squares',
     cex.lab = 1.8, cex.main = 2.0, cex.axis = 1.5)
plot(1:4, GOF_vector3, type = 'l', col = "red",
     xlab = 'Dimensions', ylab = 'Stress', main = 'Kruskal',
     cex.lab = 1.8, cex.main = 2.0, cex.axis = 1.5)
par(mfrow=c(1, 1))

# bind sammon 3D MDS with deprivation score
sam3_points = -sam3$points
data1y = cbind.data.frame(Deprivation = data1$Index22_ED_rel_wt_cat, 
                          sam3_points[,1:3])

# scatterplot for sammon 3D MDS
pairs(data1y, main = "Scatterplot of MDS vs Deprivation Score", pch = 16, cex = 0.8,
      labels = c("Deprivation score", "Dimension 1", "Dimension 2", "Dimension 3"),
      col = c("blue", "red", "green", "yellow", "black", "purple")[data1$Index22_ED_rel_wt_cat])

######################################## k-NN ###################################################

# create dataset
data1a = data1
head(data1a)

# modify the class variable
data1a$Index22_ED_rel_wt_lab = revalue(data1a$Index22_ED_rel_wt_lab,
                                       c("Extremely Disadvantaged" = "Disadvantaged",
                                         "Very Disadvantaged" = "Disadvantaged",
                                         "Disadvantaged" = "Disadvantaged",
                                         "Marginally Below Average" = "Not Disadvantaged",
                                         "Marginally Above Average" = "Not Disadvantaged",
                                         "Affluent" = "Not Disadvantaged"))

# proportion of electoral divisions in each new classification level
round(prop.table(table(data1a$Index22_ED_rel_wt_lab))*100,2)

# convert deprivation category variable into factor
data1a$Index22_ED_rel_wt_lab = as.factor(data1a$Index22_ED_rel_wt_lab)

# create dataset with original variables
data1b = cbind.data.frame(data1a$Index22_ED_rel_wt_lab, data3)
colnames(data1b)[1] = "Category" 
head(data1b)

# train/test/validate split ***run code as a chunk***
set.seed(321)
NN = nrow(data1b)  
i_train = sort(sample(1:NN, size = floor(NN*0.70)))
i_nottrain = setdiff(1:NN, i_train)
i_test = sort(sample(i_nottrain, size = length(i_nottrain)/2))
i_valid = sort(setdiff(i_nottrain, i_test))

# training and validation sets
train_orig = data1b[i_train,]
test_orig = data1b[i_test,]
valid_orig = data1b[i_valid,]

# compute PCA on training data
pca2 = prcomp(train_orig[, -1], center = T)
summary(pca2)

# create PC datasets from train_orig, test_orig sets and valid_orig
train_pc2 = cbind.data.frame(train_orig[, 1], pca2$x[, 1:2]) 
colnames(train_pc2)[1] = "Category"
head(train_pc2)

test_pc2_raw = predict(pca2, test_orig[, -1])[, 1:2] 
test_pc2 = cbind.data.frame(test_orig[, 1], test_pc2_raw)
colnames(test_pc2)[1] = "Category"
head(test_pc2)

valid_pc2_raw = predict(pca2, valid_orig[, -1])[, 1:2] 
valid_pc2 = cbind.data.frame(valid_orig[, 1], valid_pc2_raw)
colnames(valid_pc2)[1] = "Category"
head(valid_pc2)

# Set up the dataframe to store k and the misclassification rates
kmax = sqrt(3417)
k = 1:kmax

ntest_orig = nrow(test_orig)
k_summary1 = cbind(k, p = rep(0, kmax))
colnames(k_summary1) = c("k","misclassified rate")

ntest_pc2 = nrow(test_pc2)
k_summary2 = cbind(k, p = rep(0, kmax))
colnames(k_summary2) = c("k","misclassified rate")

### classification for original twelve variables ###

# Performs k-NN for k=1, ..., kmax and calculates the misclassification rate
for(i in 1:kmax){
  result1b = knn(train_orig[,-1], test_orig[,-1], cl = train_orig[, 1], k = i)
  class_agree1 = table(result1b, test_orig[,1])
  sum_agree1 = sum(diag(class_agree1))
  k_summary1[i, 2] = (ntest_orig - sum_agree1) / ntest_orig}
k_summary1

plot(k_summary1[,1], k_summary1[,2],
     xlab = 'k', ylab = 'Misclassification Rate') 

# run knn with k = 4
result1 = knn(train_orig[,-1], test_orig[,-1], cl = train_orig[,1], k = 4)
head(result1)

# confusion matrix
confusionMatrix(result1, test_orig[, 1])

### classification for first two PCs ###

# Performs k-NN for k=1, ..., kmax and calculates the misclassification rate
for(i in 1:kmax){
  result2b = knn(train_pc2[,-1], test_pc2[,-1], cl = train_pc2[, 1], k = i)
  class_agree2 = table(result2b, test_pc2[,1])
  sum_agree2 = sum(diag(class_agree2))
  k_summary2[i, 2] = (ntest_pc2 - sum_agree2) / ntest_pc2}
k_summary2

plot(k_summary2[,1], k_summary2[,2],
     xlab = 'k', ylab = 'Misclassification Rate') 

# run knn with k = 16
result2 = knn(train_pc2[,-1], test_pc2[,-1], cl = train_pc2[,1], k = 16)
head(result2)

# confusion matrix
confusionMatrix(result2, test_pc2[, 1])

### validation

levels(valid_pc2[, 1])

# validation performance - original variables
result_valid1 = knn(train_orig[,-1], valid_orig[,-1], cl = train_orig[, 1], k = 28)
a = confusionMatrix(result_valid1, valid_orig[, 1])
a
a$byClass["F1"]

# validation performance - PCA
result_valid2 = knn(train_pc2[,-1], valid_pc2[,-1], cl = train_pc2[, 1], k = 28)
b = confusionMatrix(result_valid2, valid_pc2[, 1])
b
b$byClass["F1"]

#####

# misclassified rows for PCA model
misclassified1 = which(valid_pc2[,1] != result_valid2)
valid_pc2[misclassified1, ]

# plot validation set with correct categories 
plot(valid_pc2$PC1, valid_pc2$PC2, main = "PCA Classification", 
     ylab = "PC2", xlab = "PC1",
     col = c("green", "black")[valid_pc2$Category], cex = 2, pch = 16)
legend("topleft", 
       legend = c("Disadvantaged","Not Disadvantaged"),  
       col = c("green", "black"), 
       pch = 16,  bty = "n", y.intersp = 0.6, cex = 0.8) 

# highlight rows misclassified by PCA model in red 
points(valid_pc2$PC1[misclassified1], valid_pc2$PC2[misclassified1], 
       col = "red", cex = 2, pch = 1)

# misclassified rows for original variables model
misclassified2 = which(valid_orig[,1] != result_valid1)
valid_orig[misclassified2, ]

# plot validation set with correct categories 
plot(valid_pc2$PC1, valid_pc2$PC2, main = "Original Variables Classification", 
     ylab = "PC2", xlab = "PC1",
     col = c("green", "black")[valid_pc2$Category], cex = 2, pch = 16)
legend("topleft", 
       legend = c("Disadvantaged","Not Disadvantaged"),  
       col = c("green", "black"), 
       pch = 16,  bty = "n", y.intersp = 0.6, cex = 0.8) 

# highlight rows misclassified by original variables model in red 
points(valid_pc2$PC1[misclassified2], valid_pc2$PC2[misclassified2], 
       col = "red", cex = 2, pch = 1)

#############################################################################################
