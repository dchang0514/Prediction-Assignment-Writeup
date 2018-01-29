---
title: "Practical Machine Learning Project - Prediction Assignment"
author: "David W Chang"
date: "01/28/2018"
output: 
  html_document:
    keep_md: true
---

## Overview
Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify **how well they do it**. 

In this project, there are data measured from *accelerometers* on the *belt, forearm, arm,* and *dumbell* of 6 participants. They are asked to perform *barbell lifts* correctly and incorrectly in 5 different ways. The goal of this project is to predict the manner in which they did the exercise.

## Approach
The goal of this project is to predict the manner in which they did the exercise. My approach to select a model and make predictions are the following:

1. First download the training and testing data sets. 
2. After exploring the training data set, cleaning up the data by extracting only relevant feature sets. Also, eliminating the "NA" columns.
3. Split the original training data set into two datasets for training and cross validation purpose.
4. Random forest is generally a better model if the goal is for prediction. In other words, we'd want to reduce the variance of the model. Thus, Random forest model is selected.
5. Train the Random model with multiple cores and use cross validaion data set to validate the accuracy of the model.
6. If the accuracy of the model is in satisfied range, the model will be use to predict the testing data set and deliver the answers to the questions

## Load required packages


```r
# Load appropriate library
require(caret)
require(corrplot)
require(Rtsne)
require(stats)
require(knitr)
require(ggplot2)
require(randomForest)
require(foreach)
require(doParallel)

# setup cache
knitr::opts_chunk$set(echo = TRUE)
```

## Preparing Data
### Downloading Data to a local directory
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


Thanks for the source site: http://groupware.les.inf.puc-rio.br/har providing the data sets that are used in this project


```r
# Download training and testing data to a local directory
train_url ="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
train_fname = "./train_data.csv"
test_fname = "./test_data.csv"
if(!file.exists(train_fname))
  download.file(train_url, destfile=train_fname, method="curl")
if(!file.exists(test_fname))
  download.file(test_url, destfile=test_fname, method="curl")

# load the CSV files as data.frame 
train.data = read.csv(train_fname, na.strings=c("NA",""))
testing.data = read.csv(test_fname, na.strings=c("NA",""))
dim(train.data)
```

```
## [1] 19622   160
```

```r
dim(testing.data)
```

```
## [1]  20 160
```

```r
#str(train.data)
#str(test.data)
```

The original training data has 19622 rows of measurements and 160 features. Whereas the testing data has 20 rows and the same 160 features. There is one column of target outcome named `classe` in training data set.

### Split Training Data
The train dataset is split into training and crossval dataset and remember outcomes column


```r
set.seed(8888)

Partition.idx = createDataPartition(train.data$classe, p=0.70, list=FALSE)
training.data = train.data[Partition.idx,]
crossval.data  = train.data[-Partition.idx,]

# Save outcomes
training.classe = train.data[Partition.idx, "classe"]
crossval.classe = train.data[-Partition.idx, "classe"]

dim.train = dim(training.data); print(dim.train)
```

```
## [1] 13737   160
```

```r
dim.cross = dim(crossval.data); print(dim.cross)
```

```
## [1] 5885  160
```

```r
str(training.data[,1:10])
```

```
## 'data.frame':	13737 obs. of  10 variables:
##  $ X                   : int  1 3 5 6 7 8 9 11 12 13 ...
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 820366 196328 304277 368296 440390 484323 500302 528316 560359 ...
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window          : int  11 11 12 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.42 1.48 1.45 1.42 1.42 1.43 1.45 1.43 1.42 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.06 8.09 8.13 8.16 8.18 8.18 8.2 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
```

### Cleaning All Data 
The project task asks to use data from accelerometers on only *belt*, *forearm*, *arm*, and *dumpbell*, so
filter these columns only accrodingly.


```r
# filter columns on: belt, forearm, arm, dumbell
filter = grepl("belt|arm|dumbell", names(train.data))
training.data = training.data[, filter]
crossval.data = crossval.data[, filter]
testing.data = testing.data[, filter]
#str(training.data)
```

Instead of dealing with less-accurate missing data columns, remove 
all columns with NA values.


```r
# remove columns with NA, use testing data as referal for NA
cols.without.na = colSums(is.na(testing.data)) == 0
training.data = training.data[, cols.without.na]
crossval.data = crossval.data[, cols.without.na]
testing.data = testing.data[, cols.without.na]
#str(training.data)
```

### Check for features's variance

Based on the principal component analysis, it is important that features have maximum variance for maximum uniqueness, 
so that each feature is as distant as possible from the other features.   

```r
# check for zero variance
zero.var = nearZeroVar(training.data, saveMetrics=TRUE)
zero.var
```

```
##                     freqRatio percentUnique zeroVar   nzv
## roll_belt            1.072755     8.0512485   FALSE FALSE
## pitch_belt           1.066667    12.2224649   FALSE FALSE
## yaw_belt             1.078652    13.0232220   FALSE FALSE
## total_accel_belt     1.061394     0.2111087   FALSE FALSE
## gyros_belt_x         1.054303     0.9463493   FALSE FALSE
## gyros_belt_y         1.135986     0.4804542   FALSE FALSE
## gyros_belt_z         1.094891     1.2084152   FALSE FALSE
## accel_belt_x         1.023551     1.1792968   FALSE FALSE
## accel_belt_y         1.130150     0.9973065   FALSE FALSE
## accel_belt_z         1.076312     2.1183665   FALSE FALSE
## magnet_belt_x        1.103586     2.2493994   FALSE FALSE
## magnet_belt_y        1.112867     2.1110868   FALSE FALSE
## magnet_belt_z        1.021277     3.1811895   FALSE FALSE
## roll_arm            44.884615    17.5584189   FALSE FALSE
## pitch_arm           93.400000    20.2009172   FALSE FALSE
## yaw_arm             30.311688    19.0507389   FALSE FALSE
## total_accel_arm      1.015504     0.4731746   FALSE FALSE
## gyros_arm_x          1.000000     4.5715950   FALSE FALSE
## gyros_arm_y          1.397260     2.6643372   FALSE FALSE
## gyros_arm_z          1.113260     1.7034287   FALSE FALSE
## accel_arm_x          1.040984     5.5470627   FALSE FALSE
## accel_arm_y          1.100671     3.8363544   FALSE FALSE
## accel_arm_z          1.054945     5.5907403   FALSE FALSE
## magnet_arm_x         1.105263     9.6382034   FALSE FALSE
## magnet_arm_y         1.078125     6.2750237   FALSE FALSE
## magnet_arm_z         1.101266     9.1286307   FALSE FALSE
## roll_forearm        11.500000    13.7948606   FALSE FALSE
## pitch_forearm       58.744681    19.0434593   FALSE FALSE
## yaw_forearm         15.081967    12.9358666   FALSE FALSE
## total_accel_forearm  1.143023     0.4877339   FALSE FALSE
## gyros_forearm_x      1.040541     2.1038072   FALSE FALSE
## gyros_forearm_y      1.031128     5.2777171   FALSE FALSE
## gyros_forearm_z      1.089855     2.0965276   FALSE FALSE
## accel_forearm_x      1.080645     5.6416976   FALSE FALSE
## accel_forearm_y      1.041096     7.0976196   FALSE FALSE
## accel_forearm_z      1.069307     4.0183446   FALSE FALSE
## magnet_forearm_x     1.120690    10.6136711   FALSE FALSE
## magnet_forearm_y     1.129032    13.3071267   FALSE FALSE
## magnet_forearm_z     1.047619    11.7492902   FALSE FALSE
```
There is no features without variability (all has enough variance). So there is no feature to be removed further.  

### Plot of correlation matrix  

Plot a correlation matrix between features to validate the principal component analysis.
The plot below shows average of correlation is not too high, so no further PCA processing is performed.   

```r
corrplot.mixed(cor(training.data), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
```

![](Prediction_Assignment_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

## Training the model
To train the random forest model, we model 250x4(1000) trees. We have four cores so we split up the problem into four pieces. This is accomplished by executing the randomForest function four times, with the ntree argument set to 250

```r
registerDoParallel()
variables <- training.data[-ncol(training.data)]
rf = foreach(ntree=rep(250, 4), .combine=randomForest::combine, .packages='randomForest') %dopar% {
  randomForest(variables, training.classe, ntree=ntree) 
}
training.predictions <- predict(rf, newdata=training.data)
confusionMatrix(training.predictions, training.classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
## Validating the model 
Using the cross validate data set to validate the model 


```r
crossval.predictions <- predict(rf, newdata=crossval.data)
confusionMatrix(crossval.predictions, crossval.classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    5    0    0    0
##          B    0 1133   10    0    0
##          C    0    1 1009   10    0
##          D    1    0    7  951    1
##          E    0    0    0    3 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9911, 0.9954)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9918          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9947   0.9834   0.9865   0.9991
## Specificity            0.9988   0.9979   0.9977   0.9982   0.9994
## Pos Pred Value         0.9970   0.9913   0.9892   0.9906   0.9972
## Neg Pred Value         0.9998   0.9987   0.9965   0.9974   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1925   0.1715   0.1616   0.1837
## Detection Prevalence   0.2851   0.1942   0.1733   0.1631   0.1842
## Balanced Accuracy      0.9991   0.9963   0.9906   0.9923   0.9992
```

The validation result showed the prediction accuracy around 0.9935. The trained model is good to go!

## Predict the answers to the 20 questions with new testing data


```r
testing.predictions <- predict(rf, newdata=testing.data)
testing.predictions
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

## Coursera provided code for submission

Method to write answers to separate .txt files


```r
paw_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename, quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
```

## Write the answer to text files


```r
paw_write_files(testing.predictions)
```
