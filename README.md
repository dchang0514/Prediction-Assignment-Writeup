Practical Machine Learning Project - Prediction Assignment
==========================================================

## Introduction

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify **how well they do it**. 

In this project, there are data measured from *accelerometers* on the *belt, forearm, arm,* and *dumbell* of 6 participants. They are asked to perform *barbell lifts* correctly and incorrectly 
in 5 different ways. The goal of this project is to predict the manner in which they did the exercise.

## Goal
The goal of this project is to predict the manner in which atheletes did the exercise. This is the "classe" variable in the training set. I may use any of the other variables to predict with. 
A report is expected to be created describing the followings:

1. How I built my model, 
2. How you use cross validation
3. What I think the expected output of sample error is, and 
4. Why I made the choices yI did. 

Also I will also use the prediction model to predict 20 different test cases.

## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## Approach

The goal of this project is to predict the manner in which they did the exercise. My approach to select a model and make predictions are the following:

1. First download the training and testing data sets. 
2. After exploring the training data set, cleaning up the data by extracting only relevant feature sets. Also, eliminating the "NA" columns.
3. Split the original training data set into two datasets for training and cross validation purpose.
4. Random forest is generally a better model if the goal is for prediction. In other words, we'd want to reduce the variance of the model. Thus, Random forest model is selected.
5. Train the Random model with multiple cores and use cross validaion data set to validate the accuracy of the model.
6. If the accuracy of the model is in satisfied range, the model will be use to predict the testing data set and deliver the answers to the questions

## HTML report page (gh-pages)

Please open this for easy access HTML version of report