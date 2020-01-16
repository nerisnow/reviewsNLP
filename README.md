# Restaurant Review Analysis Using Machine Learning Models

This repository includes a project on Sentiment Analysis of Reviews using Machine Learning Models. As our baseline model, Naive Bayes Classifier is used for the classification of positive and negative reviews. Machine learning models used here are:
1. Logistic Regression
2. Naive Bayes Classifier
3. Random Forest Classifier
4. Support Vector Machines
5. K-Nearest Neighbors

## Getting Started
Clone this repo into you local directory with `git clone https://github.com/nerisnow/reviewsNLP.git`


## Running the trained model
Run the following code in the terminal to run the trained model.

`python3 trainfile.py`
As models are already available in the repo, the next step would be to run the test file.

## Running the test file for evaluation
Run the following code in the terminal to run the test file for evaluation.

`python3 testfile.py`

## Running the cross validation tests
Run the following code in the terminal to run the cross validation tests and evaluate the best model out of all the models.

`python3 cross_val.py`

## Description
`NBModel.sav` is the trained Naive Bayes model saved in a file.
`LRModel.sav` is the trained Logistic Regression model saved in a file.
`RFCModel.sav` is the trained Random Forest Classifier model saved in a file.
`SVMModel.sav` is the trained Support Vector Machine model saved in a file.
`KNNModel.sav` is the trained K-Nearest Neighbor model saved in a file.

`trainfile.py` contains the code to training the train dataset.
`testfile.py` contains the code to testing in the test dataset.
`cross_val.py` contains the code to perform cross validation.

`reviews.csv` is the dataset used for the project.


## Dataset link
The dataset used was importedd from : [https://www.kaggle.com/vigneshwarsofficial/reviews](https://www.kaggle.com/vigneshwarsofficial/reviews)

