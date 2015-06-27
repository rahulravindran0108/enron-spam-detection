---
output: pdf_document
---

## Identifying Fraud from Enron Emails and Financial Data

### Introduction

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for to executives.

Utilizing the classifiers and techniques taught in Into to Machine Learning Class, I built a classifier to detect if a person is culpable or not.

### Short Questions
> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it.  As part of your answer, give some background on the dataset and how it can be used to answer the project question.  Were there any outliers in the data when you got it, and how did you handle those?

The features in the data fall into three major types, namely financial features, email features and POI labels. 

- financial features: salary, deferral_payments, total_payments, loan_advances, bonus, restricted_stock_deferred, deferred_income, total_stock_value, expenses, exercised_stock_options, other, long_term_incentive, restricted_stock, director_fees 
- email features: to_messages, email_address, from_poi_to_this_person, from_messages, from_this_person_to_poi, shared_receipt_with_poi
- poi label(A total of 18 entries was labelled as POI)

The goal of this project was simply to leverage the features above in order to mark an individual as a person of interest.

In order to detect outlier and explore the data, I first built a code to convert the dictionay to csv file and is available with the submission. Later, I explored all the values to find out which features had a lot of values and which did not. Immediately looking at the csv, I noticed that the number of data is only 146 and so individually scanning each name - I found the following data points redundant

- Total: Does not convey information pertaining to any individual, hence marked as outlier.
- THE TRAVEL AGENCY IN THE PARK: It does not represent any individual and hence was removed.
- LOCKHART EUGENE E: This record contained no useful data.

After cleaning the data only 143 records remained.

> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that doesn't come ready-made in the dataset--explain what feature you tried to make, and the rationale behind it. If you used an algorithm like a decision tree, please also give the feature importances of the features that you use.

To pick best features, I used the select k best features of scikit learn. After trying different K values, The K-best approach is an automated univariate feature selection algorithm, and in using it. I decided to go with two algorithms namely Gaussian NB and Logistic Regression. In, both cases I had to use different number of features to get the required result

The features I went ahead with are as follows(for logistic regression):

Precision: 0.51508	Recall: 0.21350

| Feature                 | Score??? | Valid |
| :---------------------- | -----: | ----: |
| exercised_stock_options | 24.815 |   102 |
| total_stock_value       | 24.183 |   126 |
| bonus                   | 20.792 |    82 |
| salary                  | 18.290 |    95 |
| deferred_income         | 11.458 |    49 |
| long_term_incentive     |  9.922 |    66 |
| restricted_stock        |  9.212 |   110 |
| total_payments          |  8.772 |   125 |
| shared_receipt_with_poi |  8.589 |    86 |
| loan_advances           |  7.184 |     3 |

The features I went ahead with Gaussian NB are as follows:

Precision: 0.38355	Recall: 0.31700

| Feature                 | Score??? | Valid |
| :---------------------- | -----: | ----: |
| exercised_stock_options | 24.815 |   102 |
| total_stock_value       | 24.183 |   126 |
| bonus                   | 20.792 |    82 |
| salary                  | 18.290 |    95 |
| deferred_income         | 11.458 |    49 |
| long_term_incentive     |  9.922 |    66 |
| restricted_stock        |  9.212 |   110 |
| total_payments          |  8.772 |   125 |

It is to be noted that different k-best values were tried from 5-15 to get the maximum result. Average recall values were found to be between 20-23 in the case of Logistic Regression and for the case of gaussian NB between 25-30. In order to inrease this value it was necessary to engineer new features. I went ahead by adding two new features.

- financial_aggregate: This was the combined sum of exercised_stock_options, salary, and total_stock_value. This captured both liquid and semi solid wealth an individual has. 
- email_interaction which was a ratio of the total number of emails to and from a POI to the total number of emails sent or received


### justifition

The k-best values for the new feature are as follows:

- financial_aggregate:15.971747347749965
- email_interaction:8.7727777300916792


| Classifier              | Precision | Recall    | Features  |
| ----------------------- | --------: | --------: | --------: |
| Logistic Regression     |     0.515 |     0.213 |        10 |
| Gaussian NB             |     0.383 |     0.317 |        8  |


New Features(the new engineered features):

| Classifier              | Precision | Recall    | Features  |
| ----------------------- | --------: | --------: | --------: |
| Logistic Regression     |     0.536 |     0.301 |        12 |
| Gaussian NB             |     0.340 |     0.365 |       10  |


I scaled all features using a min-max scaler. This ensures features are evenly balanced and overcomes the disparity due to the units of financial and email features.

The values above justify the usage of my engineered features with a significant increase on both recall and precision value.

> What algorithm did you end up using?  What other one(s) did you try?

After having performed various ml related projects during my undergraduate studies, a two class problem usually is best for logistic regression and K-means clustering. K-means clustering with PCA and mahalanobis distance provides a very fortified technique for two class detection. However, I went with Gaussian NB as my final algorithm due to the fact that it provided me with great results which were consistent throughout my test cases.

I tried several algorithms, with a K-means clustering algorithm also performing well. I also tested a support vector machine, a random forest classifier, and stochastic gradient descent. The best reults I got were from gaussian NB. I used this for the fact that I got a really nice consistency overall in all my validation techniques.

following were the parameters I tuned:
- Logistic regression: C (inverse regularization parameter), tol (tolerance), and class_weight (over/undersampling)
- K-means clustering: tol

K-means clustering was initialized with K (n_clusters) of 2 to represent POI and non-POI clusters. It performed well with the 8 set of features.  

Auto-weighting in the case of logistic regression caused a dip in precision and hence I decided to use evenly balanced features.

Gaussian NB perfomed extreemly well because this problem is a two class problem and GNB works very well in such cases

Parameter tuning is important because it helps improve the algorithm perform better. Each algorithm is associated with certain parameters that help wih the learning rate. Just like in the case of gradient descent, it would so happen that if the learning rate is too high we would miss the minima and if it is too small, we might not end up converging.

> What is validation, and what's a classic mistake you can make if you do it wrong? How did you validate your analysis?

Validation is performed to ensure that a machine learning algorithm generalizes well. The classic problem that can occur is over-fitting. This happens when we overfit the training data and perform really well in it due to which there is a considerable dip in the performance in the other two datsets(cross validation and testing dataset).

I validated my result using two techniques
- bootstrapping (cleaner.py)
- k fold method (cleaner.py)

> Give at least 2 evaluation metrics, and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm's performance.

I am using precision and recall as the evaluation metric. 
- Precision: The precision can be interpreted as the likelihood that a person who is identified as a POI is actually a true POI
- Recall:  Recall measures how likely it is that, given that there???s a POI in the test set, this identifier would flag him or her

I did not choose accuracy because with such a small dataset using it would mean that a simple heurestic with all data marked false would give an accuracy of more than 85%

*Validation 1 (Stratified K-folds, K=3) 

| Classifier              | Precision | Recall    | Features  |
| ----------------------- | --------: | --------: | --------: |
| Logistic Regression     |     0.416 |     0.25  |        12 |
| Gaussian NB             |     0.666 |     0.416 |       10  |



*Validation 2 randomised sampling(n=1000)*

| Classifier              | Precision | Recall    | Features  |
| ----------------------- | --------: | --------: | --------: |
| Logistic Regression     |     0.401 |     0.275 |        12 |
| Gaussian NB             |     0.386 |     0.344 |        10 |

*Validation 3 Udacity tester*

| Classifier              | Precision | Recall    | Features  |
| ----------------------- | --------: | --------: | --------: |
| Logistic Regression     |     0.536 |     0.301 |        12 |
| Gaussian NB             |     0.340 |     0.365 |       10  |

The level of consistency shown by Gaussian NB in all three validation technique is much more than logistic regression. Hence I went ahead with Gaussian NB.

Both algorithms do well inspite of the dataset being noisy.
Let us understand what the values are actually speaking to us. Now, in terms of precision and recall you would want a hgh recall in such a case. Simply, because you want to be suspecting people. Having a high precision would mean that we are looking for too strict of a conditions to flag an individual. Hence, recall plays a high role in such a case.

## Conclusion
The dataset was sparse and most algorithms will perform well only if given a decent number of data to learn. Had there been a large dataset random forest would also work very well. However, I am more interested in what a classic anomaly detection system would do in such a case. A simple combination of
- K-means
- PCA
- Mahalanobis distance measure 
would result in a very robust outlier detection algorithm and probably would have both a high recall and precision value.

### Resources and References

- [Introduction to Machine Learning (Udacity)](https://www.udacity.com/course/viewer#!/c-ud120-nd)
- [Machine Learning (Stanford/Coursera)](https://www.coursera.org/course/ml)
- [scikit-learn Documentation](http://scikit-learn.org/stable/documentation.html)
