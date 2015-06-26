#!/usr/bin/python
import csv
from copy import copy
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from udacity_tester import test_classifier

import cleaner
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

target_label = 'poi'

features_list_email =  ['from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages'] 

features_list_financial = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
 ]

final_features = [target_label]+features_list_email+features_list_financial
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )







#cleaner.create_csv('data.csv',data_dict)

### Task 2: Remove outliers
outlier_keys = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
cleaner.remove_keys(data_dict,outlier_keys)


def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1

my_dataset = copy(data_dict)
my_features = copy(final_features)

cleaner.add_financial_aggregate(my_dataset, my_features)


#get k best features for logisctic regression
num_features = 10
best_features = cleaner.get_k_best(my_dataset, my_features, num_features)
my_features = [target_label] + best_features.keys()+['fraction_from_poi_email','fraction_to_poi_email']


#get k best features for K-means
#num_features = 6
#best_features = cleaner.get_k_best(my_dataset, my_features, num_features)
#my_features = [target_label] + best_features.keys()+['fraction_from_poi_email','fraction_to_poi_email']



# print features
print "{0} selected features: {1}\n".format(len(my_features) - 1, my_features[1:])



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

# scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


from sklearn.linear_model import LogisticRegression
l_clf = None

# brute-force parameter optimizer; uncomment to run
# TODO: use GridSearchCV
# k = 0
# best_combo = None
# max_exponent = 21
# for i in range(0, max_exponent, 3):
#     for j in range(0, max_exponent, 3):
#         print "i: {0}, j: {1}".format(i, j)
#         l_clf = LogisticRegression(C=10**i, tol=10**-j, class_weight='auto')
#         results = cleaner.evaluate_clf(l_clf, features, labels)
#         if sum(results) > k:
#             k = sum(results)
#             best_combo = (i, j)
#      l_clf = LogisticRegression(C=10**i, tol=10**-j)

if not l_clf:
    l_clf = LogisticRegression(C=10**18, tol=10**-21)

from sklearn.naive_bayes import GaussianNB
gauss_clf = GaussianNB()    

### K-means Clustering
from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.01)


#Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

#Stochastic Gradient Descent - Logistic Regression
from sklearn.linear_model import SGDClassifier
g_clf = SGDClassifier(loss='log')


### Selected Classifiers Evaluation

pickle.dump(l_clf, open("my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("my_dataset.pkl", "w"))
pickle.dump(my_features, open("my_feature_list.pkl", "w"))

#stratified k fold
#cleaner.stratified_k_fold(l_clf,features,labels)

#randomized Sampling
print "First Validation Technique:Randomized Sampling"
cleaner.evaluate_clf(l_clf, features, labels)

print "Second Validation Tehnique: Stratified K-folds"
cleaner.stratified_k_fold(l_clf,features,labels)

