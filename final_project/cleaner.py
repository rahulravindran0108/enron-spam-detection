import csv
from sklearn.feature_selection import SelectKBest
import sys
from numpy import mean
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

def remove_keys(dict_object, keys):
    """ removes a list of keys from a dict object """
    for key in keys:
        dict_object.pop(key, 0)

def create_csv(data_filename,data_dict):
	writer = csv.writer(open(data_filename, 'wb'))
	actual_keys = ['name','salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'email_address', 'from_poi_to_this_person']
	writer.writerow(actual_keys)
	for key, value in data_dict.items():
		row = []

    	row.append(key)
    	for key2,value2 in value.items():
   			row.append(value2)
   	writer.writerow(row)

def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1} - {2}\n".format(k, k_best_features.keys(),k_best_features.values())
    return k_best_features

def add_email_interaction(data_dict, features_list):
    """ mutates data dict to add proportion of email interaction with pois """
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] +\
                             person['from_messages']
            poi_messages = person['from_poi_to_this_person'] +\
                           person['from_this_person_to_poi']
            person['email_interaction'] = float(poi_messages) / total_messages
        else:
            person['email_interaction'] = 'NaN'
    features_list += ['email_interaction']

def add_financial_aggregate(data_dict, features_list):
    """ mutates data dict to add aggregate values from stocks and salary """
    fields = ['total_stock_value', 'exercised_stock_options', 'salary']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            person['financial_aggregate'] = sum([person[field] for field in fields])
        else:
            person['financial_aggregate'] = 'NaN'
    features_list += ['financial_aggregate']

def stratified_k_fold(clf,features,labels):
    skf = StratifiedKFold( labels, n_folds=3 )
    precisions = []
    recalls = []
    for train_idx, test_idx in skf:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)


        ### for each fold, print some metrics
        print
        print "precision score: ", precision_score( labels_test, pred )
        print "recall score: ", recall_score( labels_test, pred )

        precisions.append( precision_score(labels_test, pred) )
        recalls.append( recall_score(labels_test, pred) )

    ### aggregate precision and recall over all folds
    print "average precision: ", sum(precisions)/2.
    print "average recall: ", sum(recalls)/2.


def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print "done.\n"
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall)