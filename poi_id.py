#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import random
import numpy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot

### Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi',
    'salary', 
    'deferral_payments', 
    'total_payments', 
    'exercised_stock_options', 
    'bonus', 
    'restricted_stock', 
    'shared_receipt_with_poi', 
    'restricted_stock_deferred', 
    'total_stock_value', 
    'expenses', 
    'other', 
    'from_this_person_to_poi', 
    'director_fees', 
    'deferred_income', 
    'long_term_incentive',
    'from_poi_to_this_person',
    'to_messages', 
    'loan_advances', 
    'from_messages'	] 

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers
features = ["salary", "bonus"]
data = featureFormat(data_dict,features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary,bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()

# we found one outlier which is actually the total salary
# remove it from the data
data_dict.pop("TOTAL",0)
data = featureFormat(data_dict,features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()

### Task 3: Do feature selection,feature scaling and add any new feature(s) if needed

#between regression and classfication selectors
#choose classifier selectors since poi/NON-poi

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
import numpy
"""
data = featureFormat(data_dict,features_list)
labels, features = targetFeatureSplit(data)
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#using f_classif
X_new = SelectKBest (f_classif, k='all')
X_new=X_new.fit (features_train,labels_train)
import json
print "Feature Ranking using selectK f_classif:"
#print(json.dumps(features_list,indent=4))
print(json.dumps(list(X_new.scores_),indent=4))

#using mutual_info_classif
X_new = SelectKBest (mutual_info_classif, k='all')
X_new=X_new.fit (features_train,labels_train )
print "Feature Ranking using selectK mutual_f_classif:"
#print(json.dumps (features_list,indent=4))
print(json.dumps (list (X_new.scores_),indent=4))

#using Decision Tree importance
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit( features_train, labels_train)
 
accuracy = dt.score(features_test, labels_test)
print("\n Accuracy of Model: %0.3F %%" % (accuracy * 100))

# Find the top features in the decision tree
importances = dt.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print "Feature Ranking using DTree:"
for i in range (len(importances)):
    print "{}...{}".format(importances[indices[i]],features_list[i+1])
"""
# according to the decision tree we get
features_list = ['poi',
    'salary', 
    'deferral_payments', 
    'total_payments', 
    'exercised_stock_options', 
    'bonus', 
    'restricted_stock', ]

# which gives accuracy ( 80% ) but recall and precision are low ( < 0.4 )

# Manually check better features observing all three rankings
features_list = ['poi',
    'salary', 
    'exercised_stock_options', 
    'bonus', 
    'total_payments', 
    'from_this_person_to_poi', 
    'from_poi_to_this_person',
    'to_messages',
    'from_messages'	] 

# which gives accuracy ( 80% ) but recall and precision are higher ( > 0.5 )

# For feature scaling, normalize 'from_this_person_to_poi' and 'from_poi_to_this_person'
# dividing them with 'to_messages' and 'from_messages'.

for key in data_dict:
    
    num = data_dict[key]['from_this_person_to_poi']
    den = data_dict[key]['from_messages']
    if num == 'NaN' or den=='NaN':
        continue
    num=int(num)
    den=int(den)
    data_dict[key]['from_this_person_to_poi'] = str(num/den)
    
for key in data_dict:
    
    num = data_dict[key]['from_poi_to_this_person']
    den = data_dict[key]['to_messages']
    if num == 'NaN' or den=='NaN':
        continue
    num=int(num)
    den=int(den)
    data_dict[key]['from_poi_to_this_person'] = str(num/den)
    	
# Update features list to:
features_list = [
    'poi',
    'salary', 
    'exercised_stock_options', 
    'from_this_person_to_poi', 
    'from_poi_to_this_person',
    'shared_receipt_with_poi' 
		] 
		
# Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

from sklearn.tree import DecisionTreeClassifier
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using the testing script.
clf=DecisionTreeClassifier(min_samples_split=5)

### Task 6:Dump  classifier, dataset, and features_list so anyone can check your results.
# run tester.py after poi_id.py to check results
import pickle

f = open('my_classifier.pkl', 'w')
pickle.dump(clf, f)
f.close()

f = open('my_dataset.pkl', 'w')
pickle.dump(my_dataset, f)
f.close()

f = open('my_feature_list.pkl', 'w')
pickle.dump(features_list, f)
f.close()

dump_classifier_and_data(clf, my_dataset, features_list)
