import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import csv
import numpy as np
import csv
import xgboost as xgb
import pickle

'''
Runs xgboost models on the feature extracted data. First a random forest
regressor is run to get importances of the  variables and select
which to use in the final ensembled xgboost model. Final predictions
extracted to submission.csv.
'''

if __name__ == '__main__':

	#Load the data extracted using extraction.py
	train = pickle.load(open('extracted_train.pkl'))
	test = pickle.load(open('extracted_test.pkl'))

    #Do not want to use features in train and test that are the target
    #variable ("cost") or have an 'object' data type.
	feature_names = []
	for feature in train:
		if train[feature].dtype != 'object' and feature != 'cost':
			feature_names.append(feature)

	y_train = train['cost']
	train = train.drop(['cost'], 1)

	clf = RandomForestRegressor(n_estimators = 20)
	clf.fit(train[feature_names], y_train)

	#Only keep features with an importance >= .0005 from random forest classifier
	importance_count = 0
	best_features = []
	for idx, col in enumerate(train[feature_names]):
		if clf.feature_importances_[importance_count] >= .0005:
			best_features.append(col)
		importance_count += 1

	#Run several xgboost models using only the features with importance >= .0005 in 
	#random forest model above. The code below is forked from  Gilberto Titericz Junior.

	params = {}
	params["objective"] = "reg:linear"
	params["eta"] = 0.02
	params["min_child_weight"] = 6
	params["subsample"] = 0.7
	params["colsample_bytree"] = 0.6
	params["scale_pos_weight"] = 0.8
	params["silent"] = 1
	params["max_depth"] = 8
	params["max_delta_step"]=2

	plst = list(params.items())

	train = train[best_features].astype(float)
	test_xgboost = test[best_features].astype(float)

	train = np.array(train)
	test_xgboost = np.array(test_xgboost)

	xgtrain = xgb.DMatrix(train, label=y_train.values)
	xgtest = xgb.DMatrix(test_xgboost)

	print('2000')

	num_rounds = 2000
	model = xgb.train(plst, xgtrain, num_rounds)
	preds1 = model.predict(xgtest)

	print('3000')

	num_rounds = 3000
	model = xgb.train(plst, xgtrain, num_rounds)
	preds2 = model.predict(xgtest)

	print('4000')

	num_rounds = 4000
	model = xgb.train(plst, xgtrain, num_rounds)
	preds4 = model.predict(xgtest)

	y_train = np.exp(y_train) - 1
	label_log = np.power(y_train, 1.0/16.0)
	xgtrain = xgb.DMatrix(train, label=label_log.values)
	xgtest = xgb.DMatrix(test_xgboost)

	print('power 1/16 4000')

	num_rounds = 4000
	model = xgb.train(plst, xgtrain, num_rounds)
	preds3 = model.predict(xgtest)

	preds = 0.4*np.expm1(preds4)+.1*np.expm1(preds1)+0.1*np.expm1(preds2)+0.4*np.power(preds3,16)

	preds = pd.DataFrame({"id": test["id"], "cost": preds})
	preds.to_csv('submission.csv', index=False)