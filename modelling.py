import pandas as pd
import evaluation
from sklearn.ensemble import RandomForestRegressor
import csv
from sklearn.preprocessing import Imputer
from sklearn import ensemble, preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import csv
import xgboost as xgb
import pickle

def output_final_model(X_train, y_train, X_test, clf, submission_filename, feature_names):
	clf.fit(X_train[feature_names], y_train)

	#Print importances to csv
	with open('importances.csv', 'wb') as f:
		writer = csv.writer(f)
		importance_count = 0
		features_to_return = []
		for idx, col in enumerate(X_train[feature_names]):
			print col + ':' + str(clf.feature_importances_[importance_count])
			writer.writerow([col, clf.feature_importances_[importance_count]])

			
			if clf.feature_importances_[importance_count] >= .0005:
				features_to_return.append(col)
			importance_count += 1


	train_predictions = clf.predict(X_train[feature_names])
	predictions = clf.predict(X_test[feature_names])
	predictions = np.exp(predictions) - 1
	submission = pd.DataFrame({"id": test["id"], "cost": predictions})
	submission.to_csv(submission_filename, index=False)

	train_predictions = pd.DataFrame({"predicted_cost": train_predictions})
	y_train = pd.DataFrame({"cost": y_train})
	extracted_train = pd.concat(objs = [X_train, y_train, train_predictions], axis = 1)
	extracted_train.to_csv('extracted_data_with_predictions/extracted_train.csv', index = False)
	predictions = pd.DataFrame({"predicted_cost": predictions})
	extracted_test = pd.concat(objs = [X_test, predictions], axis = 1)
	extracted_test.to_csv('extracted_data_with_predictions/extracted_test.csv', index = False)

	return features_to_return
def run_xgboost(X_train, y_train, X_test, clf, submission_filename, feature_names):

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

	X_train = X_train[feature_names].astype(float)
	X_test = X_test[feature_names].astype(float)

	X_train = np.array(X_train)
	X_test = np.array(X_test)

	xgtrain = xgb.DMatrix(X_train, label=y_train.values)
	xgtest = xgb.DMatrix(X_test)
	
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
	xgtrain = xgb.DMatrix(X_train, label=label_log.values)
	xgtest = xgb.DMatrix(X_test)

	print('power 1/16 4000')

	num_rounds = 4000
	model = xgb.train(plst, xgtrain, num_rounds)
	preds3 = model.predict(xgtest)

	#for loop in range(2):
	#    model = xgb.train(plst, xgtrain, num_rounds)
	#    preds1 = preds1 + model.predict(xgtest)
	preds = 0.4*np.expm1(preds4)+.1*np.expm1(preds1)+0.1*np.expm1(preds2)+0.4*np.power(preds3,16)
	#preds = (0.58*np.expm1( (preds1+preds2+preds4)/3))+(0.42*np.power(preds3,16))

	preds = pd.DataFrame({"id": test["id"], "cost": preds})
	preds.to_csv('xgboost_submission.csv', index=False)



if __name__ == '__main__':

	train = pickle.load(open('extracted_train.pkl'))
	test = pickle.load(open('extracted_test.pkl'))

    #Gather features to use - don't want to use "cost" nor any feature with an object data type.
	feature_names = []
	for feature in train:
		if train[feature].dtype != 'object' and feature != 'cost':
			feature_names.append(feature)

	y_train = train['cost']
	train = train.drop(['cost'], 1)

	#X_test = test[feature_names]

	clf = RandomForestRegressor(n_estimators = 20)

	#evaluation.get_kfold_scores(X = train, y = y_train, n_folds = 8, clf = clf, feature_names = feature_names)
	best_features = output_final_model(X_train = train, y_train = y_train, X_test = test, clf = clf, submission_filename = 'submission.csv', feature_names = feature_names)
	print best_features
	run_xgboost(X_train = train, y_train = y_train, X_test = test, clf = clf, submission_filename = 'submission.csv', feature_names = best_features)