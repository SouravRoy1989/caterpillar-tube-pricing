from sklearn.linear_model import LinearRegression
import pandas as pd
import evaluation
from sklearn.ensemble import RandomForestRegressor
import csv
from sklearn.preprocessing import Imputer

def extract(data):
	data.loc[data['bracket_pricing'] == 'Yes', 'bracket_pricing'] = 1
	data.loc[data['bracket_pricing'] == 'No', 'bracket_pricing'] = 0

	data['bracket_pricing'] = data['bracket_pricing'].astype(int)

	tube_data = pd.read_csv('competition_data/tube.csv').fillna("")
	data_merged = pd.merge(left = data, right = tube_data, how='inner', on='tube_assembly_id')
	bill_of_materials = pd.read_csv('competition_data/bill_of_materials.csv')
	data_merged = pd.merge(left = data_merged, right = bill_of_materials, how='inner', on='tube_assembly_id')

	data_merged['quote_year'] = [int(x.split('-')[0]) for x in data_merged['quote_date']]
	data_merged['quote_month'] = [int(x.split('-')[1]) for x in data_merged['quote_date']]

	data_merged.loc[data_merged['end_a_1x'] == 'Y', 'end_a_1x'] = 1
	data_merged.loc[data_merged['end_a_1x'] == 'N', 'end_a_1x'] = 0
	data_merged.loc[data_merged['end_a_2x'] == 'Y', 'end_a_2x'] = 1
	data_merged.loc[data_merged['end_a_2x'] == 'N', 'end_a_2x'] = 0

	data_merged.loc[data_merged['end_x_1x'] == 'Y', 'end_x_1x'] = 1
	data_merged.loc[data_merged['end_x_1x'] == 'N', 'end_x_1x'] = 0
	data_merged.loc[data_merged['end_x_2x'] == 'Y', 'end_x_2x'] = 1
	data_merged.loc[data_merged['end_x_2x'] == 'N', 'end_x_2x'] = 0

	data_merged['end_a_1x'] = data_merged['end_a_1x'].astype(int)
	data_merged['end_a_2x'] = data_merged['end_a_2x'].astype(int)
	data_merged['end_x_1x'] = data_merged['end_x_1x'].astype(int)
	data_merged['end_x_2x'] = data_merged['end_x_2x'].astype(int)

	end_form = pd.read_csv('competition_data/tube_end_form.csv')

	data_merged.loc[data_merged['end_a'] == "NONE", 'end_a_forming'] = -1
	data_merged.loc[data_merged['end_x'] == "NONE", 'end_x_forming'] = -1

	for idx,row in end_form.iterrows():
		if row['forming'] == 'Yes':
			end_forming_value = 1
		if row['forming'] == 'No':
			end_forming_value = 0

		data_merged.loc[data_merged['end_a'] == row['end_form_id'], 'end_a_forming'] = end_forming_value
		data_merged.loc[data_merged['end_x'] == row['end_form_id'], 'end_x_forming'] = end_forming_value

	quantity_vars = [x for x in data_merged.columns.values if x[:9] == 'quantity_']
	#data_merged[quantity_vars] = data_merged[quantity_vars].fillna(0, axis = 1)
	#print data_merged[quantity_vars].sum(axis = 1)
	data_merged['total_quantity_components'] = data_merged[quantity_vars].fillna(0, axis = 1).sum(axis = 1)
	data_merged = data_merged.fillna("")
	return data_merged

def output_final_model(X_train, y_train, X_test, clf, submission_filename):
	clf.fit(X_train, y_train)
	for idx, col in enumerate(X_train):
		print col + ':' + str(clf.feature_importances_[idx])
	predictions = clf.predict(X_test)
	predictions = [max(x, 0) for x in predictions]
	submission = pd.DataFrame({"id": test["id"], "cost": predictions})
	submission.to_csv(submission_filename, index=False)


train = pd.read_csv('competition_data/train_set.csv')
test = pd.read_csv('competition_data/test_set.csv')

train = extract(train)
test = extract(test)

print train.describe()

train.to_csv('training_extracted.csv', index=False)

#Gather features to use - don't want to use "cost" nor any feature with an object data type.
feature_names = []
for feature in train:
	if train[feature].dtype != 'object' and feature != 'cost':
		feature_names.append(feature)

X_train = train[feature_names]
y_train = train['cost']

X_test = test[feature_names]

clf = RandomForestRegressor(n_estimators = 20)

evaluation.get_kfold_scores(X = X_train, y = y_train, n_folds = 8, clf = clf)
output_final_model(X_train = X_train, y_train = y_train, X_test = X_test, clf = clf, submission_filename = 'submission.csv')
