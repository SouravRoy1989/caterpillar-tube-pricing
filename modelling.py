from sklearn.linear_model import LinearRegression
import pandas as pd
import evaluation
from sklearn.ensemble import RandomForestRegressor

def extract(data):
	data.loc[data['bracket_pricing'] == 'Yes', 'bracket_pricing'] = 1
	data.loc[data['bracket_pricing'] == 'No', 'bracket_pricing'] = 0

	tube_data = pd.read_csv('competition_data/tube.csv').fillna("")
	data_merged = pd.merge(left = data, right = tube_data, how='inner', on='tube_assembly_id')

	data_merged['quote_year'] = [x.split('-')[0] for x in data_merged['quote_date']]
	data_merged['quote_month'] = [x.split('-')[1] for x in data_merged['quote_date']]

	data_merged.loc[data_merged['end_a_1x'] == 'Y', 'end_a_1x'] = 1
	data_merged.loc[data_merged['end_a_1x'] == 'N', 'end_a_1x'] = 0
	data_merged.loc[data_merged['end_a_2x'] == 'Y', 'end_a_2x'] = 1
	data_merged.loc[data_merged['end_a_2x'] == 'N', 'end_a_2x'] = 0

	data_merged.loc[data_merged['end_x_1x'] == 'Y', 'end_x_1x'] = 1
	data_merged.loc[data_merged['end_x_1x'] == 'N', 'end_x_1x'] = 0
	data_merged.loc[data_merged['end_x_2x'] == 'Y', 'end_x_2x'] = 1
	data_merged.loc[data_merged['end_x_2x'] == 'N', 'end_x_2x'] = 0

	return data_merged

def output_final_model(X_train, y_train, X_test, clf, submission_filename):
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	predictions = [max(x, 0) for x in predictions]
	submission = pd.DataFrame({"id": test["id"], "cost": predictions})
	submission.to_csv(submission_filename, index=False)


train = pd.read_csv('competition_data/train_set.csv').fillna("")
test = pd.read_csv('competition_data/test_set.csv').fillna("")

train = extract(train)
test = extract(test)


feature_names = train.columns.values
feature_names = [x for x in feature_names if x not in ['tube_assembly_id', 'supplier', 'quote_date', 'material_id', 'end_a', 'end_x', 'cost']]


X_train = train[feature_names]
y_train = train['cost']

X_test = test[feature_names]

clf = RandomForestRegressor(n_estimators = 50)

evaluation.get_kfold_scores(X = X_train, y = y_train, n_folds = 8, clf = clf)
output_final_model(X_train = X_train, y_train = y_train, X_test = X_test, clf = clf, submission_filename = 'submission.csv')
