import pandas as pd
import evaluation
from sklearn.ensemble import RandomForestRegressor
import csv
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def extract(data):
	
	#tube_data = pd.read_csv('competition_data/tube.csv').fillna("")
	tube_data = pd.read_csv('competition_data/tube.csv').fillna("")
	data_merged = pd.merge(left = data, right = tube_data, how='inner', on='tube_assembly_id')
	bill_of_materials = pd.read_csv('competition_data/bill_of_materials.csv')
	data_merged = pd.merge(left = data_merged, right = bill_of_materials, how='inner', on='tube_assembly_id')

	data_merged['bracket_pricing'] = LabelEncoder().fit_transform(data_merged['bracket_pricing'])

	data_merged['quote_year'] = [int(x.split('-')[0]) for x in data_merged['quote_date']]
	data_merged['quote_month'] = [int(x.split('-')[1]) for x in data_merged['quote_date']]

	data_merged['end_a_1x'] = LabelEncoder().fit_transform(data_merged['end_a_1x'])
	data_merged['end_a_2x'] = LabelEncoder().fit_transform(data_merged['end_a_2x'])
	data_merged['end_x_1x'] = LabelEncoder().fit_transform(data_merged['end_x_1x'])
	data_merged['end_x_2x'] = LabelEncoder().fit_transform(data_merged['end_x_2x'])

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
	data_merged[quantity_vars] = data_merged[quantity_vars].fillna(0, axis = 1)
	data_merged['total_quantity_components'] = data_merged[quantity_vars].sum(axis = 1)
	data_merged = data_merged.fillna("")

	type_component = pd.read_csv('competition_data/type_component.csv')
	for component_type_id in type_component['component_type_id']:
		data_merged[component_type_id] = 0
	data_merged['OTHER'] = 0

	with open('competition_data/components.csv', 'rb') as csvfilereader:
		with open('competition_data/components_v2.csv', 'wb') as csvfilewriter:
			reader = csv.reader(csvfilereader)
			writer = csv.writer(csvfilewriter)
			for row in reader:
				if len(row) == 4:
					del row[2]
					writer.writerow(row)
				else:
					writer.writerow(row)


	component = pd.read_csv('competition_data/components_v2.csv')
	component_mapping = {}
	for idx, row in component.iterrows():
		component_mapping[row['component_id']] = row['component_type_id']
	component_vars = ['component_id_' + str(x) for x in range(1,9)]
	for idx, row in data_merged.iterrows():
		for var in component_vars:
			if row[var] in component_mapping:
				data_merged.set_value(idx, component_mapping[row[var]], 1)

	return data_merged

def output_final_model(X_train, y_train, X_test, clf, submission_filename):
	clf.fit(X_train, y_train)
	for idx, col in enumerate(X_train):
		print col + ':' + str(clf.feature_importances_[idx])
	predictions = clf.predict(X_test)
	#predictions = [max(x, 0) for x in predictions]
	predictions = np.exp(predictions) - 1
	submission = pd.DataFrame({"id": test["id"], "cost": predictions})
	submission.to_csv(submission_filename, index=False)


if __name__ == '__main__':
	train = pd.read_csv('competition_data/train_set.csv')
	test = pd.read_csv('competition_data/test_set.csv')

	train['cost'] = np.log(train['cost'] + 1)
	train = extract(train)
	test = extract(test)

	#Additional processing of component data and adding it to train/test.
	comp_files = ['comp_adaptor.csv', 'comp_boss.csv', 'comp_elbow.csv', 'comp_float.csv', 'comp_hfl.csv', 'comp_nut.csv', 'comp_other.csv', 'comp_sleeve.csv', 'comp_straight.csv', 'comp_tee.csv', 'comp_threaded.csv']
	component_names = ['component_id_' + str(x) for x in range(1, 9)]
	concat_final_output = []
	for comp_filename in comp_files:
	    comp_file = pd.read_csv('competition_data/' + comp_filename)
	    data_frames_to_concat = []
	    for name in component_names:
	        to_merge = train[[name, 'cost']]
	        data_merged = pd.merge(left = comp_file, right = to_merge, how='left', left_on='component_id', right_on = name)
	        data_merged = data_merged.drop(name, 1)
	        data_frames_to_concat.append(data_merged)
	    final = pd.concat(data_frames_to_concat, ignore_index = True)

	    #Fill in NAs and make sure columns are appropriate data types
	    for column in final:
	        if column[:12] == 'component_id':
	            continue
	        elif column == 'cost':
	            continue
	        elif final[column].dtype == 'object':
	            final[column] = LabelEncoder().fit_transform(final[column])
	        elif final[column].dtype == 'float64':
	            final[column] = final[column].fillna(-1)

	    y_train = final.loc[final['cost'].notnull(),'cost']
	    y_train = np.log(y_train + 1)
	    X_train = final.loc[final['cost'].notnull()]
	    X_train = X_train.drop('cost', 1)
	    X_test = final.loc[final['cost'].isnull()]
	    X_test = X_test.drop('cost', 1)

	    clf = RandomForestRegressor(n_estimators = 20)

	    #Want to include all columns except 'component_id'
	    cols = [x for x in X_test.columns.values if x != 'component_id']

	    clf.fit(X_train[cols], y_train)
	    predictions = clf.predict(X_test[cols])
	    predictions = np.exp(predictions) - 1
	    y_train = np.exp(y_train) - 1
	    train_output = pd.concat([X_train, y_train], axis=1)
	    X_test.reindex(index = range(len(X_test)))
	    X_test['cost'] = predictions
	    final_output = pd.concat([X_test, train_output])
	    final_output.to_csv('component_data/' + comp_filename + '_concat_with_predictions.csv', index=False)
	    concat_final_output.append(final_output.groupby(final_output['component_id']).mean()['cost'])

	output = pd.concat(concat_final_output)
	output.to_csv('component_data/final.csv')

	comp_dict = {}
	for idx, row in output.iteritems():
	    comp_dict[idx] = row

	for idx, row in train.iterrows():
	    for name in component_names:
	        if row[name] in comp_dict:
	            train.set_value(idx, name + '_comp_cost', comp_dict[row[name]])
	        else:
	            train.set_value(idx, name + '_comp_cost', 0)
	            
	train.to_csv('training_extracted.csv', index=False)

	for idx, row in test.iterrows():
	    for name in component_names:
	        if row[name] in comp_dict:
	            test.set_value(idx, name + '_comp_cost', comp_dict[row[name]])
	        else:
	            test.set_value(idx, name + '_comp_cost', 0)


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
