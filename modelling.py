from sklearn.linear_model import LinearRegression
import pandas as pd
import evaluation
from sklearn.ensemble import RandomForestClassifier


def output_final_model(X_train, y_train, X_test, clf, submission_filename):
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	predictions = [max(x, 0) for x in predictions]
	submission = pd.DataFrame({"id": test["id"], "cost": predictions})
	submission.to_csv(submission_filename, index=False)

feature_names = ['annual_usage', 'min_order_quantity', 'bracket_pricing', 'quantity']

train = pd.read_csv('competition_data/train_set.csv').fillna("")
test = pd.read_csv('competition_data/test_set.csv').fillna("")

train.loc[train['bracket_pricing'] == 'Yes', 'bracket_pricing'] = 1
train.loc[train['bracket_pricing'] == 'No', 'bracket_pricing'] = 0

test.loc[test['bracket_pricing'] == 'Yes', 'bracket_pricing'] = 1
test.loc[test['bracket_pricing'] == 'No', 'bracket_pricing'] = 0

X_train = train[feature_names]
y_train = train['cost']

X_test = test[feature_names]

clf = LinearRegression()

evaluation.get_kfold_scores(X = X_train, y = y_train, n_folds = 8, clf = clf)
output_final_model(X_train = X_train, y_train = y_train, X_test = X_test, clf = clf, submission_filename = 'first_submission.csv')
