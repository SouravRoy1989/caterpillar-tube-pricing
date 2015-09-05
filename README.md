
##Kaggle Caterpillar Tube Pricing Competition

Scripts used for my submission the Kaggle Caterpillar Tube Pricing Competition (https://www.kaggle.com/c/caterpillar-tube-pricing). My final submission scored with a Root Mean Squared Logarithmic Error (RMSLE) of 0.213605 on the private leaderboard which gave me a rank of 50 out of 1323 competitors. 

## Running the Program
* Clone the repo: `git clone https://github.com/marknagelberg/caterpillar-tube-pricing.git`
* [Download the data](https://www.kaggle.com/c/caterpillar-tube-pricing/data) and save to a subfolder named 'competition_data'
* Run extraction.py to perform feature extraction (outputs pickled data frame for train and test sets)
* Run modelling.py to run model on the feature extracted data and output the submission.csv file used for submission.

##Overview of the competition
Caterpillar relies on a variety of suppliers to manufacture tube assemblies, each having their own unique pricing model. This competition provides detailed tube, component, and annual volume datasets, and the challenge is to predict the price a supplier will quote for a given tube assembly. 

##Overview of feature extraction and model
Some of the more useful features I developed in the model include the large number of component features spread across sevearal csv files, the length of the relationship with the supplier at the time of quote, the total number of suppliers for each tube assembly, the total number of quotes from suppliers, and the cost of "adjacent" tube assembly ids. I used a random forest model to aid in variable selection (only kept variables with an importance >= .0005) and then used these selected variables in an ensemble of xgboost models (forked from  Gilberto Titericz Junior, who was one of the members of the winning team).

##Creator

**Mark Nagelberg**

* <https://twitter.com/MarkNagelberg>
* <https://github.com/marknagelberg>
* <https://kaggle.com/marknagelberg>




