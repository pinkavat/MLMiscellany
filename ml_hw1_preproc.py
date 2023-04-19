"""
process.py

    Data-crunching pandas/numpy/scikit/matplotlib code for CSS 581 HW 1

    written January 2023 by Thomas Pinkava
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate, cross_val_predict



# Parse data file
data = pd.read_csv('data.csv')

# Select target feature and encode categorical (should really be part of pipeline, no?)
targetEncoder = OrdinalEncoder()
data.NoShow = targetEncoder.fit_transform(data[["NoShow"]])
y = data.NoShow

# Note: y is dropped from data by the preprocessor pipeline, no need to redefine an 'X'


# ========== EXPLORATORY DATAVIS ==========

# Exploratory code (preserved for future use)

# NaN Assessment (No missing values whatsoever)
#for column in X.columns:
#	print(column + ":\t" + str(X[column].isna().sum() / len(X[column])))


# Categorical-preds-boolean plotter (for Handicap and Neighborhood)
# (Also used for the booleans, as well; why not?)
def cat_preds_bool_plot(df, attribute_name, target_name, filename=None):

    # Use reindexing to ensure all attribute fields are present
    true_bar_data = df[df[target_name] == 1.0][attribute_name].value_counts().reindex(data[attribute_name].unique(),fill_value = 0).sort_values(ascending = False)
    false_bar_data = df[df[target_name] == 0.0][attribute_name].value_counts().reindex(data[attribute_name].unique(),fill_value = 0).sort_values(ascending = False)

    # Generate stacked bar graphs
    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, left=0.1, right=0.9, bottom=0.1, top=0.9,
                      hspace=0.05)
    false_plot = fig.add_subplot(grid[1, 0])
    true_plot = fig.add_subplot(grid[0, 0], sharex = false_plot)
    true_plot.tick_params(axis="x", labelbottom = False)
    
    # Make strings of the labels so that the lib doesn't interpolate tickmarks
    true_plot.bar([str(x) for x in list(true_bar_data.index)], list(true_bar_data.values), color="blue") 
    false_plot.bar([str(x) for x in list(false_bar_data.index)], list(false_bar_data.values), color="red")
    
    #true_plot.bar(list(true_bar_data.index), list(true_bar_data.values), color="blue") 
    #false_plot.bar(list(false_bar_data.index), list(false_bar_data.values), color="red")
    
    true_plot.set_title("Distribution of " + attribute_name + " with respect to " + target_name)
    true_plot.set_ylabel(target_name + " = 1.0")
    false_plot.set_ylabel(target_name + " = 0.0")
    false_plot.set_xlabel(attribute_name)

    # Manually change font size here if need be
    #false_plot.set_xticklabels(false_plot.get_xticklabels(), rotation=90, ha='right', size=4)
    false_plot.set_xticklabels(false_plot.get_xticklabels(), rotation=0, ha='right', size=10)

    # Save to image or display
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight')


# Generate charts
#cat_preds_bool_plot(data, "Neighbourhood", "NoShow", "fig_neighbourhood.png")
#cat_preds_bool_plot(data, "Handicap", "NoShow", "fig_handicap.png")
#cat_preds_bool_plot(data, "Gender", "NoShow", "fig_gender.png")
#cat_preds_bool_plot(data, "Scholarship", "NoShow", "fig_scholarship.png")
#cat_preds_bool_plot(data, "Hypertension", "NoShow", "fig_hypertension.png")
#cat_preds_bool_plot(data, "Diabetes", "NoShow", "fig_diabetes.png")
#cat_preds_bool_plot(data, "Alcoholism", "NoShow", "fig_alcoholism.png")
#cat_preds_bool_plot(data, "SMSReceived", "NoShow", "fig_SMSReceived.png")

#cat_preds_bool_plot(data, "Age", "NoShow", "fig_age.png")



# ========== PREPROCESSING PIPELINE ==========

# Can't find a date parser in scikit-learn. Fascinating.
class ParseDate(BaseEstimator, TransformerMixin):
    #def __init__(self, features):
    #    self.features = features   TODO bespoke is not scalable!

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #for feature in self.features:
            #X[feature] = pd.to_datetime(X[feature]).apply(lambda x: x.timestamp())
            #X = X.drop(feature, axis=1) # Remove original

        # Extract Month, Day of Week, and Distance between Scheduled and Appointment bespokely
        X["ScheduledDay"] = pd.to_datetime(X["ScheduledDay"])
        X["ScheduledMonth"] = X["ScheduledDay"].dt.month
        X["AppointmentDay"] = pd.to_datetime(X["AppointmentDay"])
        X["AppointmentMonth"] = X["AppointmentDay"].dt.month
        X["ScheduleToAppointmentDays"] = (X["AppointmentDay"].apply(lambda t: t.timestamp()) -
            X["ScheduledDay"].apply(lambda t : t.timestamp())).apply(lambda t: floor(abs(t) / 86400))

        X["AppointmentDay"] = X["AppointmentDay"].dt.dayofweek  # Overwrite AppointmentDay
        X["ScheduledDay"] = X["ScheduledDay"].dt.dayofweek  # Overwrite ScheduledDay

        return X


# First pass featureset
#numeric_features = ["Age","Scholarship","Hypertension","Diabetes",
#    "Alcoholism","Handicap","SMSReceived"]
#date_features = ["ScheduledDay","AppointmentDay"]
#ordinal_cat_features = ["Gender"]
#onehot_cat_features = ["Neighbourhood"]

# Second pass featureset (removing most features -- even the date! improves performance)
numeric_features = ["Age","SMSReceived"]
date_features = ["ScheduledDay","AppointmentDay"]
ordinal_cat_features = []
onehot_cat_features = ["Neighbourhood"]


date_preproc_pipeline = Pipeline(steps = [
    ('date_parser', ParseDate()),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers = [
        ("num_preproc", StandardScaler(), numeric_features),    # Imputation superfluous, there are no missing values
        #("date_preproc", date_preproc_pipeline, date_features),#TODO note: removing the date improves performance here. Interesting.
        ("cat_ord_preproc", OrdinalEncoder(), ordinal_cat_features),
        ("cat_oht_preproc", OneHotEncoder(handle_unknown = "ignore"), onehot_cat_features)
    ], remainder = 'drop'   # Drop the target feature implicitly
                            # Note: implicitly dropping Patient ID and Appt ID too.
)


# Transform date and generate plots for it
#temp_date_transformer = ParseDate()
#transformed_dates = temp_date_transformer.fit_transform(data)
#cat_preds_bool_plot(data, "ScheduledMonth", "NoShow", "fig_scheduledmonth.png")
#cat_preds_bool_plot(data, "ScheduledDay", "NoShow", "fig_scheduledday.png")
#cat_preds_bool_plot(data, "AppointmentMonth", "NoShow", "fig_apptmonth.png")
#cat_preds_bool_plot(data, "AppointmentDay", "NoShow", "fig_apptday.png")
#cat_preds_bool_plot(data, "ScheduleToAppointmentDays", "NoShow", "fig_schedapptdiff.png")



# ========== CLASSIFIERS ==========

# Logistic Regression (misnomer)
model_logistic = LogisticRegression()
pipeline_logistic = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("model", model_logistic)
])

# Decision tree classifier
model_tree = DecisionTreeClassifier()
pipeline_tree = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("model", model_tree)
])

# Random forest classifier
model_forest = RandomForestClassifier()
pipeline_forest = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("model", model_forest)
])


# Wrapper for the cross-validation, printing the requisite statistics
def evaluate_classifier(pipeline):
    metrics = ['accuracy','precision', 'recall', 'f1', 'roc_auc']
    scores = cross_validate(pipeline, data, y, cv = 10, scoring = metrics)

    # Average and print scores
    for metric in metrics:
        avg_metric = sum(scores['test_'+metric]) / len(scores['test_'+metric])
        print(metric + ": " + str(avg_metric))


# Wrapper for cross-validation ROC plot, derived from the professor's discord code.
# I have my doubts about the significance of the plot, as I cannot see how it is generated
# with respect to the ten separate per-fold models. It will do, however; especially since
# the roc_auc from the above analytics shows the ROC is rubbish anyway.
def roc_plot_stopgap(pipeline, title):
    y_probas_forest = cross_val_predict(pipeline, data, y,  method="predict_proba", cv=10)

    y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
    fpr, tpr, thresholds_forest = roc_curve(y,y_scores_forest)

    plt.plot(fpr, tpr, "b:", label="classifier name")
    plt.title(title)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()


#print("Logistic Model")                # TODO NOT CONVERGING (....? but seems to still emit results??)
#evaluate_classifier(pipeline_logistic)
#roc_plot_stopgap(pipeline_logistic, "Logistic Classifier ROC")
#print("\nDecision Tree Model")
#evaluate_classifier(pipeline_tree)
#roc_plot_stopgap(pipeline_tree, "Decision Tree Classifier ROC")
#print("\nRandom Forest Model")
#evaluate_classifier(pipeline_forest)
#roc_plot_stopgap(pipeline_forest, "Random Forest Classifier ROC")
