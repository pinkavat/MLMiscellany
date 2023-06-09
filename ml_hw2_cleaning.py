# -*- coding: utf-8 -*-
"""HW2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UELLUZ9GOjteIyPUZAInXfCzr400zVKe

# Homework 2
Written January/February 2023 by Thomas Pinkava
"""

# Mount drive for reading
from google.colab import drive
drive.mount('/content/drive')

"""Initial exploratory analysis of the entire dataset shows the following percentage of NaN values for each feature:

<table>
<tr><td>id</td><td>0.0</td></tr>
<tr><td>state</td><td>0.0</td></tr>
<tr><td>stop_date</td><td>0.0</td></tr>
<tr><td>stop_time</td><td>0.9636412486017072</td></tr>
<tr><td>location_raw</td><td>0.035257380035580356</td></tr>
<tr><td>county_name</td><td>0.4832906888033208</td></tr>
<tr><td>county_fips</td><td>0.4832906888033208</td></tr>
<tr><td>fine_grained_location</td><td>0.9020057785639883</td></tr>
<tr><td>police_department</td><td>0.0</td></tr>
<tr><td>driver_gender</td><td>0.00000007</td></tr>
<tr><td>driver_age_raw</td><td>0.00000007</td></tr>
<tr><td>driver_age</td><td>0.0002345658397645386</td></tr>
<tr><td>driver_race_raw</td><td>0.0</td></tr>
<tr><td>driver_race</td><td>0.00000007</td></tr>
<tr><td>violation_raw</td><td>0.0</td></tr>
<tr><td>violation</td><td>0.0</td></tr>
<tr><td>search_conducted</td><td>0.0</td></tr>
<tr><td>search_type_raw</td><td>0.9927658095492778</td></tr>
<tr><td>search_type</td><td>0.9927658095492778</td></tr>
<tr><td>contraband_found</td><td>0.0</td></tr>
<tr><td>stop_outcome</td><td>0.0</td></tr>
<tr><td>is_arrested</td><td>0.0</td></tr>
<tr><td>search_basis</td><td>0.9927749117919449</td></tr>
<tr><td>officer_id</td><td>0.03482036776408326</td></tr>
<tr><td>drugs_related_stop</td><td>0.9992617767326589</td></tr>
<tr><td>ethnicity</td><td>0.00000007</td></tr>
<tr><td>district</td><td>0.035257380035580356</td></tr>
</table>

Therefore we decide to drop the features `stop_time`, `fine_grained_location`,`search_type_raw`,`search_type`, `search_basis`, and `drugs_related_stop`, as they consist of 90% or more missing values.

We also drop any feature labelled "`*_raw`" as they have matching features, which presumably contain the same information, 'cooked'. We can also drop `state`, as it is always North Carolina.

For machine learning purposes we also drop `id`, as this is presumably unique and directly identifies specific observations.

We also observe that `officer_id` has some mixed contents; we therefore replace any non-numerical officer ID with 0.

In order to maintain workability, we sampled 100000 observations at random from the source data to form `sample100k.csv`.
"""

import pandas as pd

# Converter function for removing strange string values of officer id
def eliminate_weird_officer_ids(officer_id):
  try:
    return int(officer_id)
  except:
    return 0

# Read csv
data_raw = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/HW2Data/sample100k.csv', converters={'officer_id':eliminate_weird_officer_ids})

# Drop the aforementioned columns, plus an unnamed index remnant
X = data_raw.drop(['Unnamed: 0','state','id','stop_time','stop_outcome','fine_grained_location','location_raw','search_type_raw','search_type','search_basis','drugs_related_stop','driver_age_raw','driver_race_raw','violation_raw'], axis=1)

"""## Part A

Due to the nature of the question we choose to drop observations where the `driver_age`, `driver_gender`, or `driver_race` are unspecified, rather than impute them.
"""

# Function for adding the desired 'age bucket' column
# There's probably an exceedingly clever way to do this (perhaps using pandas' cut function)
# but who has the time?
def age_bucketer(age):
  if age < 20:
    return '15-19'
  elif age < 30:
    return '20-29'
  elif age < 40:
    return '30-39'
  elif age < 50:
    return '40-49'
  else:
    return '50+'


# Process the data for part A, adding age buckets and dropping observations with missing fields
part_A_data = X.dropna(subset=['driver_gender','driver_age','driver_race'])
part_A_data['age_bucket'] = part_A_data['driver_age'].apply(age_bucketer) # Emits an indexing warning but seems to work
part_A_data = part_A_data[['age_bucket','driver_gender','driver_race','is_arrested']] # simplify things

# Compute the proportion of a given group arrested
def get_proportion_arrested_by_group(parameter, param_value):
  subset = part_A_data[part_A_data[parameter] == param_value]
  proportion_arrested = len(subset[subset['is_arrested']]) / len(subset[subset['is_arrested'] != True])
  return proportion_arrested

arrested_proportions = [
    get_proportion_arrested_by_group('driver_gender','M'),
    get_proportion_arrested_by_group('driver_gender','F'),
    get_proportion_arrested_by_group('driver_race','Asian'),
    get_proportion_arrested_by_group('driver_race','Black'),
    get_proportion_arrested_by_group('driver_race','Hispanic'),
    get_proportion_arrested_by_group('driver_race','White'),
    get_proportion_arrested_by_group('age_bucket','15-19'),
    get_proportion_arrested_by_group('age_bucket','20-29'),
    get_proportion_arrested_by_group('age_bucket','30-39'),
    get_proportion_arrested_by_group('age_bucket','40-49'),
    get_proportion_arrested_by_group('age_bucket','50+')
]

# Make a dataframe because it's easier on the eyes
pd.DataFrame(data={'Group':[
    'Males',
    'Females',
    'Asian People',
    'Black People',
    'Hispanic People',
    'White People',
    'Age 15-19',
    'Age 20-29',
    'Age 30-39',
    'Age 40-49',
    'Age 50+'
],'Proportion Arrested':arrested_proportions, 'Percentage Arrested':["{:.2%}".format(x) for x in arrested_proportions]})

"""Running the above code cell produces a table of results. We observe that Males are more often arrested than Females, that Hispanic people are the most arrested of the races computed, and that ages 20-29 are the most arrested ages.

## Part B

*   Age: We could use ordinal encoding on the buckets, as there is an order relating the categories of the categorical type. Alternatively, we could just not perform the bucketing at all, as the data contains age as a numerical feature already.
*   Gender: Ordinal encoding is also viable here, as it is a boolean and the technique is therefore irrelevant.
*   Race: There is no underlying ordering in the race category, so one-hot encoding must be used.

## Part C

##### Failed Pipeline
The field below contains an attempt at using Pipelines to preprocess the data. Pipelines don't keep the feature names, though, so we have to do it by hand.

# Define a preprocessing pipeline for the data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# TODO: separate stop_outcome, is_arrested, and search_conducted as ys.

# In order to preserve the class labels for post-analysis, we need some custom
# transformers. These are nonportable kitbashes to save on time.
from sklearn.base import BaseEstimator, TransformerMixin

# One-hot encoder for Race and Age Bucket
class FixedCategoricalEncoder(BaseEstimator, TransformerMixin):
  def __init__(self, fixed_categories, feature):
    self.fixed_categories = fixed_categories
    self.feature = feature

  def fit(self, X, y=None): # No need to fit
    return self
  
  def transform(self, X, y=None):
    for category in self.fixed_categories:
      X[self.feature + "_" + category] = X[self.feature].apply(lambda f: 1.0 if f == self.feature else 0.0)
    return X


# Binary categorizer for Gender, so that we can say with CERTAINTY that 
# Females are 1.0, Males and uncategorized 0.0.
class GenderEncoder(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None): # No need to fit
    return self
  
  def transform(self, X, y=None):
    X['driver_gender'] = X['driver_gender'].apply(lambda g: 1.0 if g == 'F' else 0.0)
    return X


# wrapper for age bucketer function
class AgeBucketEncoder(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None): # No need to fit
    return self

  def transform(self, X, y=None):
    X['driver_age'] = X['driver_age'].apply(age_bucketer)
    return X


uncaring_onehot_cat_features = ['county_name','police_department','violation','district']
uncaring_ordinal_cat_features = ['search_conducted','contraband_found']
numeric_features = ['county_fips','officer_id']


num_preproc_pipeline = Pipeline(steps=[
    ('imputation', SimpleImputer()),
    ('scaling', StandardScaler())
])

age_preproc_pipeline = Pipeline(steps=[
    ('bucketing', AgeBucketEncoder()),
    ('categorization', FixedCategoricalEncoder(['15-19','20-29','30-39','40-49','50+'],'driver_age'))
])

preprocessor = ColumnTransformer(
    transformers = [
        #('num_preproc', num_preproc_pipeline, numeric_features),
        #('unc_ord_preproc', OrdinalEncoder(), uncaring_ordinal_cat_features),
        #('unc_oht_preproc', OneHotEncoder(), uncaring_onehot_cat_features),
        ('gender_preproc', GenderEncoder(), ['driver_gender']),
        #('race_preproc', FixedCategoricalEncoder(["White","Black","Hispanic","Other","Asian"],'driver_race'),['driver_race']),
        #('age_preproc', age_preproc_pipeline, 'driver_age')
        # Ethnicity dropped: it's just hispanic/nonhispanic
        # Stop date not mined for other params.
    ]
)

preprocessor.fit_transform(X)

##### Pipeline-free attempt
Note: the above failed partly because of the inconvenience of sparse-matrix output from an anonymous one-hot encoder.

In keeping with the requirements of the assignment (which do not call for any performance optimization of the final model) I am going to *remove* all the features except gender, age, and race, since these are the coefficients in which we are interested. The model would of course be different with more features in play, but encoding them and then recovering the feature names is trying my patience.
"""

# Drop as before
X = X.dropna(subset=['driver_gender','driver_age','driver_race'])

# Extract the target labels and convert booleans to their float representations
y_search = X['search_conducted'].apply(lambda b: float(int(b)))
y_arrest = X['is_arrested'].apply(lambda b: float(int(b)))
y_citation = X['stop_outcome'].apply(lambda outcome: 1.0 if outcome == "Citation" else 0.0)

# Drop all but the relevant features
X_limited = X[['driver_gender','driver_age','driver_race']]

# Encode the relevant features

import warnings
with warnings.catch_warnings(): # Warning of some copy-slice thing. Seems to work, though.
  warnings.simplefilter("ignore")

  # Gender: 1.0 Female, 0.0 Male or Other
  X_limited['driver_gender'] = X_limited['driver_gender'].apply(lambda g: 1.0 if g == 'F' else 0.0)

  def custom_one_hot(dataframe, feature, fixed_values):
    for value in fixed_values:
      dataframe[feature + "_" + value] = dataframe[feature].apply(lambda f: 1.0 if f == value else 0.0)
    return dataframe.drop(feature, axis=1)

  # Age: One-hot encoding with columns
  X_limited['driver_age'] = X_limited['driver_age'].apply(age_bucketer)
  X_limited = custom_one_hot(X_limited, 'driver_age',['15-19','20-29','30-39','40-49','50+'])

  # Race: One-hot encoding with columns
  X_limited = custom_one_hot(X_limited, 'driver_race',["White","Black","Hispanic","Other","Asian"])

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

# For each desired target:
for label, target in [("a search", y_search), ("arrest", y_arrest), ("a citation", y_citation)]:
  # Split our data into train and test sets
  train_X, val_X, train_y, val_y = train_test_split(X_limited, target, random_state=1)

  # Fit the Logistic Regression classifier
  clf = LogisticRegression()
  clf.fit(train_X, train_y)

  # Make predictions on the test set, and get the proportion estimation
  predictions = clf.predict(val_X)
  proportion = sum(predictions) / len(predictions)
  actual_proportion = sum(val_y) / len(val_y)

  # Report estimation
  print("\n\nEstimated probability of {}: {}   (actual probability: {})".format(label, proportion, actual_proportion))

  # Mean squared error
  print("Mean squared error: {}".format(mean_squared_error(val_y, predictions)))

  # Obtain and report coefficients
  print("\nCoefficients affecting {}: ".format(label))
  for (feature_label, coefficient) in zip(clf.feature_names_in_, clf.coef_[0]):
    print("{}:  \t{}".format(feature_label, coefficient))

"""## Part D: Interpretation


#### Search

The assignment hints say to approximate $p(\hat{x}) = \frac{1}{1+e^{-(\beta_{0} + \beta_{1}x_{1}...))}}$ as $e^{\beta_{1}x_{1} + \beta_{2}x_{2}...}$ so $p_{search}(\hat{x}) = e^{ -0.87*"gender"-0.438*"age15-19"+...}$

* Hispanic drivers have a search coefficient of 0.595, while White drivers have a coefficient of -0.186. $e^{0.595}=1.81$ while $e^{-0.186} = 0.83$; $\frac{1.81}{0.83}=2.18$, so Hispanic drivers *are predicted to have* about double the chance of being searched as White drivers.

* Ditto: $\frac{blacksearchprob}{whitesearchprob} = \frac{e^{0.177}}{e^{-0.186}} = \frac{1.19}{0.83} = 1.43$; Black drivers are *predicted to be* about one and a half times more likely to be searched than White drivers.

#### Arrest

We can use the same simplification as in Search:

* $\frac{hispanicarrestprob}{whitearrestprob} = \frac{e^{0.476}}{e^{-0.227}} = \frac{1.61}{0.80} = 2.01$; Hispanic drivers are *predicted to be* twice as likely to be arrested as White drivers.

* $\frac{blackarrestprob}{whitearrestprob} = \frac{e^{-0.06}}{e^{-0.227}} = \frac{0.94}{0.80} = 1.2$; Black drivers and White drivers are *predicted to be* arrested at a similar rate.

#### Citation

The assignment hints tell us to pick values for the control variables, then calculate the relative probability. Obviously we can eliminate the alternative race variables altogether. The 'typical' values of the other variables we can take from their distributions:

<table>
<tr><th>One-hot flag</th><th>Proportion</th><th>Coefficient</th><th>Coefficient x Proportion</th></tr>
<tr><td>Female</td><td>0.332</td><td>0.124</td><td>0.04</td></tr>
<tr><td>Age 15-19</td><td>0.086</td><td>0.490</td><td>0.04</td></tr>
<tr><td>Age 20-29</td><td>0.315</td><td>0.443</td><td>0.12</td></tr>
<tr><td>Age 30-39</td><td>0.237</td><td>0.228</td><td>0.05</td></tr>
<tr><td>Age 40-49</td><td>0.184</td><td>0.049</td><td>0.01</td></tr>
<tr><td>Age 50+</td><td>0.179</td><td>-0.217</td><td>-0.04</td></tr>
</table>
total "$\beta_{0}$" (sum of the above contributions): 0.22

* So for Hispanic versus White: $\frac{hispaniccitationprob}{whitecitationprob} = \frac{\frac{1}{1+e^{-(0.22 + 0.325)}}}{\frac{1}{1+e^{-(0.22 + 0.049)}}} =\frac{0.633}{0.567}=1.12$

* And Black versus White: $\frac{blackcitationprob}{whitecitationprob} = \frac{\frac{1}{1+e^{-(0.22 + 0.072)}}}{\frac{1}{1+e^{-(0.22 + 0.049)}}} =\frac{0.572}{0.567}=1.01$


#### Age and Gender

* The gender coefficient (which in our model is associated with the state of being Female) is quite a large negative in both search and arrest, suggesting that womanhood significantly reduces the chances of both. The coefficient of citation is a small positive, so not much effect of gender on citation rate.

* The age coefficients show that age group 20-29 has the strongest effect on the rate of all three targets. We also see weak negative correlations on all three for group age 50+, so older people are less likely to be searched, arrested, or cited.

#### Accuracy of model

The mean squared error for search and arrest rates is very low. Citation's MSE is less low, but for a metric that is squared it doesn't seem to signal a significant problem with the model's accuracy.
"""