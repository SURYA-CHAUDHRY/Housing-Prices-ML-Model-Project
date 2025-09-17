import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

# 1. Loading the Dataset

housing = pd.read_csv("housing.csv")

# 2. Creating Stratified Test set

housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins = [0,1.5,3.0,4.5,6.0, np.inf],
                               labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Working on the copy of the test set

housing = strat_train_set.copy()

# 3. Feature Engineering

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1).copy()

# print(housing, housing_labels)

# 4. List the numerical and categorical columns

numerical_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
categorical_attributes = ["ocean_proximity"]

# 5. Making pipeline 
# (1) For numerical attributes

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])  

# (2) For categorical attributes
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder())
])

# Construcitng the full pipeline
full_pipeline = ColumnTransformer([
    ("num", numerical_pipeline, numerical_attributes),
    ("cat", categorical_pipeline, categorical_attributes)
])

# 6. Applying the pipeline to the data
housing_prepared = full_pipeline.fit_transform(housing)

# print(housing_prepared)
# print(housing_prepared.shape)

# 7. Selecting and training models

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_predictions = lin_reg.predict(housing_prepared)
# lin_reg_rmse = root_mean_squared_error(housing_labels, lin_predictions)
lin_reg_rmse = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"Linear Regression RMSE: {lin_reg_rmse} ")
print("Linear Regression Model Results:")
print(pd.Series(lin_reg_rmse).describe())

# Decison Tree Model
desc_reg = DecisionTreeRegressor()
desc_reg.fit(housing_prepared, housing_labels)
desc_predictions = desc_reg.predict(housing_prepared)
#desc_reg_rmse = root_mean_squared_error(housing_labels, Tree_predictions)
desc_reg_rmse = -cross_val_score(desc_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"Decision Tree RMSE: {desc_reg_rmse} ")
print("Decision Tree Model Results:")
print(pd.Series(desc_reg_rmse).describe())

# Random Forest Model
random_forest_reg = RandomForestRegressor()   
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_predictions = random_forest_reg.predict(housing_prepared)
# random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_predictions)
random_forest_rmse = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"Random Forest RMSE: {random_forest_rmse} ")
print("Random Forest Model Results:")
print(pd.Series(random_forest_rmse).describe())

## Here We can see that Random Forest is performing the best because it has the lowest mean and standard deviation.
# print("Random Forest is performing the best because it has the lowest mean and standard deviation.")