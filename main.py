import os
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(numerical_attributes, categorical_attributes):
    # For numerical attributes
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])  

    # For categorical attributes
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder", OneHotEncoder())
    ])

    # Combine both numerical and categorical pipeline
    full_pipeline = ColumnTransformer([
        ("num_pipeline", numerical_pipeline, numerical_attributes),
        ("cat_pipeline", categorical_pipeline, categorical_attributes)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # We have to train the model
    housing = pd.read_csv("housing.csv")

    # Creating Stratified Test set

    housing["income_cat"] = pd.cut(housing["median_income"], 
                                bins = [0,1.5,3.0,4.5,6.0, np.inf],
                                labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)
        

    # Feature Engineering

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # List the numerical and categorical columns

    numerical_attributes = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    categorical_attributes = ["ocean_proximity"]

    pipeline= build_pipeline(numerical_attributes, categorical_attributes)
    
    housing_prepared = pipeline.fit_transform(housing_features)
    
    # Training the model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Congratulations! Model has been trained successfully and saved to disk.", flush=True)
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    tranformed_input = pipeline.transform(input_data)
    predictions = model.predict(tranformed_input)
    input_data["median_house_value"] = predictions

    input_data.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv", flush=True)