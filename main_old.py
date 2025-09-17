
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


# Load the dataset
housing = pd.read_csv("/Users/prathmeshkarale/Desktop/Housing_model/house_price_prediction/csv/housing_data.csv")


# create a stratified shuffle set
housing['income_cat']=pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set=housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set=housing.loc[test_index].drop("income_cat",axis=1)
    
    
# work on copy of training set data
housing = strat_train_set.copy()


# seperate labels and predictors
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)


# list the numerical and categorical columns
num_attribs=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs= ["ocean_proximity"]


# making the pipeline for numerical columns
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])


# making the pipeline for categorical columns
cat_pipeline=Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="error"))
])


# construct full pipeline
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",cat_pipeline,cat_attribs)
])
 
 
# transfrom the data
housing_prepared=full_pipeline.fit_transform(housing)


# Train the model

# linear regression model
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_pred=lin_reg.predict(housing_prepared)
lin_rmse=root_mean_squared_error(housing_labels,lin_pred)
print(f"The root mean square error for Linear Regression is {lin_rmse}")

lin_rmse_scores = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print(pd.Series(lin_rmse_scores).describe())


# Descision Tree model
dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_pred=dec_reg.predict(housing_prepared)
dec_rmse=root_mean_squared_error(housing_labels,dec_pred)
print(f"The root mean square error for Decision Tree is {dec_rmse}")

dec_rmse_scores= -cross_val_score(
    dec_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print(pd.Series(dec_rmse_scores).describe())


# Random Forest model
rf_reg= RandomForestRegressor()
rf_reg.fit(housing_prepared,housing_labels)
rf_pred=rf_reg.predict(housing_prepared)
rf_rmse=root_mean_squared_error(housing_labels,rf_pred)
print(f"The root mean square error for Random Forest Regressor is {rf_rmse}")

rf_rmse_scores= -cross_val_score(
    dec_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print(pd.Series(rf_rmse_scores).describe())

