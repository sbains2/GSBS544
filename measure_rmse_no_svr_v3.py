import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Load data
train_df = pd.read_csv('regrs_train.csv')
y = train_df['SalePrice']
X = train_df.drop(['SalePrice', 'PID'], axis=1)

# Basic preprocessing
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def rmse_log_scorer(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1) # Ensure positive
    # y_true is raw, y_pred is raw (because TransformedTargetRegressor inverses it)
    return -np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))

scorer = make_scorer(rmse_log_scorer, greater_is_better=True)

print("Starting Log-Target Evaluation (No SVR)...")

# Lasso with Log Target
print("\n--- Lasso (Log Target) ---")
# Note: We apply log to target, so essentially predicting log price.
# TransformedTargetRegressor handles the log/exp conversion automatically.
lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(max_iter=5000))
])
model_lasso = TransformedTargetRegressor(regressor=lasso_pipeline, func=np.log1p, inverse_func=np.expm1)

# Need to search over regressor__regressor__alpha because of the nesting
lasso_param_grid = {
    'regressor__regressor__alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05] 
}
grid_lasso = GridSearchCV(model_lasso, lasso_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_lasso.fit(X, y)
print(f"Lasso (Log) CV RMSE (log): {-grid_lasso.best_score_:.4f}")
print(f"Best Lasso Params: {grid_lasso.best_params_}")


# Ridge with Log Target
print("\n--- Ridge (Log Target) ---")
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])
model_ridge = TransformedTargetRegressor(regressor=ridge_pipeline, func=np.log1p, inverse_func=np.expm1)

ridge_param_grid = {
    'regressor__regressor__alpha': [1, 5, 10, 20, 50, 100]
}
grid_ridge = GridSearchCV(model_ridge, ridge_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_ridge.fit(X, y)
print(f"Ridge (Log) CV RMSE (log): {-grid_ridge.best_score_:.4f}")
print(f"Best Ridge Params: {grid_ridge.best_params_}")


# Poly Ridge with Log Target (Degree 2, Interaction Only)
print("\n--- Poly Ridge (Log Target, Deg 2, Int Only) ---")
poly_ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('regressor', Ridge())
])
model_poly_ridge = TransformedTargetRegressor(regressor=poly_ridge_pipeline, func=np.log1p, inverse_func=np.expm1)

poly_ridge_param_grid = {
    'regressor__regressor__alpha': [10, 20, 50, 100, 200]
}
grid_poly_ridge = GridSearchCV(model_poly_ridge, poly_ridge_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_poly_ridge.fit(X, y)
print(f"Poly Ridge (Log) CV RMSE (log): {-grid_poly_ridge.best_score_:.4f}")
print(f"Best Poly Ridge Params: {grid_poly_ridge.best_params_}")
