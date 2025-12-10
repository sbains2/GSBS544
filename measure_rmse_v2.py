import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
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
    y_pred = np.maximum(y_pred, 1)
    return -np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))

scorer = make_scorer(rmse_log_scorer, greater_is_better=True)

print("Starting extended model evaluation...")

# SVR Expanded
print("\n--- SVR ---")
svr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])
svr_param_grid = {
    'regressor__C': [10, 20, 50, 100], 
    'regressor__kernel': ['rbf', 'poly'], 
    'regressor__degree': [2], # Stick to 2 for speed
    'regressor__epsilon': [0.01, 0.05, 0.1],
    'regressor__gamma': ['scale', 0.01]
}
grid_svr = GridSearchCV(svr_pipeline, svr_param_grid, cv=cv, scoring=scorer, n_jobs=-1, error_score='raise')
grid_svr.fit(X, y)
print(f"SVR CV RMSE (log): {-grid_svr.best_score_:.4f}")
print(f"Best SVR Params: {grid_svr.best_params_}")

# Polynomial Ridge
print("\n--- Polynomial Ridge ---")
poly_ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', Ridge())
])
poly_ridge_param_grid = {
    'poly__interaction_only': [False, True],
    'regressor__alpha': [10, 20, 50, 100, 200]
}
grid_poly = GridSearchCV(poly_ridge_pipeline, poly_ridge_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_poly.fit(X, y)
print(f"Poly Ridge CV RMSE (log): {-grid_poly.best_score_:.4f}")
print(f"Best Poly Ridge Params: {grid_poly.best_params_}")
