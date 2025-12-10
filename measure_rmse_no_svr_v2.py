import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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

print("Starting Poly Model Evaluation Round 2 (No SVR)...")

# Poly Ridge (Degree 2, Interaction Only) - Fine Tune around 20
print("\n--- Poly Ridge (Degree 2, Interaction Only) ---")
poly_ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('regressor', Ridge())
])
poly_ridge_param_grid = {
    'regressor__alpha': [15, 20, 25, 30, 40]
}
grid_poly_ridge = GridSearchCV(poly_ridge_pipeline, poly_ridge_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_poly_ridge.fit(X, y)
print(f"Poly Ridge CV RMSE (log): {-grid_poly_ridge.best_score_:.4f}")
print(f"Best Poly Ridge Params: {grid_poly_ridge.best_params_}")

# Poly ElasticNet (Degree 2, Interaction Only) - Fine Tune
print("\n--- Poly ElasticNet (Degree 2, Interaction Only) ---")
poly_enet_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('regressor', ElasticNet(max_iter=5000, selection='random'))
])
poly_enet_param_grid = {
    'regressor__alpha': [0.05, 0.1, 0.2, 0.5],
    'regressor__l1_ratio': [0.7, 0.8, 0.9, 0.95, 0.99]
}
grid_poly_enet = GridSearchCV(poly_enet_pipeline, poly_enet_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_poly_enet.fit(X, y)
print(f"Poly ElasticNet CV RMSE (log): {-grid_poly_enet.best_score_:.4f}")
print(f"Best Poly ElasticNet Params: {grid_poly_enet.best_params_}")

# Poly Ridge (Degree 3, Interaction Only)
print("\n--- Poly Ridge (Degree 3, Interaction Only) ---")
poly_ridge3_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)),
    ('regressor', Ridge())
])
poly_ridge3_param_grid = {
    'regressor__alpha': [50, 100, 200, 500, 1000] # Likely needs higher regularization
}
grid_poly_ridge3 = GridSearchCV(poly_ridge3_pipeline, poly_ridge3_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_poly_ridge3.fit(X, y)
print(f"Poly Ridge (Deg 3) CV RMSE (log): {-grid_poly_ridge3.best_score_:.4f}")
print(f"Best Poly Ridge (Deg 3) Params: {grid_poly_ridge3.best_params_}")
