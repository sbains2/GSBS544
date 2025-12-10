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

print("Starting Round 3 model evaluation...")

# Polynomial Lasso - Higher Alphas
print("\n--- Polynomial Lasso (Interaction Only) ---")
poly_lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('regressor', Lasso(max_iter=5000))
])
poly_lasso_param_grid = {
    'regressor__alpha': [1, 5, 10, 20, 50, 100] 
}
grid_poly_lasso = GridSearchCV(poly_lasso_pipeline, poly_lasso_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_poly_lasso.fit(X, y)
print(f"Poly Lasso CV RMSE (log): {-grid_poly_lasso.best_score_:.4f}")
print(f"Best Poly Lasso Params: {grid_poly_lasso.best_params_}")

# Polynomial ElasticNet - Higher Alphas
print("\n--- Polynomial ElasticNet (Interaction Only) ---")
poly_enet_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('regressor', ElasticNet(max_iter=5000))
])
poly_enet_param_grid = {
    'regressor__alpha': [0.1, 1, 5, 10, 20],
    'regressor__l1_ratio': [0.1, 0.5, 0.9]
}
grid_poly_enet = GridSearchCV(poly_enet_pipeline, poly_enet_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_poly_enet.fit(X, y)
print(f"Poly ElasticNet CV RMSE (log): {-grid_poly_enet.best_score_:.4f}")
print(f"Best Poly ElasticNet Params: {grid_poly_enet.best_params_}")

# SVR RBF - Refined Grid
print("\n--- SVR (RBF) ---")
svr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf'))
])
svr_param_grid = {
    'regressor__C': [10000, 50000, 100000],  # Increase C significantly for SVR
    'regressor__epsilon': [0.01, 0.1],
    'regressor__gamma': ['scale', 0.001]
}
grid_svr = GridSearchCV(svr_pipeline, svr_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_svr.fit(X, y)
print(f"SVR (RBF) CV RMSE (log): {-grid_svr.best_score_:.4f}")
print(f"Best SVR (RBF) Params: {grid_svr.best_params_}")
