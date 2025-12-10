import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error

# Load the training and test datasets
train_df = pd.read_csv('regrs_train.csv')
test_df = pd.read_csv('regrs_test.csv')

# Define target variable and features (exclude PID and SalePrice)
y = train_df['SalePrice']
X = train_df.drop(['SalePrice', 'PID'], axis=1)

# Fill missing values in numeric columns with median
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# Fill missing values in categorical columns with mode (most frequent)
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Numeric preprocessing: standardize to zero mean and unit variance
numeric_transformer = StandardScaler()

# Categorical preprocessing: one-hot encode, handle unknown categories in test set
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine transformers into single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Set up 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Custom scorer: negative RMSE on log-transformed predictions (negative for sklearn)
def rmse_log_scorer(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1)  # Ensure no negative predictions
    return -np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))

scorer = make_scorer(rmse_log_scorer, greater_is_better=True)

print("Starting model evaluation...")

# 1. Linear Regression (Baseline)
# lr_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', LinearRegression())
# ])
# lr_scores = cross_val_score(lr_pipeline, X, y, cv=cv, scoring=scorer)
# print(f"Linear Regression CV RMSE (log): {-lr_scores.mean():.4f}")

# 2. Lasso Regression (Extended)
lasso2_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(max_iter=10000))
])
lasso2_param_grid = {
    'regressor__alpha': [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
}
grid_lasso2 = GridSearchCV(lasso2_pipeline, lasso2_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_lasso2.fit(X, y)
print(f"Lasso (Extended) CV RMSE (log): {-grid_lasso2.best_score_:.4f}")
print(f"Best Lasso Params: {grid_lasso2.best_params_}")


# 3. Elastic Net (Extended)
elasticnet2_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(max_iter=10000))
])
elasticnet2_param_grid = {
    'regressor__alpha': [0.00000001, 0.00000005, 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001],
    'regressor__l1_ratio': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
}
grid_elasticnet2 = GridSearchCV(elasticnet2_pipeline, elasticnet2_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_elasticnet2.fit(X, y)
print(f"Elastic Net (Extended) CV RMSE (log): {-grid_elasticnet2.best_score_:.4f}")
print(f"Best Elastic Net Params: {grid_elasticnet2.best_params_}")

# 4. SVR
svr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])
svr_param_grid = {
    'regressor__C': [0.5, 1, 2, 5, 8, 10, 15, 20],
    'regressor__kernel': ['rbf', 'linear'],
    'regressor__epsilon': [0.005, 0.01, 0.02, 0.05, 0.08, 0.1],
    'regressor__gamma': ['scale', 'auto', 0.01, 0.05]
}
grid_svr = GridSearchCV(svr_pipeline, svr_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_svr.fit(X, y)
print(f"SVR CV RMSE (log): {-grid_svr.best_score_:.4f}")
print(f"Best SVR Params: {grid_svr.best_params_}")

