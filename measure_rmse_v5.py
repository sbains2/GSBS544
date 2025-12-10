import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
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

print("Starting Final SVR fine-tuning...")

# SVR RBF - Fine-tuning
print("\n--- SVR (RBF) Fine-tune ---")
svr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf'))
])
svr_param_grid = {
    'regressor__C': [50000, 80000, 100000, 150000],  
    'regressor__epsilon': [0.05, 0.08, 0.1, 0.12],
    'regressor__gamma': ['scale', 0.0005, 0.001, 0.005]
}
grid_svr = GridSearchCV(svr_pipeline, svr_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_svr.fit(X, y)
print(f"SVR (RBF) CV RMSE (log): {-grid_svr.best_score_:.4f}")
print(f"Best SVR (RBF) Params: {grid_svr.best_params_}")
