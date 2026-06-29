import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("../data/combined_urban_features.csv")
df_filtered = df[df['building_count'] >= 3].copy()

df_filtered['road_to_building_ratio'] = df_filtered['total_road_length'] / (df_filtered['avg_building_area'] + 1e-5)
df_filtered['building_footprint_complexity'] = df_filtered['shape_complexity_ratio'] * df_filtered['courtyard_index']
df_filtered['building_to_road_density_ratio'] = df_filtered['building_density'] / (df_filtered['road_density_ratio'] + 1e-5)

features = [
    'building_density', 
    'road_density_ratio',
    'road_to_building_ratio',
    'building_footprint_complexity',
    'std_building_area',
    'building_to_road_density_ratio',
    'avg_building_elongation',
    'building_density_spatial_lag'
]

train_df = df_filtered[~df_filtered['city_name'].str.contains('Copenhagen', case=False, na=False)]
test_df = df_filtered[df_filtered['city_name'].str.contains('Copenhagen', case=False, na=False)]

X_train_raw = train_df[features]
y_train = train_df['target_style']
X_test_raw = test_df[features]
y_test = test_df['target_style']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_train = pd.DataFrame(X_train_scaled, columns=features)
X_test = pd.DataFrame(X_test_scaled, columns=features)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'class_weight': ['balanced'],
    'subsample': [0.8, 1.0],
    'random_state': [42]
}

lgbm = LGBMClassifier(verbosity=-1)

print("Starting Grid Search with 5-Fold Cross-Validation for LightGBM...")
grid_search = GridSearchCV(
    estimator=lgbm, 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1_macro', 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\nBest Parameters Found for LightGBM:")
print(grid_search.best_params_)
print(f"Best CV Macro F1-Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

pipeline_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'features': features,
    'X_test': X_test,
    'y_test': y_test,
    'X_train': X_train,
    'y_train': y_train
}

joblib.dump(pipeline_artifacts, "../models/lgbm.joblib")
print("\nSuccessfully saved LightGBM artifacts to ../models/lgbm.joblib")