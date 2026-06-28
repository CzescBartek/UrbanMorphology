import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


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
    'building_to_road_density_ratio'
]

train_df = df_filtered[df_filtered['city_name'] != 'Copenhagen Municipality, Denmark']
test_df = df_filtered[df_filtered['city_name'] == 'Copenhagen Municipality, Denmark']

X_train = train_df[features]
y_train = train_df['target_style']

X_test = test_df[features]
y_test = test_df['target_style']


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],             
    'min_samples_split': [5, 10],
    'max_features': ['log2'],            
    'class_weight': ['balanced']         
}


rf = RandomForestClassifier(random_state=42)


print("Starting Grid Search with 5-Fold Cross-Validation on non CPH data...")
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1_macro',                  
    n_jobs=-1,                           
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(grid_search.best_params_)
print(f"Best CV Macro F1-Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- Final Classification Report on Copenhagen Test Set ---")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title("Confusion Matrix - Optimized Random Forest")
plt.ylabel("Actual Style")
plt.xlabel("Predicted Style")
plt.show()


importances = best_model.feature_importances_
feat_importances = pd.Series(importances, index=features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_importances.plot(kind='barh', color='#3498DB')
plt.title("Feature Importances (Optimized Model)")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()