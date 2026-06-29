import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import shap

print("Loading trained pipeline artifacts...")
artifacts = joblib.load("../models/urban_morphology_pipeline.joblib")

best_model = artifacts['model']
features = artifacts['features']
X_test = artifacts['X_test']
y_test = artifacts['y_test']

y_pred = best_model.predict(X_test)

print("\n--- Final Classification Report on Copenhagen Test Set (Scaled) ---")
print(classification_report(y_test, y_pred))

plot_dir = "../plots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

print("\n--- Initializing SHAP Analysis on Scaled Features ---")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test)
class_names = best_model.classes_
print(f"Model class order: {class_names}")

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:, :, :], X_test, class_names=class_names, show=False)
plt.title("Global Feature Importance (SHAP Summary Plot - Scaled)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "shap_global_summary.png"))
plt.close()
print("Saved: shap_global_summary.png")

for class_idx, class_name in enumerate(class_names):
    plt.figure(figsize=(10, 6))
    shap_values_for_class = shap_values[:, :, class_idx]
    shap.plots.beeswarm(shap_values_for_class, show=False)
    plt.title(f"Feature Impact on Class: {class_name} (SHAP Beeswarm - Scaled)")
    plt.tight_layout()
    
    filename = f"shap_beeswarm_{class_name.lower()}.png"
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()
    print(f"Saved: {filename}")

print(f"\nAll SHAP plots have been successfully saved to {plot_dir}")