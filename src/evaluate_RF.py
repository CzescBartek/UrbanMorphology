import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve, StratifiedKFold, validation_curve
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier

print("Loading trained pipeline artifacts...")
model_path = "../models/RandomForest.joblib"
artifacts = joblib.load(model_path)

best_model = artifacts['model']
features = artifacts['features']
X_test = artifacts['X_test']
y_test = artifacts['y_test']

y_pred = best_model.predict(X_test)
class_names = best_model.classes_

plot_dir = "../plots_RF/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

print("\n--- Saving Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Copenhagen Test Set")
plt.ylabel("Actual Style")
plt.xlabel("Predicted Style")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.close()
print("Saved: confusion_matrix.png")

print("\n--- Saving Learning Curve ---")
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X_test,  
    y=y_test,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
plt.title("Learning Curve (F1-Macro)")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "learning_curve.png"))
plt.close()
print("Saved: learning_curve.png")

print("\n--- Saving Multiclass CV ROC-AUC Curve with SD ---")
y_test_bin = label_binarize(y_test, classes=class_names)
n_classes = len(class_names)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tprs = {i: [] for i in range(n_classes)}
aucs = {i: [] for i in range(n_classes)}
mean_fpr = np.linspace(0, 1, 100)

for train_idx, val_idx in cv.split(X_test, y_test):
    X_cv_train, X_cv_val = X_test.iloc[train_idx], X_test.iloc[val_idx]
    y_cv_train, y_cv_val = y_test.iloc[train_idx], y_test.iloc[val_idx]
    
    y_cv_val_bin = label_binarize(y_cv_val, classes=class_names)
    
    cv_model = best_model.fit(X_cv_train, y_cv_train)
    y_score = cv_model.predict_proba(X_cv_val)
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_cv_val_bin[:, i], y_score[:, i])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs[i].append(interp_tpr)
        aucs[i].append(auc(fpr, tpr))

plt.figure(figsize=(10, 8))

for i in range(n_classes):
    mean_tpr = np.mean(tprs[i], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs[i])
    
    plt.plot(
        mean_fpr, 
        mean_tpr, 
        label=f'Mean ROC {class_names[i]} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
        lw=2, 
        alpha=0.8
    )
    
    std_tpr = np.std(tprs[i], axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.15)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass 5-Fold CV ROC-AUC Curve (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "cv_roc_auc_curve.png"))
plt.close()
print("Saved: cv_roc_auc_curve.png")

print("\n--- Saving Hyperparameter Tuning Curves ---")
n_estimators_range = [10, 50, 100, 200, 300, 400, 500, 600]
max_depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

print("Calculating validation curve for n_estimators...")
train_scores_n, val_scores_n = validation_curve(
    estimator=base_rf,
    X=X_test,
    y=y_test,
    param_name="n_estimators",
    param_range=n_estimators_range,
    cv=5,
    scoring="roc_auc_ovr_weighted",
    n_jobs=-1
)

val_mean_n = np.mean(val_scores_n, axis=1)
val_std_n = np.std(val_scores_n, axis=1)

ax1.errorbar(n_estimators_range, val_mean_n, yerr=val_std_n, fmt='s-', color='r', ecolor='r', elinewidth=1.5, capsize=0)
ax1.set_title("Cross-Validation: N Estimators (Selected Optimal: 200)", fontsize=11)
ax1.set_xlabel("n_estimators")
ax1.set_ylabel("Mean ROC AUC")
ax1.grid(True)

print("Calculating validation curve for max_depth...")
rf_with_estimators = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

train_scores_d, val_scores_d = validation_curve(
    estimator=rf_with_estimators,
    X=X_test,
    y=y_test,
    param_name="max_depth",
    param_range=max_depth_range,
    cv=5,
    scoring="roc_auc_ovr_weighted",
    n_jobs=-1
)

val_mean_d = np.mean(val_scores_d, axis=1)
val_std_d = np.std(val_scores_d, axis=1)

ax2.errorbar(max_depth_range, val_mean_d, yerr=val_std_d, fmt='o-', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.5, capsize=0)
ax2.set_title("Cross-Validation: Max Depth (Selected Optimal: 5)", fontsize=11)
ax2.set_xlabel("max_depth")
ax2.set_ylabel("Mean ROC AUC")
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "hyperparameter_tuning_curves.png"))
plt.close()
print("Saved: hyperparameter_tuning_curves.png")

print(f"\nAll performance evaluation plots have been successfully saved to {plot_dir}")