import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score

print("Loading trained LightGBM pipeline artifacts...")
model_path = "../models/lgbm.joblib"
artifacts = joblib.load(model_path)

best_model = artifacts['model']
features = artifacts['features']
X_test = artifacts['X_test']
y_test = artifacts['y_test']
X_train = artifacts['X_train']
y_train = artifacts['y_train']

class_names = best_model.classes_
n_classes = len(class_names)

print("\n--- Tuning Decision Thresholds for LightGBM ---")
y_train_probs = best_model.predict_proba(X_train)

best_thresholds = np.ones(n_classes) / n_classes
best_f1 = 0.0

for t0 in np.linspace(0.1, 0.6, 6):
    for t1 in np.linspace(0.1, 0.6, 6):
        for t2 in np.linspace(0.1, 0.6, 6):
            thresholds = np.array([t0, t1, t2])
            scaled_probs = y_train_probs / thresholds
            preds_idx = np.argmax(scaled_probs, axis=1)
            preds = class_names[preds_idx]
            
            score = f1_score(y_train, preds, average='macro')
            if score > best_f1:
                best_f1 = score
                best_thresholds = thresholds

print(f"Optimal Thresholds Found: {dict(zip(class_names, best_thresholds))}")
print(f"Best Training Macro F1-Score with Adjusted Thresholds: {best_f1:.4f}")

y_test_probs = best_model.predict_proba(X_test)
y_pred_idx = np.argmax(y_test_probs / best_thresholds, axis=1)
y_pred = class_names[y_pred_idx]

print("\n--- Final LightGBM Classification Report on Copenhagen Test Set ---")
print(classification_report(y_test, y_pred))

plot_dir = "../plots_lgbm/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

print("\n--- Saving LightGBM Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (LightGBM) - Copenhagen Test Set")
plt.ylabel("Actual Style")
plt.xlabel("Predicted Style")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "confusion_matrix_lgbm.png"))
plt.close()

print("\n--- Saving LightGBM Learning Curve ---")
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
plt.title("Learning Curve (LightGBM - F1-Macro)")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "learning_curve_lgbm.png"))
plt.close()

print("\n--- Saving LightGBM Multiclass CV ROC-AUC Curve with SD ---")
y_test_bin = label_binarize(y_test, classes=class_names)

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
plt.title('Multiclass 5-Fold CV ROC-AUC Curve (LightGBM - One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "cv_roc_auc_curve_lgbm.png"))
plt.close()

print(f"\nAll LightGBM performance plots have been successfully saved to {plot_dir}")