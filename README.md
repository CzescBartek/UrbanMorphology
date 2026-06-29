# International Urban Morphology Classifier using Spatial Data Science & ML

An advanced R&D Spatial Data Science pipeline that uses machine learning to automatically classify urban fabrics into distinct architectural eras:
- `Medieval_Center`
- `XIX_Century_Blocks`
- `Modern_Urbanism`

The classification is based solely on raw vector geometries from OpenStreetMap.

The project focuses on cross-border generalization challenges (**Domain Shift** and **Scale Shift**) across 16 major Central and Northern European metropolises, with **Copenhagen** used as a completely unseen final test bed.

---

## 📂 Project Structure

Following a production-grade modular design, the repository is organized as follows:

- `data/config.json` - spatial bounds and definitions for 16 target cities.
- `data/city_data/` - city grids used for spatial processing and inference.
- `data/combined_urban_features.csv` - final integrated multi-city dataset used for training.
- `models/RandomForest.joblib` - serialized Random Forest pipeline artifacts (model + scaler + features).
- `plots_RF/confusion_matrix.png` - threshold-tuned test performance layout.
- `plots_RF/cv_roc_auc_curve.png` - multiclass One-vs-Rest ROC analysis.
- `plots_RF/hyperparameter_tuning_curves.png` - validation curves for `max_depth` and `n_estimators`.
- `plots_RF/learning_curve.png` - convergence and data scalability analysis.
- `plots/copenhagen_prediction_map.html` - interactive spatial inference map.
- `src/building_features.py` - feature extractor for polygon elongation and shape metrics.
- `src/saving_data.py` - OSMnx/GeoPandas pipeline for feature extraction and dataset assembly.
- `src/evaluate_RF.py` - Random Forest evaluation and visualization generator.
- `src/generate_map.py` - focused inference and Folium HTML map pipeline.
- `src/road_features.py` - feature extractor for structural linear density.
- `src/save_features.py` - primary feature consolidation and data assembly.
- `src/Baseline_RF_model.py` - scaler alignment and hyperparameter grid search for Random Forest.
- `src/Lgbm_model.py` - benchmark training pipeline for LightGBM.

---

## 🎯 Project Overview & Objectives

Traditional urban morphology workflows often rely on manual classification or localized satellite-based interpretation.
This project implements a scalable, production-ready alternative inspired by quantitative urban morphometrics (Fleischmann et al., 2021).

The core objective is to train an ensemble model on historical and spatial footprints from 15 continental cities (Poland, Germany, Netherlands, Austria, Sweden) and achieve robust predictive generalization on a fully unseen urban layout (**Copenhagen, Denmark**) despite differences in local building culture and urban form.

---

## 🛠️ Data & Engineering Pipeline

The architecture is divided into decoupled, production-grade stages:

### 1) Extraction (`saving_data.py`, `save_features.py`)

Connects to OpenStreetMap through OSMnx and processes raw building/road geometries into standardized `400 x 400 m` grids mapped in `copenhagen_grid.geojson`.

### 2) Feature Engineering (`building_features.py`, `road_features.py`, spatial lag in `saving_data.py`)

Transforms absolute values into dynamic, scale-aware spatial descriptors:

- `avg_building_elongation`  
  Bounding-box aspect ratio using minimum rotated rectangles; useful for separating narrow medieval parcels from larger modern blocks.

- `building_density_spatial_lag`  
  Local neighborhood context based on moving average building density of adjacent cells.

- `building_to_road_density_ratio` and `courtyard_index`  
  Metrics that help distinguish compact perimeter blocks from contemporary open layouts.

### 3) Training & Scaling (`Baseline_RF_model.py`)

Standardizes predictors with `StandardScaler` and trains an optimized `RandomForestClassifier` through cross-validated hyperparameter search.

---

## 📈 Model Performance & Evaluation

### 1) Model Selection Benchmark (Random Forest vs LightGBM)

During development, LightGBM was benchmarked against Random Forest.
Under high local class overlap and noisy small-cell contexts, LightGBM showed underfitting behavior, while Random Forest provided more stable separation of non-linear spatial patterns.

**Figure 1: Learning Curve**  
Cross-validation score scales toward ~`0.62` macro F1 with a positive boundary trend, suggesting meaningful cross-border generalization potential.

![Learning Curve](plots_RF/learning_curve.png)

### 2) Hyperparameter Optimization

A 5-fold cross-validation grid search was used to control model complexity and reduce overfitting on municipality-specific patterns.

**Figure 2: Validation Curves**  
`n_estimators` and `max_depth` tuning show a balanced optimum between robustness and computational cost.

![Hyperparameter Tuning Curves](plots_RF/hyperparameter_tuning_curves.png)

### 3) Classification Threshold Fine-Tuning

Because medieval cores form a minority class in most large cities, default multiclass argmax thresholds were replaced by class-specific tuned thresholds derived from training precision-recall dynamics.

Final report on Copenhagen test set (`copenhagen_features`):

- `Medieval_Center` - Precision: `0.55` | Recall: `0.48` | F1: `0.51` | Support: `25`
- `XIX_Century_Blocks` - Precision: `0.79` | Recall: `0.82` | F1: `0.80` | Support: `40`
- `Modern_Urbanism` - Precision: `0.60` | Recall: `0.56` | F1: `0.58` | Support: `32`
- Accuracy: `0.64` | Support: `97`
- Macro Avg - Precision: `0.65` | Recall: `0.62` | F1: `0.63` | Support: `97`
- Weighted Avg - Precision: `0.66` | Recall: `0.64` | F1: `0.65` | Support: `97`

**Confusion Matrix (Copenhagen holdout):**

![Confusion Matrix](plots_RF/confusion_matrix.png)

---

## 🔬 Spatial Interpretability & Mapping

### 1) Discriminative Power (ROC-AUC)

To measure class separability independently of fixed decision thresholds, the project uses multiclass One-vs-Rest ROC-AUC analysis.

**Figure 3: ROC-AUC**
- `XIX_Century_Blocks`: up to `0.98` AUC
- `Modern_Urbanism`: around `0.83` AUC

These results indicate strong separability for key morphology classes under engineered geometric features.

![Multiclass ROC-AUC](plots_RF/cv_roc_auc_curve.png)

### 2) Spatial Visualization (Inference Map)

Inference outputs are projected back to geographic space and visualized as an interactive map.

Open locally:
- `plots/copenhagen_prediction_map.html`
  
Direct link from GitHub file view:
- [Open interactive Copenhagen map](plots/copenhagen_prediction_map.html)

Qualitative observations from Copenhagen:
- **North Cluster (Nordhavn/Østerbro):** predominantly `Modern_Urbanism`.
- **Center Cluster (Nørrebro):** compact perimeter fabric mapped as `XIX_Century_Blocks`.
- **South Cluster (Ørestad/Tårnby):** contemporary expansion zones with mixed transitional pockets.

---

## 🚀 Key Takeaways for Technical Interviews

- **Domain Shift Mitigation**  
  Cross-country generalization improved by engineering relative geometric descriptors instead of country-specific absolute values.

- **MLOps-Oriented Design**  
  ETL, training, and evaluation are decoupled into modular scripts. Artifacts are serialized with `joblib` for reuse and deployment.

- **Data-Centric Optimization**  
  Performance improvements came primarily from spatial feature engineering (e.g., spatial lag and elongation) rather than increasing model complexity alone.
