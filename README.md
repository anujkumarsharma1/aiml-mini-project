# AquaGuard AI – Water Quality Classification System

An end-to-end, beginner-friendly Machine Learning and Streamlit web application designed for college final-year and semester projects. **AquaGuard AI** analyzes historical water-quality physicochemical parameters and classifies water samples as **Potable** or **Non-Potable** using authentic machine learning models trained on a genuine public dataset.

---

## 📌 1. Problem Definition

### Problem Statement
> *"Water quality depends on several physical and chemical parameters. This project uses machine learning to analyze historical water-quality data and classify water samples as potable or non-potable according to their recorded dataset labels."*

### Key Objectives
1. **Analyze Physicochemical Indicators:** Evaluate water quality parameters (pH, hardness, solids, chloramines, sulfates, conductivity, organic carbon, trihalomethanes, turbidity).
2. **Handle Incomplete Data Without Leakage:** Employ robust median imputation for missing sensor values and fit all transformers strictly on training data.
3. **Compare Linear vs Ensemble Algorithms:** Demonstrate why linear models (Logistic Regression) fail on multi-variable chemical relationships, and how ensemble tree algorithms (Random Forest) succeed.
4. **Optimize Hyperparameters:** Apply 5-fold cross-validation with `GridSearchCV` on the training dataset.
5. **Interactive Web Deployment:** Provide a clean, water-themed Streamlit interface with instant predictions, class probability estimation, dataset analysis, and academic presentation guidance.

### Scope and Limitations
- **Educational Scope:** AquaGuard AI is created for educational and decision-support demonstration.
- **Scientific Limitation:** Machine learning predictions reflect historical dataset labels and statistical correlations. They do not substitute certified laboratory chemical testing or official drinking water safety compliance (e.g., WHO, EPA, BIS standards).

### Potential Applications
- Preliminary screening for environmental monitoring and water treatment plants.
- College demonstrations of machine learning pipelines, data leakage prevention, and model evaluation.
- Decision support for water conservation and watershed management organizations.

---

## 📊 2. Genuine Dataset Collection & Schema

- **Dataset Name:** Water Potability Dataset
- **Source:** [Kaggle – Aditya Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Total Records:** 3,276 water samples
- **Total Columns:** 10 (9 input features + 1 target variable)
- **Duplicate Records:** 0

### Input Features Description

| # | Feature | Unit / Metric | WHO / Typical Standard | Description | Missing Values |
|---|---|---|---|---|---|
| 1 | `ph` | pH scale (0–14) | 6.5 – 8.5 | Acid-base balance of water | 491 (15.0%) |
| 2 | `Hardness` | mg/L | < 200 mg/L | Capacity to precipitate soap (calcium & magnesium) | 0 (0.0%) |
| 3 | `Solids` | ppm (TDS) | < 500 – 1000 ppm | Total Dissolved Solids in water | 0 (0.0%) |
| 4 | `Chloramines` | ppm | < 4 ppm | Chlorine + ammonia disinfectant residual | 0 (0.0%) |
| 5 | `Sulfate` | mg/L | < 250 mg/L | Dissolved mineral salts from natural deposits | 781 (23.8%) |
| 6 | `Conductivity` | μS/cm | < 400 μS/cm | Electrical conductivity from dissolved ionic substances | 0 (0.0%) |
| 7 | `Organic_carbon` | ppm | < 2 – 4 ppm | Total Organic Carbon (TOC) from decaying biomass | 0 (0.0%) |
| 8 | `Trihalomethanes` | μg/L | < 80 μg/L | Chlorination by-products | 162 (4.9%) |
| 9 | `Turbidity` | NTU | < 5 NTU | Measure of water clarity and light scattering | 0 (0.0%) |

### Target Variable
- **`Potability`**:
  - `0` = **Non-Potable** (1,998 records, 61.0%)
  - `1` = **Potable** (1,278 records, 39.0%)

---

## ⚙️ 3. Data Preprocessing & Leakage Prevention

1. **Missing Numerical Values:** Handled via **Median Imputation** (`SimpleImputer(strategy='median')`). The median is chosen over the mean because chemical features contain skewness and outliers.
2. **Stratified Split:** 80% training data (2,620 samples) and 20% testing data (656 samples) using `stratify=y` to preserve the 61:39 class ratio.
3. **Feature Scaling:** `StandardScaler` standardizes numerical features to zero mean and unit variance.
4. **Data Leakage Prevention:** The preprocessing pipeline is fitted **only** on `X_train`. The held-out `X_test` is transformed using the parameters learned from `X_train`, ensuring complete independence.

---

## 🤖 4. Model Development & Comparison

> **Note:** sections 4 to 6 describe the *original* results (before the changes listed in **Improvements (Anuj)** below). The numbers are reproduced by `train_model.py` as the "before" result.

Two distinct classification paradigms were trained using identical training data and evaluated on the same 656-sample held-out test set:

### Model A: Logistic Regression (Baseline)
- Linear classifier establishing a benchmark.
- **Result:** Fails to separate the non-linear feature space. Predicts all test samples as Class 0 (Non-Potable).
- Accuracy: **60.98%**, Precision: **0.0000**, Recall: **0.0000**, F1-Score: **0.0000**.

### Model B: Random Forest Classifier (Baseline Ensemble)
- Non-linear ensemble of 100 decision trees.
- Accuracy: **66.31%**, Precision: **0.6446**, Recall: **0.3047**, F1-Score: **0.4138**.

---

## ⚡ 5. Model Optimization (GridSearchCV)

Hyperparameter optimization was performed on the Random Forest pipeline using 5-fold cross-validation on the training data only:
- **Tuned Hyperparameters:**
  - `n_estimators`: `[100, 200]`
  - `max_depth`: `[None, 10, 20]`
  - `min_samples_split`: `[2, 5, 10]`
- **Best Hyperparameters Found:** `{'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 100}`
- **Best 5-Fold Cross-Validation Accuracy:** **67.29%**
- **Held-Out Test Set Performance:**
  - Test Accuracy: **66.16%**
  - Test Precision: **0.6417**
  - Test Recall: **0.3008**
  - Test F1-Score: **0.4096**

### Honest Performance Comparison

| Metric | Logistic Regression (Baseline) | Random Forest (Baseline) | Random Forest (Optimized) |
|---|---|---|---|
| **Accuracy** | 60.98% | **66.31%** | 66.16% |
| **Precision** | 0.0000 | **0.6446** | 0.6417 |
| **Recall** | 0.0000 | **0.3047** | 0.3008 |
| **F1-Score** | 0.0000 | **0.4138** | 0.4096 |
| **5-Fold CV Accuracy** | N/A | N/A | **67.29%** |

> **Evaluation Takeaway:** Cross-validation stabilized tree depth (`max_depth: 20`) and split regularization (`min_samples_split: 5`). Baseline Random Forest and Tuned Random Forest perform very similarly on the held-out test set, demonstrating that default parameters were already robust, while both dramatically outperform the linear baseline.

---

## 📐 6. Confusion Matrix Breakdown

### Logistic Regression
```
                Predicted 0    Predicted 1
Actual 0 (TN/FP)    400             0
Actual 1 (FN/TP)    256             0
```
*The linear model suffers from zero sensitivity (recall = 0), simply guessing the majority class.*

### Random Forest (Selected Model)
```
                Predicted 0    Predicted 1
Actual 0 (TN/FP)    357            43
Actual 1 (FN/TP)    179            77
```
- **True Negatives (TN):** 357 non-potable samples correctly identified.
- **False Positives (FP):** 43 non-potable samples incorrectly classified as potable.
- **False Negatives (FN):** 179 potable samples misclassified as non-potable.
- **True Positives (TP):** 77 potable samples correctly identified.

---

## ✨ Improvements (Anuj)

This section documents the changes added in the `anuj-improvements` branch. All numbers below were produced by running `python train_model.py` (same 80/20 stratified split, `random_state=42`) and are stored in `model_results.json`. The test set has 656 samples.

### Change 1: Imbalance-aware training and honest retuning (Syllabus: Unit III evaluation, Unit V tuning and cross-validation)

**Problem found:** the dataset is 61% non-potable. The original Logistic Regression only predicted the majority class (accuracy 60.98%, F1 0.00), and the "optimized" Random Forest found only 30% of the potable samples. It was also tuned for *accuracy*, which rewards guessing the majority class.

**What changed:**
- `class_weight='balanced'` for every model that supports it (Logistic Regression, Decision Tree, SVM, Random Forest).
- `GridSearchCV(scoring='f1')` with `StratifiedKFold(n_splits=5, shuffle=True)`, run on the training data only.
- ROC-AUC added next to accuracy, precision, recall, F1 and the confusion matrix.
- The original models are still trained in `train_model.py` so that the before/after comparison is reproducible.

**Before (original approach):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Original) | 60.98% | 0.0000 | 0.0000 | 0.0000 | 0.5481 |
| Random Forest (Default) | 66.31% | 0.6446 | 0.3047 | 0.4138 | 0.6411 |
| Random Forest (Tuned for Accuracy) | 66.16% | 0.6417 | 0.3008 | 0.4096 | 0.6491 |

**Random Forest, before vs after:**

| | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|---|---|
| Before (accuracy-tuned) | 66.16% | 0.6417 | 0.3008 | 0.4096 | 0.6491 | `[[357, 43], [179, 77]]` |
| After (balanced, F1-tuned) | 63.72% | 0.5425 | 0.4492 | 0.4915 | 0.6630 | `[[303, 97], [141, 115]]` |

**Reading the result honestly:** potable recall rises from 0.30 to 0.45 and F1 from 0.41 to 0.49, but accuracy falls from 66.2% to 63.7% because the model now labels more samples as potable. This is the normal precision/recall trade-off, not a free win.

### Change 2: Syllabus model bake-off (Syllabus: Unit III classification, Unit V SVM / ensembles / AdaBoost)

Seven classifiers use the same pipeline (median imputation, scaling, model), the same split and the same tuning method. Results are sorted by cross-validation F1 on the training data:

| Model | CV F1 (train) | Accuracy | Precision | Recall | F1 | ROC-AUC | class_weight | Best parameters |
|---|---|---|---|---|---|---|---|---|
| Support Vector Machine (RBF) | 0.5704 | 62.20% | 0.5159 | 0.5078 | 0.5118 | 0.6440 | balanced | C=1, gamma=scale |
| Random Forest | 0.5491 | 63.72% | 0.5425 | 0.4492 | 0.4915 | 0.6630 | balanced | max_depth=None, min_samples_split=10, n_estimators=200 |
| Decision Tree | 0.5348 | 60.06% | 0.4901 | 0.5820 | 0.5321 | 0.6297 | balanced | max_depth=None, min_samples_leaf=10 |
| K-Nearest Neighbors | 0.4644 | 60.82% | 0.4974 | 0.3750 | 0.4276 | 0.5811 | not supported | n_neighbors=3 |
| Logistic Regression | 0.4179 | 53.05% | 0.4217 | 0.5469 | 0.4762 | 0.5488 | balanced | C=0.01 |
| Naive Bayes (Gaussian) | 0.3379 | 61.43% | 0.5161 | 0.1875 | 0.2751 | 0.6090 | not supported | var_smoothing=1e-09 |
| AdaBoost | 0.1989 | 62.04% | 0.6061 | 0.0781 | 0.1384 | 0.5863 | not supported | learning_rate=1.0, n_estimators=200 |

- **Saved model:** Support Vector Machine (RBF) (Balanced, F1-tuned), chosen by the highest cross-validation F1 (0.5704) on the *training* data, so the test set was not used for model selection. It is stored in `water_model.joblib`.
- The top models are close to each other. The best test F1 is Decision Tree (0.5321) and the best test ROC-AUC is Random Forest (0.6630); no model is clearly better on this dataset.
- KNN, Naive Bayes and AdaBoost do not have a `class_weight` option, and they have the lowest recall. AdaBoost in particular finds only 8% of potable samples.
- Feature importance now uses permutation importance on the test set (works for any model, including the SVM).
- The **Model Performance & Comparison** tab shows the comparison table, F1 / ROC-AUC charts and all seven confusion matrices.

### Change 3: Unsupervised Analysis tab (Syllabus: Unit IV PCA, K-Means, hierarchical clustering, cluster evaluation)

A new **Unsupervised Analysis** page in the app. The labels are not used to build the clusters; they are only used afterwards for comparison.
- **PCA:** 2-D scatter coloured by potability and an explained-variance chart. The first two components explain only 26.0% of the variance and 8 of 9 components are needed for 90%; the two classes overlap almost completely.
- **K-Means:** elbow curve and silhouette score for k = 2 to 10. The silhouette score is low (best 0.082 at k = 2) and the elbow curve has no sharp bend. For k = 2 the cluster purity is 61.0%, which equals the majority-class share, and the Adjusted Rand Index is about 0.000.
- **Hierarchical clustering:** Ward dendrogram on a random sample (30 to 150 rows, default 60), with potable / non-potable leaves coloured, and cluster-vs-label agreement for a 2-cluster cut. On small samples this result is unstable, which the page points out.
- **Conclusion:** the features show no natural potable / non-potable grouping, which is consistent with the modest supervised scores.

### Other fixes
- Sidebar model name, feature-importance caption and the Academic Report tab text now read from `model_results.json`, so they cannot drift from the real results.
- The Dataset Analysis insight no longer says non-linear models are "necessary"; no model exceeds about 0.66 ROC-AUC.
- `scipy` added to `requirements.txt` (dendrogram).

---

## 📁 7. Project Structure

```
aquaguard-ai/
│
├── app.py                  # Streamlit web application with UI, predictor, and charts
├── train_model.py          # Preprocessing, model training, GridSearchCV, and evaluation
├── water_potability.csv    # Genuine Kaggle water potability dataset (3,276 records)
├── water_model.joblib      # Serialized Scikit-learn Pipeline (Imputer + Scaler + RF)
├── model_results.json      # Actual evaluated metrics, confusion matrices, and presets
├── requirements.txt        # Python library dependencies
├── README.md               # Project documentation and presentation guide
└── .gitignore              # Git ignore rules for clean repository
```

---

## 💻 8. Installation & Local Execution (Windows PowerShell)

Follow these exact steps in **Windows PowerShell**:

### Step 1: Open Terminal in Project Directory
```powershell
cd d:\AIML-PROJECT
```

### Step 2: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy warning, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

### Step 4: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Train and Evaluate Models (Generates artifacts)
```powershell
python train_model.py
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view AquaGuard AI!

---

## 🚀 9. GitHub Upload & Streamlit Community Cloud Deployment

### A. Push Code to GitHub
1. Open PowerShell in `d:\AIML-PROJECT`.
2. Initialize Git and commit your files:
```powershell
git init
git add .
git commit -m "Initial commit: AquaGuard AI Water Quality Classification System"
```
3. Create a new public repository on [GitHub](https://github.com/new) named `aquaguard-ai`.
4. Link and push your repository:
```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/aquaguard-ai.git
git push -u origin main
```

### B. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/aquaguard-ai`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**.
7. Streamlit Cloud will install dependencies from `requirements.txt` and launch your live web application!

---

## 🎓 10. 10-Minute College Presentation Guide

Use this script to deliver an outstanding college presentation:

- **Minute 0:00 – 1:30 (Title & Problem Definition):**  
  *"Good morning respected professors. Today I am presenting AquaGuard AI, a machine learning system for classifying water potability from physical and chemical indicators."*
- **Minute 1:30 – 3:00 (Dataset & Preprocessing):**  
  *"We used a genuine dataset of 3,276 water samples. We handled missing values in Sulfate and pH using median imputation to prevent outlier distortion, and fitted our pipeline only on the 80% training split to eliminate data leakage."*
- **Minute 3:00 – 5:00 (Algorithm Comparison & Key Discovery):**  
  *"We compared seven syllabus classifiers. Our original Logistic Regression scored 61% accuracy only by predicting non-potable every time, so we judged models by F1-score and ROC-AUC, not accuracy alone."*
- **Minute 5:00 – 6:30 (Optimization & Feature Importance):**  
  *"We used class weights and GridSearchCV with stratified 5-fold cross-validation scored on F1. The SVM had the best cross-validation F1. PCA and K-Means show the two classes overlap heavily, which explains the modest scores."*
- **Minute 6:30 – 8:30 (Live System Demonstration):**  
  *Open the Predictor tab. Click 'Load Known Potable Sample' -> Predict. Then click 'Load Known Non-Potable Sample' -> Predict. Point out the probability gauge, model badge, and academic disclaimer.*
- **Minute 8:30 – 10:00 (Conclusion & Viva Defense):**  
  *"AquaGuard AI shows how machine learning can serve as an automated preliminary screening tool while respecting water safety ethics. Thank you, I am ready for questions."*

---

## 🎯 11. Viva Questions & Model Answers

**Q1: Why did Logistic Regression fail with 0.00 recall?**  
*Answer:* The correlation between individual water features and potability is close to 0. A single linear hyperplane cannot partition the complex overlapping chemical parameters, so the algorithm minimized loss by classifying everything as the majority class (Class 0, 61%).

**Q2: What is data leakage, and how did you prevent it?**  
*Answer:* Data leakage occurs when test set statistics influence model training. We wrapped preprocessing in a Scikit-learn Pipeline so that the median imputer and standard scaler were fitted strictly on `X_train`.

**Q3: Why use Median Imputation instead of Mean?**  
*Answer:* Water parameters like Solids and Sulfate have wide spreads and high-value outliers. The median is a non-parametric statistic unaffected by extreme outliers.

---

## ⚠️ Academic Disclaimer

> **AquaGuard AI is an educational machine learning project. Its predictions reflect patterns learned from historical dataset labels and do not establish whether water is safe to drink. Laboratory testing and applicable drinking-water standards are required to assess actual water safety.**
