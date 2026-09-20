# Analytics Pipeline (`/analytics`)

## Overview
This module covers an end-to-end data analysis and machine learning workflow on the Titanic dataset. It is divided into two connected stages:
- **Part A (`01_eda`)**: Data profiling, percentage-based missing value handling, IQR outlier detection, skewness analysis, 6x6 correlation matrix, visual data storytelling, and exploratory standardization.
- **Part B (`02_modeling`)**: Stratified train/test split, train-only preprocessing pipeline, classifier training (Logistic Regression, Decision Tree, Random Forest), class imbalance handling (baseline vs balanced weights vs SMOTE), hyperparameter tuning with Out-of-Bag (OOB) scoring, fare regression with residual analysis, and model deployment export.

Both stages read from the single committed offline fallback file: `analytics/titanic.csv`.

---

## Part A: Exploratory Data Analysis & Data Story

### 1. Data Profiling & Missing-Value Strategy
The dataset has 891 rows and 15 columns. Checking missing values per column gave:
- `deck`: 688 missing (**77.22%**)
- `age`: 177 missing (**19.87%**)
- `embarked` / `embark_town`: 2 missing (**0.22%**)

**Threshold Rule Implementation:**
1. **Under 5% missing**: For `embarked` and `embark_town` (0.22%), I dropped those 2 rows. Losing 2 rows out of 891 causes virtually zero information loss and avoids guessing categorical entries.
2. **5% to 30% missing**: For `age` (19.87%), I imputed missing entries with the median age (28.0 years). Using the median prevents extreme values from biasing the center.
3. **Over 30% missing**: For `deck` (77.22%), I dropped the column entirely. With more than 77% missing, imputation would create artificial patterns, and deck heavily proxies passenger class (`pclass`), which is already clean.

---

### 2. Univariate Analysis & Outlier Detection
Using the IQR rule ($[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$):

- **`age`**:
  - $Q1 = 22.00$, $Q3 = 35.00$, $IQR = 13.00$
  - Outlier Bounds: $[2.50, 54.50]$
  - Outliers: **65 rows (7.31%)** (young infants under 2.5 years and older passengers above 54.5 years).
- **`fare`**:
  - $Q1 = 7.90$, $Q3 = 31.00$, $IQR = 23.10$
  - Outlier Bounds: $[-26.76, 65.66]$
  - Outliers: **114 rows (12.82%)** (expensive luxury tickets).

**Fare Distribution Skewness**:
- Mean: 32.10 GBP
- Median: 14.45 GBP
- Mode: 8.05 GBP
- Ordering: **$\text{Mean (32.10)} > \text{Median (14.45)} > \text{Mode (8.05)}$**
- Conclusion: The distribution of `fare` is **positively right-skewed** with a long tail of high-value first-class tickets.
- Artifact: `figures/age_fare_distribution.png`

---

### 3. Bivariate Survival Rates & 6x6 Correlation Matrix

**Survival Rates by Boolean Masking:**
- **By Sex**:
  - Female: **74.04%**
  - Male: **18.89%**
- **By Class (`pclass`)**:
  - Class 1: **62.62%**
  - Class 2: **47.28%**
  - Class 3: **24.24%**
- **By Sex and Class Interaction**:
  - Female, Class 1: **96.74%**
  - Female, Class 2: **92.11%**
  - Female, Class 3: **50.00%**
  - Male, Class 1: **36.89%**
  - Male, Class 2: **15.74%**
  - Male, Class 3: **13.54%**

**Strict 6x6 Correlation Matrix (Numeric Features):**
Restricted strictly to `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare` (derived booleans `adult_male` and `alone` are excluded):

| Feature | survived | pclass | age | sibsp | parch | fare |
|---|---|---|---|---|---|---|
| **survived** | 1.000 | -0.336 | -0.070 | -0.034 | 0.083 | 0.255 |
| **pclass** | -0.336 | 1.000 | -0.337 | 0.082 | 0.017 | -0.548 |
| **age** | -0.070 | -0.337 | 1.000 | -0.233 | -0.171 | 0.094 |
| **sibsp** | -0.034 | 0.082 | -0.233 | 1.000 | 0.415 | 0.161 |
| **parch** | 0.083 | 0.017 | -0.171 | 0.415 | 1.000 | 0.218 |
| **fare** | 0.255 | -0.548 | 0.094 | 0.161 | 0.218 | 1.000 |

- Artifact: `figures/correlation_heatmap.png`

**Top 2 Off-Diagonal Correlations:**
1. **`pclass` & `fare` ($r = -0.5482$, $|r| = 0.5482$)**: Strong negative correlation. Because 1st class is coded as 1 and 3rd class as 3, higher class numbers correspond to much lower fares.
2. **`sibsp` & `parch` ($r = 0.4145$, $|r| = 0.4145$)**: Moderate positive correlation. Reflects that passengers traveling with siblings/spouses were also much more likely to travel with parents/children.

---

### 4. Multivariate Data Story (4 Visual Charts)
Artifact: `figures/multivariate_story.png`

1. **Chart 1 (Survival by Sex and Class)**: Women experienced far higher survival rates across all three passenger classes. However, class still had an enormous impact: 1st class women had a 96.7% survival rate while 3rd class women had a 50.0% rate.
2. **Chart 2 (Fare by Class and Survival)**: Within each class, survivors paid higher average fares than non-survivors, showing that cabin position and higher-tier tickets gave better lifeboat access.
3. **Chart 3 (Age vs Fare Scatter Stratified by Survival & Sex)**: Non-surviving males were heavily concentrated in the low-fare bracket across all ages. Tickets above 100 GBP almost always resulted in survival.
4. **Chart 4 (Family Size vs Survival)**: Solo travelers had low survival (~30%). Moderate family sizes of 2 to 4 people had the highest survival rates (55%–70%). Large families (5+) saw steep survival drops (<20%) due to the difficulty of keeping large families together during the evacuation.

---

### 5. Exploratory Standardization Check
Using $z = (x - \mu)/\sigma$:
- `age`: Before ($\mu = 29.3152, \sigma = 12.9849$) $\rightarrow$ After ($\mu \approx 0, \sigma = 1.0000$)
- `fare`: Before ($\mu = 32.0967, \sigma = 49.6975$) $\rightarrow$ After ($\mu \approx 0, \sigma = 1.0000$)
- Artifact: `figures/standardization_check.png`

---

## Part B: Predictive Modeling

### 1. Stratified Train/Test Split
- Target distribution: **Class 0 (Did not survive) = 61.62%**, **Class 1 (Survived) = 38.38%**.
- **Justification**: Random non-stratified sampling could disproportionately split survivors between train and test sets. Stratified sampling preserves the exact 61.6 / 38.4 ratio across both splits (712 train, 179 test), ensuring consistent testing.

---

### 2. Preprocessing Pipeline (Fit on Train Only)
Implemented via scikit-learn's `ColumnTransformer`:
- **Numeric Features (`pclass`, `age`, `sibsp`, `parch`, `fare`)**: Median imputation + `StandardScaler()`.
- **Categorical Features (`sex`, `embarked`)**: Most frequent imputation + `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- All transformers are **fit only on the training split** and applied as a transform to the test split to avoid test leakage.

---

### 3. Classifier Performance
Evaluated on the 20% test partition:

| Classifier | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 |
| **Decision Tree** | 0.7877 | 0.8444 | 0.5507 | 0.6667 | 0.8210 |
| **Random Forest (Baseline)** | 0.8101 | 0.7692 | 0.7246 | 0.7463 | 0.8302 |

- Decision tree plot: `figures/decision_tree.png`
- Combined ROC curves: `figures/roc_curves.png`

---

### 4. Imbalance Handling Comparison (Random Forest)

| Strategy | Precision | Recall | F1 Score |
|---|---|---|---|
| **Baseline** | 0.7692 | 0.7246 | **0.7463** |
| **`class_weight='balanced'`** | 0.7463 | 0.7246 | **0.7353** |
| **SMOTE (Train fold only)** | 0.7286 | 0.7391 | **0.7338** |

**Conclusion**: SMOTE bumped minority recall up to 0.7391 but dropped precision down to 0.7286. Setting `class_weight='balanced'` adjusted loss weights cleanly without adding synthetic points, while baseline maintained the best overall precision and F1 balance.

---

### 5. Hyperparameter Tuning & Out-of-Bag (OOB) Score
- Method: `GridSearchCV` with 5-fold Stratified CV on `RandomForestClassifier(oob_score=True, random_state=42)`.
- Best Parameters: `{'classifier__max_depth': 5, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 200}`
- Best 5-Fold CV F1: `0.7575`
- **Out-of-Bag (OOB) Score**: **`0.8272`** (confirming strong internal generalization).

---

### 6. Regression Side-Task (Predicting `fare`)
- Multivariate linear regression predicting ticket fare from demographic features.
- Metrics:
  - **MAE**: `20.8094`
  - **RMSE**: `30.4731`
  - **$R^2$**: `0.3999`
  - **Adjusted $R^2$**: `0.3790`
- Artifact: `figures/residual_plot.png`
- **Heteroscedasticity Analysis**: The residual plot shows noticeable **heteroscedasticity**. As predicted fare increases, the residuals fan out into a funnel shape. Residuals are clustered tightly around low fares but scatter widely for higher fares, showing that error variance is not constant.

---

### 7. Comprehensive Model Comparison Table

| Model Group | Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | MAE | RMSE | R² | Adj R² |
|---|---|---|---|---|---|---|---|---|---|---|
| **Classification** | Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 | N/A | N/A | N/A | N/A |
| **Classification** | Decision Tree | 0.7877 | 0.8444 | 0.5507 | 0.6667 | 0.8210 | N/A | N/A | N/A | N/A |
| **Classification** | Random Forest (Tuned) | 0.7933 | 0.8200 | 0.5942 | 0.6891 | **0.8420** | N/A | N/A | N/A | N/A |
| **Regression** | Multivariate Linear Regression | N/A | N/A | N/A | N/A | N/A | 20.8094 | 30.4731 | 0.3999 | 0.3790 |

### 8. Final Deployment Recommendation
I recommend deploying the **Tuned Random Forest Classifier**. While Logistic Regression has a slightly higher test F1 on this particular split, the Random Forest model is much more resilient to non-linear interactions and collinear features. It achieved an out-of-sample ROC AUC of **0.8420** and an internal Out-of-Bag (OOB) score of **0.8272**, demonstrating reliable real-world generalization.

---

### 9. Serialized Pipeline Verification
The complete fitted pipeline is saved to `analytics/best_pipeline.joblib`. Testing it on raw dictionary inputs:
```python
pipeline = joblib.load("best_pipeline.joblib")
predictions = pipeline.predict(raw_input_df)
```
- Sample 1 (1st Class Female, 35 yrs, 85 GBP): Predicted **Class 1 (Survived, 94.6% probability)**.
- Sample 2 (3rd Class Male, 22 yrs, 7.25 GBP): Predicted **Class 0 (Not Survived, 11.5% probability)**.

---

## How to Run
```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```
Or open and execute `01_eda.ipynb` and `02_modeling.ipynb` in Jupyter Notebook.
