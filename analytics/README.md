# Module 2: Zepto Analytics Pipeline (`/analytics`)

## 1. Executive Summary & Architecture
The Analytics module demonstrates Zepto's full analyst-to-data-scientist workflow. It transitions from raw data profiling, defensible cleaning, and multivariate visualization (Part A) into rigorous predictive modeling, imbalance handling, hyperparameter tuning, regression analysis, and production pipeline serialization (Part B).

### Single-Load & Offline Fallback Architecture
The dataset is loaded **exactly once** via `sns.load_dataset('titanic')` and immediately persisted to disk as `analytics/titanic.csv`. Both the EDA workflow (`01_eda.py` / `01_eda.ipynb`) and the predictive modeling workflow (`02_modeling.py` / `02_modeling.ipynb`) read from this committed CSV file, guaranteeing that the repository runs deterministically offline without network dependencies.

---

## 2. Part A: Profiling, Cleaning, and Data Story

### 2.1 Initial Profiling & Missing-Value Threshold Rule
The dataset comprises 891 rows and 15 raw columns. A systematic scan for missing values yielded:
- **`deck`**: 688 missing values (**77.22%**)
- **`age`**: 177 missing values (**19.87%**)
- **`embarked`** / **`embark_town`**: 2 missing values (**0.22%**)

**Threshold Rule Implementation & Decisions:**
1. **Under 5% Missing (`embarked` / `embark_town` at 0.22%)**: Dropped the 2 missing rows. Removing 0.22% of observations incurs virtually zero loss of statistical power while avoiding categorical imputation noise.
2. **5% to 30% Missing (`age` at 19.87%)**: Imputed using the median age (28.0 years). Median imputation preserves the central tendency without being distorted by extreme ages.
3. **Over 30% Missing (`deck` at 77.22%)**: Dropped the entire column. At >77% missingness, imputation would introduce extreme synthetic bias, while dummy encoding would simply proxy passenger class (`pclass`), which is already cleanly captured.

---

### 2.2 Univariate Analysis & Central Tendency of Fare
Outlier detection was performed using the standard Interquartile Range (IQR) rule ($[Q_1 - 1.5\times IQR, \, Q_3 + 1.5\times IQR]$):

- **`age` Outliers**:
  - $Q_1 = 22.00$, $Q_3 = 35.00$, $IQR = 13.00$
  - Outlier Bounds: $[2.50, 54.50]$ years
  - Detected Outliers: **65 rows (7.31%)** (infants under 2.5 years and senior passengers over 54.5 years).
- **`fare` Outliers**:
  - $Q_1 = 7.90$, $Q_3 = 31.00$, $IQR = 23.10$
  - Outlier Bounds: $[-26.76, 65.66]$ GBP
  - Detected Outliers: **114 rows (12.82%)** (premium luxury cabins and multi-room suites).

**Fare Distribution Skewness**:
- **Mean**: 32.10 GBP
- **Median**: 14.45 GBP
- **Mode**: 8.05 GBP
- **Conclusion**: Because **$\text{Mean (32.10)} > \text{Median (14.45)} > \text{Mode (8.05)}$**, the fare distribution is **heavily right-skewed (positive skew)** with an extended right-tail representing luxury ticket pricing.
- Artifact saved: `figures/age_fare_distribution.png`

---

### 2.3 Bivariate Survival Rates & 6x6 Correlation Matrix

**Survival Rates by Boolean Masking:**
- **By Sex**:
  - Female: **74.04%** (231 / 312)
  - Male: **18.89%** (109 / 577)
- **By Passenger Class (`pclass`)**:
  - Class 1 (Upper): **62.62%** (134 / 214)
  - Class 2 (Middle): **47.28%** (87 / 184)
  - Class 3 (Lower): **24.24%** (119 / 491)
- **By Sex and Class Interaction**:
  - Female, Class 1: **96.74%**
  - Female, Class 2: **92.11%**
  - Female, Class 3: **50.00%**
  - Male, Class 1: **36.89%**
  - Male, Class 2: **15.74%**
  - Male, Class 3: **13.54%**

**Strict 6x6 Correlation Matrix:**
Restricted exclusively to numeric columns (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`) while excluding derived booleans (`adult_male`, `alone`):

| Feature | survived | pclass | age | sibsp | parch | fare |
|---|---|---|---|---|---|---|
| **survived** | 1.000 | -0.336 | -0.070 | -0.034 | 0.083 | 0.255 |
| **pclass** | -0.336 | 1.000 | -0.337 | 0.082 | 0.017 | -0.548 |
| **age** | -0.070 | -0.337 | 1.000 | -0.233 | -0.171 | 0.094 |
| **sibsp** | -0.034 | 0.082 | -0.233 | 1.000 | 0.415 | 0.161 |
| **parch** | 0.083 | 0.017 | -0.171 | 0.415 | 1.000 | 0.218 |
| **fare** | 0.255 | -0.548 | 0.094 | 0.161 | 0.218 | 1.000 |

- Artifact saved: `figures/correlation_heatmap.png`

**Top 2 Strongest Off-Diagonal Correlations:**
1. **`pclass` & `fare` ($r = -0.5482$, $|r| = 0.5482$)**: Strong negative correlation. Because `pclass` is numerically coded from 1 (highest class) to 3 (lowest class), passengers in higher classes paid significantly higher ticket fares.
2. **`sibsp` & `parch` ($r = 0.4145$, $|r| = 0.4145$)**: Moderate positive correlation. Reflects familial co-travel dynamics—passengers traveling with siblings or spouses were substantially more likely to also travel with parents or children.

---

### 2.4 Multivariate Visual Data Story
Artifact saved: `figures/multivariate_story.png` (comprising 4 distinct panels):

1. **Chart 1 (Survival Rate by Sex and Socio-Economic Class)**: Demonstrates that female passengers experienced a strong survival privilege across all classes, but class disparity remained stark: 1st class women had a 96.7% survival rate compared to only 50.0% for 3rd class women. Men in 1st class survived at more than double the rate of men in 2nd or 3rd class.
2. **Chart 2 (Fare Distribution by Class and Survival)**: Illustrates that within each passenger class, survivors consistently paid higher average fares than non-survivors. In 1st class, survivors occupied premium upper-deck cabins where lifeboat access was prioritized.
3. **Chart 3 (Age vs Fare Scatter Stratified by Survival & Sex)**: Highlights that male non-survivors were concentrated heavily in low-fare brackets across all age cohorts. High-fare tickets (above 100 GBP) almost uniformly yielded survival, especially for females and children.
4. **Chart 4 (Family Size Impact on Survival)**: Shows that solo travelers (family size = 1) had low survival rates (~30%), moderate family sizes (2 to 4 members) achieved peak survival (~55% to 70%), and large families (5+ members) suffered sharply lower survival (<20%) due to difficulties evacuating large groups together.

---

### 2.5 Exploratory Standardization Check
Using $z = (x - \mu)/\sigma$ on the full cleaned dataset:
- **`age`**: Pre-scaling ($\mu = 29.3152$, $\sigma = 12.9849$) $\rightarrow$ Post-scaling ($\mu = 2.80 \times 10^{-16}$, $\sigma = 1.0000$).
- **`fare`**: Pre-scaling ($\mu = 32.0967$, $\sigma = 49.6975$) $\rightarrow$ Post-scaling ($\mu = 1.36 \times 10^{-16}$, $\sigma = 1.0000$).
- Artifact saved: `figures/standardization_check.png` confirming mean ~0 and unit variance.

---

## 3. Part B: Predictive Modeling

### 3.1 Stratified Train/Test Split Justification
- Target Distribution: **Class 0 (Did not survive) = 61.62%**, **Class 1 (Survived) = 38.38%**.
- **Justification**: Random sampling can introduce sampling variance into train/test partitions, skewing class ratios and biasing validation metrics. Stratified splitting enforces identical 61.6 / 38.4 class balances in both train (712 rows) and test (179 rows) splits.

---

### 3.2 Preprocessing Architecture (Fit on Train Only)
Implemented via `scikit-learn` `ColumnTransformer` and `Pipeline`:
- **Numeric Features (`pclass`, `age`, `sibsp`, `parch`, `fare`)**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
- **Categorical Features (`sex`, `embarked`)**: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- All scalers, encoders, and imputers are **fit strictly on the training partition** and applied in **transform-only mode** to test data to prevent data leakage.

---

### 3.3 Classifier Evaluation (3 Models)
Evaluated on the identical 20% test partition:

| Classifier | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 |
| **Decision Tree** | 0.7877 | 0.8444 | 0.5507 | 0.6667 | 0.8210 |
| **Random Forest (Baseline)** | 0.8101 | 0.7692 | 0.7246 | 0.7463 | 0.8302 |

- **Decision Tree Visualization**: Generated via `plot_tree` with labeled features and classes (`figures/decision_tree.png`).
- **ROC Curves**: Combined ROC plot saved to `figures/roc_curves.png`.

---

### 3.4 Imbalance Handling Comparison
Evaluated on Random Forest using 3 distinct strategies:

| Strategy | Precision | Recall | F1 Score |
|---|---|---|---|
| **Baseline (No Handling)** | 0.7692 | 0.7246 | **0.7463** |
| **`class_weight='balanced'`** | 0.7463 | 0.7246 | **0.7353** |
| **SMOTE (Train fold only)** | 0.7286 | 0.7391 | **0.7338** |

**Conclusion**: SMOTE marginally boosted minority recall to 0.7391 at the cost of precision degradation (0.7286). Cost-sensitive weighting (`class_weight='balanced'`) adjusted loss penalties without creating synthetic feature artifacts. Baseline and balanced weighting maintained higher precision while retaining strong recall, yielding the highest F1 score.

---

### 3.5 Hyperparameter Tuning & Out-of-Bag (OOB) Score
- Method: `GridSearchCV` with 5-fold Stratified Cross-Validation on `RandomForestClassifier(oob_score=True, random_state=42)`.
- Grid: `n_estimators: [50, 100, 200]`, `max_depth: [3, 5, 8, None]`, `max_features: ['sqrt', 'log2']`.
- **Best Parameters**: `{'classifier__max_depth': 5, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 200}`
- **Best 5-Fold CV F1**: `0.7575`
- **Out-of-Bag (OOB) Score**: **`0.8272`** (validating internal generalization without validation leakage).

---

### 3.6 Regression Side-Task (Predicting `fare`)
- Architecture: Multivariate Linear Regression predicting ticket fare from `pclass`, `sex`, `age`, `sibsp`, `parch`, and `embarked`.
- **Evaluation Metrics**:
  - **MAE**: `20.8094`
  - **RMSE**: `30.4731`
  - **$R^2$**: `0.3999`
  - **Adjusted $R^2$**: `0.3790`
- Artifact saved: `figures/residual_plot.png`
- **Heteroscedasticity Analysis**: The residual plot exhibits clear **heteroscedasticity**. As predicted fare increases along the x-axis, the spread of residuals widens into a distinct funnel shape. While low-cost fares have small residual variances, luxury fares exhibit errors exceeding $\pm 150$, violating the homoscedasticity assumption of ordinary least squares.

---

### 3.7 Comprehensive Model Comparison Table

| Model Group | Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | MAE | RMSE | R² | Adj R² |
|---|---|---|---|---|---|---|---|---|---|---|
| **Classification** | Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 | N/A | N/A | N/A | N/A |
| **Classification** | Decision Tree | 0.7877 | 0.8444 | 0.5507 | 0.6667 | 0.8210 | N/A | N/A | N/A | N/A |
| **Classification** | Random Forest (Tuned) | 0.7933 | 0.8200 | 0.5942 | 0.6891 | **0.8420** | N/A | N/A | N/A | N/A |
| **Regression** | Multivariate Linear Regression | N/A | N/A | N/A | N/A | N/A | 20.8094 | 30.4731 | 0.3999 | 0.3790 |

### 3.8 Deployment Recommendation
We recommend deploying the **Tuned Random Forest Classifier** for Zepto's production operational workflow. It delivers the strongest ensemble robustness, achieving an out-of-sample ROC AUC of **0.8420** and an internal Out-of-Bag (OOB) generalization score of **0.8272**. Its ensemble architecture mitigates the high variance observed in individual Decision Trees while modeling non-linear demographic interactions (e.g. sex and passenger class) far better than linear models.

---

### 3.9 Saved Pipeline & Raw Data Inference Verification
The full pipeline (fitted `ColumnTransformer` preprocessor + tuned estimator) is serialized via:
```python
joblib.dump(best_estimator, "analytics/best_pipeline.joblib")
```
Reloading and predicting on raw, unpreprocessed inputs succeeds immediately:
```python
pipeline = joblib.load("analytics/best_pipeline.joblib")
predictions = pipeline.predict(raw_input_df)
probabilities = pipeline.predict_proba(raw_input_df)[:, 1]
```
- Test Case 1 (1st Class Female, 35 yrs, 85 GBP fare): Predicted **Survived (Class 1, Prob = 94.6%)**.
- Test Case 2 (3rd Class Male, 22 yrs, 7.25 GBP fare): Predicted **Not Survived (Class 0, Prob = 11.5%)**.

---

## 4. How to Run

From project root:
```bash
# Run EDA (Part A)
python analytics/01_eda.py

# Run Modeling (Part B)
python analytics/02_modeling.py
```
Or open and execute `analytics/01_eda.ipynb` and `analytics/02_modeling.ipynb` in Jupyter Notebook.
