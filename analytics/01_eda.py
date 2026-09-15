import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
CSV_PATH = os.path.join(os.path.dirname(__file__), 'titanic.csv')

# Step 1: Load and save offline fallback
print("="*80)
print("PART A: PROFILING, CLEANING, AND THE DATA STORY")
print("="*80)
print("1. Loading Titanic dataset from Seaborn...")
df_raw = sns.load_dataset('titanic')
df_raw.to_csv(CSV_PATH, index=False)
print(f"Dataset successfully saved to offline fallback: {CSV_PATH}")

# Profiling
print("\n--- Dataset Info ---")
df_raw.info()
print("\n--- Dataset Describe ---")
print(df_raw.describe())
print(f"\n--- Dataset Shape: {df_raw.shape} ---")

# Step 2: Missing value profiling and threshold rule
print("\n2. Missing Value Analysis:")
missing_counts = df_raw.isnull().sum()
missing_pct = (missing_counts / len(df_raw)) * 100
missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Percentage (%)': missing_pct.round(2)})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False)
print(missing_df)

print("\n--- Missing Value Handling Decisions ---")
print("Threshold Rule Applied:")
print("- Under 5% missing: 'embarked' (0.22%, 2 rows) and 'embark_town' (0.22%, 2 rows) -> Drop these 2 rows.")
print("- 5% to 30% missing: 'age' (19.87%, 177 rows) -> Impute using median (28.0 years) to maintain robust central tendency without skew.")
print("- Over 30% missing: 'deck' (77.22%, 688 rows) -> Drop column entirely. With over 77% missingness, imputation would introduce heavy synthetic bias and unreliable imputation noise.")

# Apply cleaning for EDA
df_clean = df_raw.copy()
# Drop deck column
df_clean = df_clean.drop(columns=['deck'])
# Drop rows with missing embarked/embark_town (<5%)
df_clean = df_clean.dropna(subset=['embarked']).copy()
# Impute age with median for EDA
age_median = df_clean['age'].median()
df_clean['age'] = df_clean['age'].fillna(age_median)
print(f"\nCleaned EDA DataFrame Shape: {df_clean.shape}")

# Step 3: Univariate analysis (age and fare)
print("\n3. Univariate Analysis & IQR Outlier Detection:")

def compute_iqr_outliers(series, name):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    print(f"\n{name} Outlier Analysis (IQR Rule):")
    print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)")
    return len(outliers), lower_bound, upper_bound

age_outliers, age_low, age_high = compute_iqr_outliers(df_clean['age'], "Age")
fare_outliers, fare_low, fare_high = compute_iqr_outliers(df_clean['fare'], "Fare")

fare_mean = df_clean['fare'].mean()
fare_median = df_clean['fare'].median()
fare_mode = df_clean['fare'].mode()[0]
print(f"\nFare Central Tendency:")
print(f"  Mean:   {fare_mean:.2f}")
print(f"  Median: {fare_median:.2f}")
print(f"  Mode:   {fare_mode:.2f}")
print("  Ordering: Mean (32.10) > Median (14.45) > Mode (8.05)")
print("  Skewness Conclusion: The Fare distribution is strongly RIGHT-SKEWED (positive skew) with a substantial right tail of premium luxury ticket fares.")

# Plot univariate distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df_clean['age'], kde=True, ax=axes[0, 0], color='teal')
axes[0, 0].set_title('Age Distribution (Histogram & KDE)')
sns.boxplot(x=df_clean['age'], ax=axes[0, 1], color='cyan')
axes[0, 1].set_title('Age Boxplot (IQR Outliers)')

sns.histplot(df_clean['fare'], kde=True, ax=axes[1, 0], color='crimson')
axes[1, 0].set_title('Fare Distribution (Histogram & KDE)')
sns.boxplot(x=df_clean['fare'], ax=axes[1, 1], color='lightcoral')
axes[1, 1].set_title('Fare Boxplot (IQR Outliers)')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'age_fare_distribution.png'), dpi=300)
plt.close()
print("Saved: figures/age_fare_distribution.png")

# Step 4: Bivariate analysis (survival rate via boolean masking)
print("\n4. Bivariate Analysis (Survival Rates via Boolean Masking):")
rate_female = df_clean[df_clean['sex'] == 'female']['survived'].mean()
rate_male = df_clean[df_clean['sex'] == 'male']['survived'].mean()
print(f"  (a) Survival Rate by Sex:")
print(f"      Female: {rate_female:.4f} ({rate_female*100:.2f}%)")
print(f"      Male:   {rate_male:.4f} ({rate_male*100:.2f}%)")

print(f"  (b) Survival Rate by Pclass:")
for p in [1, 2, 3]:
    rate_p = df_clean[df_clean['pclass'] == p]['survived'].mean()
    print(f"      Class {p}: {rate_p:.4f} ({rate_p*100:.2f}%)")

print(f"  (c) Survival Rate by Sex & Pclass Together:")
for s in ['female', 'male']:
    for p in [1, 2, 3]:
        rate_sp = df_clean[(df_clean['sex'] == s) & (df_clean['pclass'] == p)]['survived'].mean()
        print(f"      {s.capitalize()} in Class {p}: {rate_sp:.4f} ({rate_sp*100:.2f}%)")

# Correlation matrix restricted to EXACTLY 6 columns
corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df_clean[corr_cols].corr()
print(f"\n--- 6x6 Correlation Matrix ---")
print(corr_matrix.round(3))

# Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title('Correlation Matrix (Strict 6-Feature Subset)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=300)
plt.close()
print("Saved: figures/correlation_heatmap.png")

# Rank off-diagonal correlations
corr_pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        col1, col2 = corr_cols[i], corr_cols[j]
        val = corr_matrix.loc[col1, col2]
        corr_pairs.append((col1, col2, val, abs(val)))

corr_pairs.sort(key=lambda x: x[3], reverse=True)
print("\nTop 2 Strongest Off-Diagonal Correlations:")
for idx, (c1, c2, val, abs_val) in enumerate(corr_pairs[:2], 1):
    print(f"  {idx}. {c1} & {c2}: r = {val:.4f} (|r| = {abs_val:.4f})")
print("Interpretation of Top 2 Correlations:")
print(f"1. {corr_pairs[0][0]} & {corr_pairs[0][1]} (r = {corr_pairs[0][2]:.4f}): Strong negative correlation. Passenger class is inversely coded (1 is highest class, 3 is lowest), so 1st class passengers paid substantially higher fares.")
print(f"2. {corr_pairs[1][0]} & {corr_pairs[1][1]} (r = {corr_pairs[1][2]:.4f}): Moderate positive correlation reflecting family travel patterns—passengers with more siblings/spouses also tended to travel with more parents/children.")

# Step 5: Multivariate Data Story (4 Distinct Charts)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Chart 1: Survival rate by Sex and Class
sns.barplot(x='pclass', y='survived', hue='sex', data=df_clean, palette={'female': '#d62728', 'male': '#1f77b4'}, ax=axes[0, 0])
axes[0, 0].set_title('Chart 1: Survival Rate by Sex and Socio-Economic Class')
axes[0, 0].set_ylabel('Survival Probability')
axes[0, 0].set_xlabel('Passenger Class (1=Upper, 2=Middle, 3=Lower)')

# Chart 2: Fare distribution across classes by survival
sns.boxplot(x='pclass', y='fare', hue='survived', data=df_clean, palette='Set2', showfliers=False, ax=axes[0, 1])
axes[0, 1].set_title('Chart 2: Fare Paid by Class and Survival Outcome (Outliers Hidden)')
axes[0, 1].set_ylabel('Fare (GBP)')
axes[0, 1].set_xlabel('Passenger Class')

# Chart 3: Age vs Fare scatter colored by survival
sns.scatterplot(x='age', y='fare', hue='survived', style='sex', data=df_clean, alpha=0.7, palette='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Chart 3: Age vs Fare Scatter Stratified by Survival & Sex')
axes[1, 0].set_ylabel('Fare (GBP)')
axes[1, 0].set_xlabel('Age (Years)')

# Chart 4: Family size impact on survival
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
sns.pointplot(x='family_size', y='survived', data=df_clean, color='purple', ax=axes[1, 1])
axes[1, 1].set_title('Chart 4: Family Size vs Survival Rate')
axes[1, 1].set_ylabel('Survival Probability')
axes[1, 1].set_xlabel('Total Family Size (Self + SibSp + Parch)')

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'multivariate_story.png'), dpi=300)
plt.close()
print("Saved: figures/multivariate_story.png")

# Step 6: Exploratory Standardization Check
print("\n6. Exploratory Standardization Check (z = (x - mean)/std):")
age_orig_mean, age_orig_std = df_clean['age'].mean(), df_clean['age'].std()
fare_orig_mean, fare_orig_std = df_clean['fare'].mean(), df_clean['fare'].std()

df_clean['age_zscore'] = (df_clean['age'] - age_orig_mean) / age_orig_std
df_clean['fare_zscore'] = (df_clean['fare'] - fare_orig_mean) / fare_orig_std

print(f"  Age  Before: Mean = {age_orig_mean:.4f}, Std = {age_orig_std:.4f}")
print(f"  Age  After:  Mean = {df_clean['age_zscore'].mean():.4e}, Std = {df_clean['age_zscore'].std():.4f}")
print(f"  Fare Before: Mean = {fare_orig_mean:.4f}, Std = {fare_orig_std:.4f}")
print(f"  Fare After:  Mean = {df_clean['fare_zscore'].mean():.4e}, Std = {df_clean['fare_zscore'].std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.kdeplot(df_clean['age_zscore'], label='Age (Z-Score)', ax=axes[0], color='teal', fill=True)
sns.kdeplot(df_clean['fare_zscore'], label='Fare (Z-Score)', ax=axes[0], color='crimson', fill=True)
axes[0].set_title('Standardized Z-Score Distributions (Mean~0, Std~1)')
axes[0].legend()

# Bar comparison of means & stds
summary_stats = pd.DataFrame({
    'Metric': ['Mean (Before)', 'Std (Before)', 'Mean (After Z-score)', 'Std (After Z-score)'],
    'Age': [age_orig_mean, age_orig_std, df_clean['age_zscore'].mean(), df_clean['age_zscore'].std()],
    'Fare': [fare_orig_mean, fare_orig_std, df_clean['fare_zscore'].mean(), df_clean['fare_zscore'].std()]
})
axes[1].axis('off')
table = axes[1].table(cellText=summary_stats.round(4).values, colLabels=summary_stats.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
axes[1].set_title('Standardization Verification Summary Table')

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'standardization_check.png'), dpi=300)
plt.close()
print("Saved: figures/standardization_check.png")

print("\nEDA and Data Profiling completed successfully!")
