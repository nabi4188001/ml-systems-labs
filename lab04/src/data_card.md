# Data Card: UCI Adult (Census Income)

## 1. Source & License
- **Source**: UCI Machine Learning Repository, originally extracted from the 1994 US Census Bureau database by Ronny Kohavi and Barry Becker.
- **Access Method**: Loaded via `sklearn.datasets.fetch_openml(name='adult', version=2)`.
- **License**: Public Domain / CC0 (UCI Repository standard).

## 2. Size & Composition
- **Total Instances**: 48,842
- **Features**: 14 (6 continuous/numeric, 8 categorical)
- **Target**: Binary classification (`<=50K` or `>50K`)
- **Train/Test Split**: 34,165 training instances (after duplicate removal), 14,653 test instances.

## 3. Known Biases & Limitations
- **Demographic Skew**: The dataset is heavily skewed towards white males, reflecting historical demographics and systemic biases in the 1994 US workforce.
- **Target Imbalance**: The target variable is imbalanced (~76% `<=50K` vs ~24% `>50K`), which can cause models to default to predicting the majority class if not properly evaluated with metrics like F1-score.
- **Outdated Context**: The data is from 1994. Economic conditions, inflation, demographic shifts, and definitions of income brackets have changed significantly, limiting its real-world applicability for modern financial systems.
- **Proxy Variables**: Features like `race`, `sex`, and `native-country` can act as proxies for protected attributes, raising fairness concerns if the model is used for automated decision-making (e.g., loan approvals).

## 4. Privacy Concerns
- **Anonymization**: The dataset is aggregated and anonymized. It does not contain PII (Personally Identifiable Information) like names, exact addresses, or SSNs.
- **Re-identification Risk**: While direct identifiers are removed, the combination of granular categorical features (e.g., specific occupation, native country, race) and continuous features (e.g., exact capital gains/losses, fnlwgt) could theoretically pose a re-identification risk if cross-referenced with other public 1994 census records, though this risk is generally considered low for this specific extract.

## 5. Preprocessing Summary
- Missing values (originally denoted as `'?'`) were imputed using the mode for categorical features and median for numeric features.
- Numeric outliers were clipped to the 1st and 99th percentiles calculated strictly from the training set to prevent data leakage.
- Exact duplicate rows were removed from the training set only.