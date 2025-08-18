# 🧠 Machine Learning Model Performance Dashboard

> **Baseline Results Comparison** | *Updated: August 2025*

---

## 📊 Model Performance Summary

| Dataset | 🔍 Logistic Regression | ⚡ Linear SVM | 🎯 MultinomialNB | 🔄 ComplementNB | 🌲 Random Forest | 🚀 Gradient Boosting |
|---------|------------------------|---------------|-------------------|------------------|-------------------|-----------------------|
| 🔴 **REAL/FAKE** | **94%** 🟢<br/>*Prec: .xx \| Rec: 0.xx* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* |
| 🟡 **WELFake** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* |
| 🟣 **LIAR** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* |
| 🟠 **FEVER** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* |
| 🟢 **Indian** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* |
| ⚫ **Merged** | **58-65%** 🟡<br/>*(drop observed)* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* |

---

## 🎯 Performance Legend

| Score Range | Status | Indicator | Description |
|-------------|---------|-----------|-------------|
| **90%+** | Excellent | 🟢 | Outstanding performance |
| **70-89%** | Good | 🟡 | Solid performance |
| **<70%** | Needs Work | 🔴 | Requires improvement |
| **—** | Pending | ⏳ | Results in progress |

---

## 📈 Current Highlights

### 🏆 **Best Performer**
- **Dataset**: REAL/FAKE
- **Model**: Logistic Regression
- **Accuracy**: **94%** 🟢
- **Status**: Excellent baseline established

### ⚠️ **Attention Needed**
- **Dataset**: Merged
- **Current Range**: 58-65%
- **Issue**: Performance drop observed
- **Next Steps**: Feature engineering & hyperparameter tuning

---

## 🔄 Experiment Pipeline Status

```
[████████████████████████████████████████] 100% REAL/FAKE (Logistic Regression)
[██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  25% WELFake
[██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  25% LIAR
[██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  25% FEVER
[██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  25% Indian
[████████████████████████████████████████] 100% Merged (Logistic Regression)
```

---

## 📋 Dataset Information

| Dataset | Description | Size | Type |
|---------|-------------|------|------|
| 🔴 **REAL/FAKE** | Binary classification for news authenticity | ~9K samples | News Articles |
| 🟡 **WELFake** | Fake news detection dataset | ~72K samples | News Articles |
| 🟣 **LIAR** | Political fact-checking dataset | ~12K samples | Political Claims |
| 🟠 **FEVER** | Fact extraction and verification | ~185K samples | Wikipedia Claims |
| 🟢 **Indian** | Indian news classification | ~4K samples | News Articles |
| ⚫ **Merged** | Combined dataset | ~282K samples | Mixed Sources |

---

## 🛠 Model Configurations

### 🔍 **Logistic Regression**
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='liblinear'
)
```

### ⚡ **Linear SVM**
```python
LinearSVC(
    random_state=42,
    max_iter=1000
)
```

### 🎯 **MultinomialNB**
```python
MultinomialNB(
    alpha=1.0
)
```

### 🔄 **ComplementNB**
```python
ComplementNB(
    alpha=1.0
)
```

### 🌲 **Random Forest**
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

### 🚀 **Gradient Boosting**
```python
GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)
```

---

## 📝 Notes & Observations

> **Key Findings:**
> - ✅ REAL/FAKE dataset shows excellent performance with Logistic Regression (94%)
> - ⚠️ Merged dataset experiencing significant performance degradation (58-65%)
> - 🔄 Remaining experiments in progress across all model types

> **Next Steps:**
> 1. Complete remaining baseline experiments
> 2. Analyze feature importance for top performers
> 3. Investigate merged dataset performance issues
> 4. Implement cross-validation for robust evaluation

---

## 📊 Quick Stats

- **Total Datasets**: 6
- **Total Models**: 6  
- **Experiments Complete**: 2/36 (5.6%)
- **Best Accuracy**: 94% (REAL/FAKE + LogReg)
- **Avg Completion Time**: ~2 hours per experiment

---

*Generated on August 17, 2025 | Experiment ID: baseline-001*