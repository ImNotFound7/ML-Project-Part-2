# Kaggle Competitions Repository

This repository contains solutions for two Kaggle machine learning competitions:

1. **COPD Risk Classification Challenge** - Binary classification with medical domain
2. **Signal Cluster Classification Challenge** - Multiclass classification with geometric patterns

---

## 🎯 Projects Overview

### 1. COPD Risk Classification Challenge

**Objective:** Predict whether patients have COPD risk based on 20 clinical features

**Dataset:**
- Training: 44,553 samples, 20 features, 36.7% positive class
- Test: 11,139 samples
- Metric: F1 Score

**Results:**
- Baseline F1: 0.689
- Final F1: 0.7339 (+6.52% improvement)
- Best Model: Stacking Ensemble (LR + SVM + MLP)

**Key Techniques:**
- Advanced feature engineering (20 → 68 features)
- SMOTE for class imbalance handling
- Threshold optimization (not default 0.5)
- Hyperparameter tuning
- Stacking ensemble with meta-learner

**Leaderboard Performance:**
- Logistic Regression: 0.711 (threshold: 0.495)
- SVM (RBF): 0.722 (threshold: 0.320)
- Neural Network: 0.736 (threshold: 0.145)
- **Stacking Ensemble: 0.744 (threshold: 0.470)** ← Best

---

### 2. Signal Cluster Classification Challenge

**Objective:** Predict personality cluster (3 classes) from 2D signal features

**Dataset:**
- Training: 1,444 samples, 2 features, 3 classes
- Test: 362 samples
- Metric: Macro F1 Score

**Results:**
- Final Macro F1: 0.989
- Best Model: SVM with RBF Kernel

**Key Techniques:**
- Geometric feature engineering (polynomial, radial, angular)
- KMeans cluster distances
- Kernel-based SVM modeling
- 10-fold stratified cross-validation
- Leaderboard-driven hyperparameter tuning

**Model Performance:**
- MLP Neural Network (CV): 0.9858 → Leaderboard: 0.985
- **SVM RBF (CV): 0.9833 → Leaderboard: 0.989** ← Best
- Stacking Ensemble (CV): 0.9810 → Leaderboard: 0.977

---

## 📧 Authors

- **Team: Shivek Ranjan
   - Shivek Ranjan
   - Navish
   - Digvijaysinh Pawar

---


