import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

RANDOM_STATE = 42
N_JOBS = 4
N_FOLDS = 10 
SEED = RANDOM_STATE

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
OUTPUT_PREFIX = 'submission_improved'

def advanced_feature_engineering(df):
    """Enhanced feature engineering with domain knowledge"""
    df = df.copy()
    
    # Basic ratios
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    df['waist_height_ratio'] = df['waist_circumference_cm'] / df['height_cm']
    df['waist_weight_ratio'] = df['waist_circumference_cm'] / df['weight_kg']
    
    # Blood pressure features (critical for COPD)
    df['pulse_pressure'] = df['bp_systolic'] - df['bp_diastolic']
    df['map'] = df['bp_diastolic'] + (df['pulse_pressure'] / 3)
    df['bp_product'] = df['bp_systolic'] * df['bp_diastolic']
    df['bp_ratio'] = df['bp_systolic'] / (df['bp_diastolic'] + 1e-6)
    
    # Comprehensive cholesterol features
    df['chol_hdl_ratio'] = df['total_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-6)
    df['non_hdl_chol'] = df['total_cholesterol'] - df['hdl_cholesterol']
    df['atherogenic_index'] = np.log10(df['triglycerides'] / (df['hdl_cholesterol'] + 1e-6))
    
    # Liver enzyme ratios (inflammation markers)
    df['ast_alt_ratio'] = df['ast_enzyme_level'] / (df['alt_enzyme_level'] + 1e-6)
    df['ggt_alt_ratio'] = df['ggt_enzyme_level'] / (df['alt_enzyme_level'] + 1e-6)
    df['ggt_ast_ratio'] = df['ggt_enzyme_level'] / (df['ast_enzyme_level'] + 1e-6)
    df['liver_score'] = df['ast_enzyme_level'] + df['alt_enzyme_level'] + df['ggt_enzyme_level']
    
    # Kidney and blood features
    df['creatinine_hb_ratio'] = df['serum_creatinine'] / (df['hemoglobin_level'] + 1e-6)
    df['kidney_blood_score'] = df['serum_creatinine'] * df['hemoglobin_level']
    
    # Protein markers
    df['protein_positive'] = (df['urine_protein_level'] > 0).astype(int)
    df['protein_squared'] = df['urine_protein_level'] ** 2
    
    # Age processing
    df['age_numeric'] = df['age_group'].astype(float)
    df['age_squared'] = df['age_numeric'] ** 2
    df['age_bmi_interaction'] = df['age_numeric'] * df['bmi']
    
    # Vision and hearing
    df['vision_diff'] = np.abs(df['vision_left'] - df['vision_right'])
    df['hearing_diff'] = np.abs(df['hearing_left'] - df['hearing_right'])
    df['vision_avg'] = (df['vision_left'] + df['vision_right']) / 2
    df['hearing_avg'] = (df['hearing_left'] + df['hearing_right']) / 2
    df['sensory_score'] = df['vision_avg'] + df['hearing_avg']
    
    # Clinical risk flags (important for COPD)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df['is_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
    df['has_hypertension'] = ((df['bp_systolic'] >= 130) | (df['bp_diastolic'] >= 80)).astype(int)
    df['severe_hypertension'] = ((df['bp_systolic'] >= 140) | (df['bp_diastolic'] >= 90)).astype(int)
    df['prediabetes'] = ((df['fasting_glucose'] >= 100) & (df['fasting_glucose'] < 126)).astype(int)
    df['diabetes'] = (df['fasting_glucose'] >= 126).astype(int)
    df['low_hdl'] = (df['hdl_cholesterol'] < 40).astype(int)
    df['high_tg'] = (df['triglycerides'] >= 150).astype(int)
    df['high_ldl'] = (df['ldl_cholesterol'] >= 130).astype(int)
    
    # Metabolic syndrome score (strong COPD predictor)
    df['metabolic_syndrome_score'] = (
        df['is_obese'] + 
        df['has_hypertension'] + 
        df['low_hdl'] + 
        df['high_tg'] + 
        (df['fasting_glucose'] >= 100).astype(int)
    )
    
    # Cardiovascular risk score
    df['cv_risk_score'] = (
        df['severe_hypertension'] * 2 +
        df['diabetes'] * 2 +
        df['high_ldl'] +
        df['low_hdl'] +
        (df['bmi'] >= 35).astype(int) * 2
    )
    
    # Composite health scores
    df['waist_bmi_product'] = df['waist_circumference_cm'] * df['bmi']
    df['glucose_bmi_interaction'] = df['fasting_glucose'] * df['bmi']
    df['age_glucose_interaction'] = df['age_numeric'] * df['fasting_glucose']
    df['age_bp_interaction'] = df['age_numeric'] * df['bp_systolic']
    
    return df

# ---------- HELPERS ----------
def encode_categoricals(df, cat_cols):
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().map({"M":0,"F":1,"Y":1,"N":0})
    return df

def find_best_threshold_refined(y_true, probs, step=0.005, low=0.1, high=0.9):
    """More granular threshold search"""
    best_f1, best_t = -1, 0.5
    thresholds = np.arange(low, high + 1e-9, step)
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    
    return best_t, best_f1

# ---------- MAIN PIPELINE ----------
print("=" * 80)
print("IMPROVED COPD RISK PREDICTION PIPELINE")
print("=" * 80)

# 1) Load data
print("\n1) Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
test_ids = test_df['patient_id'].copy()

# Store patient IDs then drop
train_df = train_df.drop('patient_id', axis=1, errors='ignore')
test_df = test_df.drop('patient_id', axis=1, errors='ignore')

# Separate target
X = train_df.drop('has_copd_risk', axis=1).copy()
y = train_df['has_copd_risk'].copy()
X_test_raw = test_df.copy()

print(f"Training samples: {len(X)}, Test samples: {len(X_test_raw)}")
print(f"Positive class ratio: {y.mean():.3f}")

# 2) Categorical encoding (before feature engineering)
cat_cols = ["sex", "oral_health_status", "tartar_presence"]
X = encode_categoricals(X, cat_cols)
X_test_raw = encode_categoricals(X_test_raw, cat_cols)

# 3) Advanced feature engineering
print("\n2) Performing advanced feature engineering...")
X_full = pd.concat([X, y], axis=1)
X_full['has_copd_risk'] = y  # Add target back temporarily
X_full = advanced_feature_engineering(X_full)
X_test_engineered = advanced_feature_engineering(X_test_raw)

# Separate target again
X_engineered = X_full.drop('has_copd_risk', axis=1)

# Align columns
common_cols = list(set(X_engineered.columns) & set(X_test_engineered.columns))
X_engineered = X_engineered[common_cols]
X_test_engineered = X_test_engineered[common_cols]

print(f"Features after engineering: {X_engineered.shape[1]}")

# 4) Prepare for cross-validation
print("\n3) Setting up cross-validation pipeline...")

# Initialize arrays for OOF predictions
oof_probs_lr = np.zeros(len(X_engineered))
oof_probs_svm = np.zeros(len(X_engineered))
oof_probs_mlp = np.zeros(len(X_engineered))

# Initialize arrays for test predictions
test_probs_lr = np.zeros(len(X_test_engineered))
test_probs_svm = np.zeros(len(X_test_engineered))
test_probs_mlp = np.zeros(len(X_test_engineered))

# Setup cross-validation
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print(f"\n4) Starting {N_FOLDS}-fold cross-validation training...")

# Best hyperparameters (expanded search spaces)
best_lr_params = {
    'C': 0.5,
    'penalty': 'l2',
    'solver': 'liblinear',
    'class_weight': 'balanced',
    'max_iter': 3000
}

best_svm_params = {
    'C': 1.5,
    'gamma': 'scale',
    'kernel': 'rbf',
    'class_weight': 'balanced',
    'probability': True
}

best_mlp_params = {
    'hidden_layer_sizes': (256, 128, 64),
    'alpha': 0.0003,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 2000,
    'early_stopping': True,
    'validation_fraction': 0.1
}

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y), 1):
    print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")
    
    # Get fold data
    X_train_fold = X_engineered.iloc[train_idx].copy()
    X_val_fold = X_engineered.iloc[val_idx].copy()
    y_train_fold = y.iloc[train_idx].values
    y_val_fold = y.iloc[val_idx].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_fold),
        columns=X_train_fold.columns
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val_fold),
        columns=X_train_fold.columns
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_engineered),
        columns=X_train_fold.columns
    )
    
    # Apply SMOTE BEFORE scaling (on imputed data)
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_imp, y_train_fold)
    print(f"After SMOTE: {X_train_res.shape[0]} samples")
    
    # Scale AFTER SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Feature selection (on scaled data)
    selector = SelectKBest(mutual_info_classif, k=60)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_res)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Train models
    # Logistic Regression
    lr = LogisticRegression(**best_lr_params, random_state=SEED)
    lr.fit(X_train_selected, y_train_res)
    oof_probs_lr[val_idx] = lr.predict_proba(X_val_selected)[:, 1]
    test_probs_lr += lr.predict_proba(X_test_selected)[:, 1] / N_FOLDS
    
    # SVM
    svm = SVC(**best_svm_params, random_state=SEED)
    svm.fit(X_train_selected, y_train_res)
    oof_probs_svm[val_idx] = svm.predict_proba(X_val_selected)[:, 1]
    test_probs_svm += svm.predict_proba(X_test_selected)[:, 1] / N_FOLDS
    
    # MLP
    mlp = MLPClassifier(**best_mlp_params, random_state=SEED)
    mlp.fit(X_train_selected, y_train_res)
    oof_probs_mlp[val_idx] = mlp.predict_proba(X_val_selected)[:, 1]
    test_probs_mlp += mlp.predict_proba(X_test_selected)[:, 1] / N_FOLDS
    
    # Fold metrics
    f1_lr = f1_score(y_val_fold, (oof_probs_lr[val_idx] >= 0.5).astype(int))
    f1_svm = f1_score(y_val_fold, (oof_probs_svm[val_idx] >= 0.5).astype(int))
    f1_mlp = f1_score(y_val_fold, (oof_probs_mlp[val_idx] >= 0.5).astype(int))
    print(f"Fold F1 scores - LR: {f1_lr:.4f}, SVM: {f1_svm:.4f}, MLP: {f1_mlp:.4f}")

# 5) Threshold optimization
print("\n5) Optimizing thresholds on OOF predictions...")
lr_thresh, lr_f1 = find_best_threshold_refined(y, oof_probs_lr)
svm_thresh, svm_f1 = find_best_threshold_refined(y, oof_probs_svm)
mlp_thresh, mlp_f1 = find_best_threshold_refined(y, oof_probs_mlp)

print(f"\nOptimal thresholds and OOF F1 scores:")
print(f"LR:  threshold={lr_thresh:.3f}, F1={lr_f1:.4f}")
print(f"SVM: threshold={svm_thresh:.3f}, F1={svm_f1:.4f}")
print(f"MLP: threshold={mlp_thresh:.3f}, F1={mlp_f1:.4f}")

# 6) Stacking ensemble
print("\n6) Training stacking ensemble...")
meta_X = np.column_stack([oof_probs_lr, oof_probs_svm, oof_probs_mlp])
meta_model = LogisticRegression(
    C=0.5,
    class_weight='balanced',
    solver='liblinear',
    random_state=SEED
)
meta_model.fit(meta_X, y)

meta_oof_probs = meta_model.predict_proba(meta_X)[:, 1]
meta_thresh, meta_f1 = find_best_threshold_refined(y, meta_oof_probs)
print(f"Stacked: threshold={meta_thresh:.3f}, F1={meta_f1:.4f}")

# 7) Generate test predictions
print("\n7) Generating test predictions...")
meta_test_X = np.column_stack([test_probs_lr, test_probs_svm, test_probs_mlp])
meta_test_probs = meta_model.predict_proba(meta_test_X)[:, 1]

# Apply thresholds
test_preds_lr = (test_probs_lr >= lr_thresh).astype(int)
test_preds_svm = (test_probs_svm >= svm_thresh).astype(int)
test_preds_mlp = (test_probs_mlp >= mlp_thresh).astype(int)
test_preds_meta = (meta_test_probs >= meta_thresh).astype(int)

# 8) Save submissions
print("\n8) Saving submission files...")
os.makedirs('submissions-cl', exist_ok=True)

submissions = {
    'LR': test_preds_lr,
    'SVM': test_preds_svm,
    'MLP': test_preds_mlp,
    'STACKED': test_preds_meta
}

for name, preds in submissions.items():
    df_sub = pd.DataFrame({
        'patient_id': test_ids,
        'has_copd_risk': preds
    })
    filename = f'submissions-cl/{OUTPUT_PREFIX}_{name}.csv'
    df_sub.to_csv(filename, index=False)
    print(f"Saved: {filename} (positive rate: {preds.mean():.3f})")

# Save probabilities for analysis
pd.DataFrame({
    'patient_id': test_ids,
    'prob_lr': test_probs_lr,
    'prob_svm': test_probs_svm,
    'prob_mlp': test_probs_mlp,
    'prob_meta': meta_test_probs
}).to_csv(f'submissions-cl/{OUTPUT_PREFIX}_PROBABILITIES.csv', index=False)

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nBest F1 Score: {meta_f1:.4f}")
print("=" * 80)