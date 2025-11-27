import numpy as np
import pandas as pd
import warnings, os
warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingClassifier

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_raw = train[['signal_strength', 'response_level']].values
y_raw = train['category'].values
X_test_raw = test[['signal_strength', 'response_level']].values

# Encode
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)


def full_features(X):
    x, y = X[:, 0], X[:, 1]
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)
    return np.column_stack([
        x, y,
        x**2, y**2, x*y,
        x**3, y**3,
        r, r**2, r**3,
        np.sin(th), np.cos(th),
        np.sin(2*th), np.cos(2*th)
    ])

# Base FE
X_fe = full_features(X_raw)
X_test_fe = full_features(X_test_raw)

# + KMeans geometry
cluster_feats = []
cluster_feats_test = []
for k in [n_classes, n_classes+1, n_classes+2]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(X_raw)
    cluster_feats.append(km.transform(X_raw))
    cluster_feats_test.append(km.transform(X_test_raw))

X = np.hstack([X_fe] + cluster_feats)
X_test = np.hstack([X_test_fe] + cluster_feats_test)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# CV + scoring
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
macro_f1 = make_scorer(f1_score, average="macro")

# =======================================================
# MODEL 1 — SVM
# =======================================================
svm = SVC(kernel='rbf', probability=True, class_weight='balanced')

svm_params = {
    "C": [1, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 40, 50],
    "gamma": [0.001, 0.003, 0.005, 0.007, 0.008, 0.009,
              0.01, 0.012, 0.015, 0.02, 0.025, 0.03]
}

grid_svm = GridSearchCV(
    svm, svm_params,
    cv=cv, scoring=macro_f1,
    n_jobs=-1, verbose=0
)
grid_svm.fit(X_scaled, y)
best_svm = grid_svm.best_estimator_

print("Best SVM:", grid_svm.best_params_)
print("SVM CV:", grid_svm.best_score_)

# =======================================================
# MODEL 2 — MLP
# =======================================================

mlp = MLPClassifier(solver="adam", early_stopping=True,
                    validation_fraction=0.15, random_state=42)

nn_params = {
    "hidden_layer_sizes": [
        (256, 128, 64), (200, 100, 50), (150, 100, 50),
        (128, 64, 32), (128, 64), (100, 50)
    ],
    "alpha": [0.0001, 0.0003, 0.001, 0.003],
    "learning_rate_init": [0.0005, 0.001, 0.002]
}

grid_nn = GridSearchCV(
    mlp, nn_params,
    cv=cv, scoring=macro_f1,
    n_jobs=-1, verbose=0
)
grid_nn.fit(X_scaled, y)
best_nn = grid_nn.best_estimator_

print("Best NN:", grid_nn.best_params_)
print("NN CV:", grid_nn.best_score_)

# =======================================================
# STACKING 
# =======================================================
stack = StackingClassifier(
    estimators=[('svm', best_svm)],
    final_estimator=best_nn,
    stack_method='predict_proba',
    cv=cv,
    n_jobs=-1
)

print("\nTraining STACK model...")
stack.fit(X_scaled, y)

# =======================================================
# EXPORT SUBMISSIONS
# =======================================================
os.makedirs("submissions_n2", exist_ok=True)

def save_pred(model, fname):
    yp = le.inverse_transform(model.predict(X_test_scaled))
    pd.DataFrame({
        "sample_id": test["sample_id"],
        "category": yp
    }).to_csv(f"submissions_n2/{fname}", index=False)
    print("Saved:", fname)

save_pred(best_svm, "sub_svm.csv")
save_pred(best_nn, "sub_mlp.csv")
save_pred(stack, "sub_stack.csv")

print("\nGenerated: sub_svm.csv | sub_mlp.csv | sub_stack.csv")
