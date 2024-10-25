import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

train_df = pd.read_csv('/content/train_final.csv')
test_df = pd.read_csv('/content/test_final.csv')

# ------------------- Step 1: Data Preprocessing -------------------
X = train_df.drop('income>50K', axis=1)
y = train_df['income>50K']

X.replace('?', np.nan, inplace=True)
test_df.replace('?', np.nan, inplace=True)
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_num = SimpleImputer(strategy='mean')

X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
X[num_cols] = imputer_num.fit_transform(X[num_cols])

test_df[cat_cols] = imputer_cat.transform(test_df[cat_cols])
test_df[num_cols] = imputer_num.transform(test_df[num_cols])

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X[cat_cols] = encoder.fit_transform(X[cat_cols])
test_df[cat_cols] = encoder.transform(test_df[cat_cols])
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# ------------------- Step 2: Exploratory Data Analysis (EDA) -------------------

print(y.value_counts(normalize=True))

# ------------------- Step 3: Model Selection & Training -------------------

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_val_pred_proba = rf_model.predict_proba(X_val)[:, 1]

# Evaluate model performance using AUC-ROC
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f'Validation AUC-ROC: {auc_score:.4f}')

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Random Forest (AUC = {:.4f})'.format(auc_score))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ------------------- Step 4: Final Prediction & Submission -------------------

test_df = test_df[X.columns]
test_proba = rf_model.predict_proba(test_df)[:, 1]
submission = pd.DataFrame({
    'ID': test_df.index + 1,  # Ensure ID starts from 1
    'Prediction': test_proba
})
submission.to_csv('submission1.csv', index=False)

print('Submission file has been saved!')
