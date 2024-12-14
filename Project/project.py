from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

train_data = pd.read_csv('Project/DataSets/train_final.csv')
test_data = pd.read_csv('Project/DataSets/test_final.csv')

def preprocess_data(df):
    df = df.replace('?', np.nan)
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

train_data, encoders = preprocess_data(train_data)
target_column = 'income>50K'
id_column = 'ID'
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

test_data, _ = preprocess_data(test_data)

if id_column in test_data.columns:
    test_ids = test_data[id_column]
    X_test = test_data.drop(columns=[id_column])
else:
    test_ids = pd.Series(range(len(test_data)))
    X_test = test_data

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
power_transformer = PowerTransformer()
X = pd.DataFrame(power_transformer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(power_transformer.transform(X_test), columns=X_test.columns)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

rf = RandomForestClassifier(random_state=42, class_weight=class_weights_dict, n_estimators=500, max_depth=20)
rf.fit(X_train, y_train)
y_val_pred_rf = rf.predict_proba(X_val)[:, 1]
auc_rf = roc_auc_score(y_val, y_val_pred_rf)
print(f"Random Forest AUC: {auc_rf}")

gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
gbc.fit(X_train, y_train)
y_val_pred_gbc = gbc.predict_proba(X_val)[:, 1]
auc_gbc = roc_auc_score(y_val, y_val_pred_gbc)
print(f"Gradient Boosting AUC: {auc_gbc}")

xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_val_pred_xgb = xgb.predict_proba(X_val)[:, 1]
auc_xgb = roc_auc_score(y_val, y_val_pred_xgb)
print(f"XGBoost AUC: {auc_xgb}")

y_val_pred_stack = (y_val_pred_rf + y_val_pred_gbc + y_val_pred_xgb) / 3
auc_stack = roc_auc_score(y_val, y_val_pred_stack)
print(f"Stacked Model AUC: {auc_stack}")

y_test_pred_rf = rf.predict_proba(X_test)[:, 1]
y_test_pred_gbc = gbc.predict_proba(X_test)[:, 1]
y_test_pred_xgb = xgb.predict_proba(X_test)[:, 1]
y_test_pred_stack = (y_test_pred_rf + y_test_pred_gbc + y_test_pred_xgb) / 3

submission = pd.DataFrame({'ID': test_ids, 'Prediction': y_test_pred_stack})
submission.to_csv('submission_6.csv', index=False)
print("Submission file saved as 'submission_6.csv'")
