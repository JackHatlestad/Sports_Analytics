import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import xgboost as xgb

df = pd.read_excel("Former_Coaches.xlsx")
df.fillna(0, inplace=True)



features = ['Yrs', 'G', 'W', 'L', 'T', 'RG W-L%', 'Yr plyf', 'G plyf', 'W plyf', 'L plyf', 'PL W-L%', 'Chmp', 'SBwl', 'Conf']

X = df[features]
y = df['Hall of Fame']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc:.3f}")

current_coaches = pd.read_excel("Current_Coaches.xlsx")  
current_coaches.fillna(0, inplace=True)

probs = model.predict_proba(current_coaches[features])[:, 1]
current_coaches['Hall of Fame'] = probs

print(current_coaches[['Coach', 'Hall of Fame']].sort_values(by='Hall of Fame', ascending=False))

importances = model.feature_importances_
for f, score in zip(features, importances):
    print(f"{f}: {score:.4f}")
