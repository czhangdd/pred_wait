# Python Notebook - is_wait_prediction_v1

import numpy as np
from sklearn.metrics import log_loss, accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sn

def eval_model(y_test, y_pred, y_pred_proba):
  acc_score = accuracy_score(y_test, y_pred)
  logloss = log_loss(y_test, y_pred)
  logloss_baseline = log_loss(y_test, np.zeros(len(y_test)))
  rocauc = roc_auc_score(y_test, y_pred)
  conf_mat = confusion_matrix(y_test, y_pred)
  tn, fp, fn, tp = conf_mat.ravel()

 
  print('pct of zeros vs ones in ground truth', y_test.mean(), 1-y_test.mean())
  print('acc', acc_score)
  print('log loss pred vs baseline', logloss, logloss_baseline)
  print('roc auc', rocauc)
  print('conf matrix (percentage of all data)\n', conf_mat/len(y_test))
  print("absolute values of tn", tn, "fp", fp, "fn", fn, "tp", tp)
  
  
  annot_kws = {"ha": 'left',"va": 'top'}
  sn.heatmap(conf_mat, annot=True, annot_kws=annot_kws, cmap="YlGnBu") # font size

df = datasets['feat_v1_remove_biz_ids']
df.columns = map(str.lower, df.columns)
print(df.shape)
print(df.head(5))

print(">>> check NAs")
df.isna().sum()

df = df[df['wait_time'].notna()]
print('>>> dropping NAs in wait_time (label)', df.shape)
print('>>> fill NAs with 0')
df = df.fillna(0)

cols = ['avg_num_assigns', 'avg_subtotal', 'avg_tip', 'acceptance_rate_on_check_in', 'subtotal', 'tip', 'flf']

X = df[cols]
y = df['wait_time'] > 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# LR with class balancing
clf_balance = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train.values, y_train.values)
y_pred = clf_balance.predict(X_test)
y_pred_proba = clf_balance.predict_proba(X_test)

eval_model(y_test, y_pred, y_pred_proba)

# LR default
clf = LogisticRegression(random_state=0).fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

eval_model(y_test, y_pred, y_pred_proba)

print('coef', clf.coef_)
print('intercept', clf.intercept_)

X_train_std = np.std(X_train, 0)
importance_score = [i * j for i, j in zip (list(X_train_std), list(clf.coef_))][0]

res = dict(zip(cols, list(importance_score)))
res = {k: v for k, v in sorted(res.items(), key=lambda item: abs(item[1]), reverse=True)}
print('Feature importance')
print(res)
plt.bar(res.keys(), height=res.values())
plt.xticks(rotation=90)




