# Python Notebook - is_wait_prediction_v2

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

def create_store_level_hist_feat(df):
  d = {'wait_before_ready_time': ['mean', 'min', 'max'], 'd2r_duration': ['mean', 'min', 'max'], 'pred_horizon': ['mean', 'min', 'max']}
  df_agg = df.groupby(['store_id'], as_index=False).agg(d)
  
  df_agg.columns = ["_".join(x) for x in df_agg.columns.ravel()]
  df_agg.rename(columns={'store_id_':'store_id'}, inplace=True)
  
  df_hist = pd.merge(df, df_agg, on=['store_id'], how='left')
  return df_hist 

df = datasets['3_feat_v2_more_hist']
df.columns = map(str.lower, df.columns)
print(df.shape)
print(df.head(5))

df = create_store_level_hist_feat(df)

print(">>> check NAs")
df.isna().sum()

df = df[df['wait_before_ready_time'].notna()]
df.reindex()
print('>>> dropping NAs in wait_time (label)', df.shape)
print('>>> fill NAs with 0')
df = df.fillna(0)

print(df.head(5))
print(df.columns)

feat_hist_agg = ['avg_num_assigns', 'avg_subtotal', 'avg_tip', \
                 'acceptance_rate_on_check_in', 
                 'd2r_duration_mean', 'd2r_duration_max', 'd2r_duration_min', \
                 'wait_before_ready_time_mean',  'wait_before_ready_time_max', 'wait_before_ready_time_min',\
                 'pred_horizon_mean']
feat_real_time = ['subtotal', 'tip', 'flf', 'pred_horizon']

cols = feat_hist_agg + feat_real_time
for f in cols:
  df[f] = df[f].astype('float32')

split_date = '2020-04-09'
train_index = df[df['created_at'] < split_date].index
test_index = df[df['created_at'] >= split_date].index
assert len(train_index.intersection(test_index)) != [], 'data leakage'
print('train data size', len(train_index))
print('test data size', len(test_index))

X_train = df.loc[train_index, cols]
y_train = df.loc[train_index, 'wait_before_ready_time'] > 5*60
y_train_late = df.loc[train_index, 'late_after_ready_time'] > 5*60


X_test = df.loc[test_index, cols]
y_test = df.loc[test_index, 'wait_before_ready_time'] > 5*60
y_test_late = df.loc[test_index, 'late_after_ready_time'] > 5*60

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


df_wait = df['wait_before_ready_time']
df_wait[(df_wait < df_wait.quantile(.98)) & (df_wait > df_wait.quantile(.02))].hist(alpha=0.5, label='wait_before_ready_time')

df_late = df['late_after_ready_time']
df_late[ (df_late < df_late.quantile(.98)) & (df_late > df_late.quantile(.02))].hist(alpha=0.5, label='late_after_ready_time')
plt.legend()
plt.title('Histogram (2 weeks)')

df_wait.quantile(.64)

df_late.quantile(.65)

np.mean((y_pred == True).astype(int) == (y_train_late == True).values.astype(int) )

y_train.values, df.loc[train_index]['pred_horizon'].shape

plt.plot(df.loc[train_index]['pred_horizon'].values[:10000], y_train.values[:10000], '.')

df[df['pred_horizon'] <0].shape[0] / len(df)

df_tmp = df[df['pred_horizon']>=0]
df_tmp['pred_horizon'].hist()

df_tmp['pred_horizon'].describe()

interval = [0, 300, 800, 1200]
for i in range(len(interval) - 1):
  fig = plt.figure()
  df_interval = df_tmp[(df_tmp['pred_horizon'] <= interval[i+1]) & (df_tmp['pred_horizon'] > interval[i])]
  print(df_interval.shape)
  plt.plot(df_interval['pred_horizon'], df_interval['wait_before_ready_time'] > 300, '.')

close_index = X_test[X_test['pred_horizon'] < 300].index

y_test.loc[close_index]

df_y_pred = pd.DataFrame(data=y_pred, index=y_test.index)

df_y_pred.loc[close_index].mean()



