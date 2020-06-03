# Python Notebook - is_wait_prediction_v3

import numpy as np
from sklearn.metrics import log_loss, accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn import svm

! pip install lightgbm -t "/tmp" > /dev/null 2>&1
from lightgbm import LGBMClassifier

def eval_model(y_test, y_pred, y_pred_proba):
  acc_score = accuracy_score(y_test, y_pred)
  logloss = log_loss(y_test, y_pred)
  logloss_baseline = log_loss(y_test, np.zeros(len(y_test)))
  rocauc = roc_auc_score(y_test, y_pred_proba[:, 1])
  conf_mat = confusion_matrix(y_test, y_pred)
  tn, fp, fn, tp = conf_mat.ravel()

  print('true pct in testing data of no-wait', 1-y_test.mean(), 'vs wait', y_test.mean(), )
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

wait_thr = 7 * 60
late_thr = 20 * 60

df = datasets['3_feat_v2_more_hist']
df.columns = map(str.lower, df.columns)
print(df.shape)
print(df.head(5))

df = create_store_level_hist_feat(df)

print(">>> check NAs")
print(df.isna().sum() / len(df))

df = df[df['wait_before_ready_time'].notna()]
df.reindex()
print('>>> after dropping NAs in wait_time (label)', df.shape)
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

print('before remove pred_horizon<0, shape', df.shape)
df = df[df['pred_horizon'] >= 0]
print('after remove pred_horizon<0, shape', df.shape)

split_date = '2020-04-09'
train_index = df[df['created_at'] < split_date].index
test_index = df[df['created_at'] >= split_date].index
assert len(train_index.intersection(test_index)) != [], 'data leakage'
print('train data size', len(train_index))
print('test data size', len(test_index))

X_train = df.loc[train_index, cols]
y_train = df.loc[train_index, 'wait_before_ready_time'] > wait_thr
# y_train_late = df.loc[train_index, 'wait_before_ready_time'] < -late_thr

X_test = df.loc[test_index, cols]
y_test = df.loc[test_index, 'wait_before_ready_time'] > wait_thr
# y_test_late = df.loc[test_index, 'wait_before_ready_time'] < -late_thr

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = Normalizer()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LR with class balancing
# clf_balance = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train.values, y_train.values)
# y_pred = clf_balance.predict(X_test)
clf_balance = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train_scaled, y_train.values)
y_pred_balance = clf_balance.predict(X_test_scaled)
y_pred_proba_balance = clf_balance.predict_proba(X_test_scaled)

eval_model(y_test, y_pred_balance, y_pred_proba_balance)

# LR default
# clf = LogisticRegression(random_state=0).fit(X_train.values, y_train.values)
# y_pred = clf.predict(X_test)

clf = LogisticRegression(random_state=0).fit(X_train_scaled, y_train.values)
y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)

eval_model(y_test, y_pred, y_pred_proba)

# # LR random search
# logistic_default = LogisticRegression()

# # Create regularization penalty space
# penalty = ['l1', 'l2']
# # Create regularization hyperparameter distribution using uniform distribution
# C = uniform(loc=0, scale=4)
# # Create hyperparameter options
# hyperparameters = dict(C=C, penalty=penalty)

# random_search = RandomizedSearchCV(logistic_default, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=1)
# clf_random_search = random_search.fit(X_train_scaled, y_train.values)

# y_pred_random_search = clf_random_search.predict(X_test_scaled)
# y_pred_random_search_proba = clf_random_search.predict_proba(X_test_scaled)

# eval_model(y_test, y_pred_random_search, y_pred_random_search_proba)

# clf_svm = svm.SVC()
# clf_svm.fit(X_train_scaled, y_train.values)

# y_pred_svm = clf_svm.predict(X_test_scaled)
# y_pred_svm_proba = clf_svm.predict_proba(X_test_scaled)

# eval_model(y_test, y_pred_svm, y_pred_svm_proba)

# try lightGBM to see how good/bad the performance can be
clf_lgbm = LGBMClassifier()

clf_lgbm.fit(X_train_scaled, y_train.values)

y_pred_lgbm = clf_lgbm.predict(X_test_scaled)
y_pred_lgbm_proba = clf_lgbm.predict_proba(X_test_scaled)

eval_model(y_test, y_pred_lgbm, y_pred_lgbm_proba)

print('coef', clf.coef_)
print('intercept', clf.intercept_)

# X_train_std = np.std(X_train, 0)
X_train_std = np.std(X_train_scaled, 0)

importance_score = [i * j for i, j in zip (list(X_train_std), list(clf.coef_))][0]

res = dict(zip(cols, list(importance_score)))
res = {k: v for k, v in sorted(res.items(), key=lambda item: abs(item[1]), reverse=True)}
print('Feature importance')
print(res)
plt.bar(res.keys(), height=res.values())
plt.xticks(rotation=90)

df_wait = df['wait_before_ready_time']
df_wait[(df_wait < df_wait.quantile(.9999)) & (df_wait > df_wait.quantile(.001))].hist(alpha=0.5)

# print(sum(df_wait>wait_thr)/len(df_wait), sum(df_wait<-late_thr)/len(df_wait))

plt.legend()
plt.title('Histogram of wait_before_ready_time (all data)')

def check_if_no_wait(X, y, y_pred, pred_horizon_thr):
  close_index = X[X['pred_horizon'] < pred_horizon_thr].index
  df_y_pred = pd.DataFrame(data=y_pred, index=y.index)
  pct = df_y_pred.loc[close_index].mean().values[0]
  print(pct, 'are predicted as wait when pred_horizon is', pred_horizon_thr, 'seconds')
  return pct

#check if predict 'no wait' when time is close to food ready time
pct = []
pred_horizon_thr = [240, 360, 480, 600, 720, 840, 960]
for thr in pred_horizon_thr:
  pct.append(check_if_no_wait(X_test, y_test, y_pred, thr))

plt.plot(pred_horizon_thr, pct)
plt.xlabel('pred_horizon (seconds)')
plt.ylabel('pct of wait predicted by model')

test_sampled_index = X_test.sample(1000).index

X_test_sampled = X_test.loc[test_sampled_index].copy()
y_test_sampled = y_test.loc[test_sampled_index].copy()

res = []
ranges = range(0, 60*13, 60)
for var in ranges:
  # print(var)
  X_test_sampled['pred_horizon'] = var
  X_test_sampled_scaled = scaler.transform(X_test_sampled)
  y_pred_sampled_scaled = clf.predict(X_test_sampled_scaled)
  y_pred_sampled_scaled_proba = clf.predict_proba(X_test_sampled_scaled)
  # eval_model(y_test_sampled, y_pred_sampled_scaled, y_pred_sampled_scaled_proba)
  res.append(y_pred_sampled_scaled)

res = np.array(res)
res_mean = res.mean(axis=1)

plt.plot(list(ranges), res_mean)
plt.title('Randomly sampled from testing, and artificially created pred_horizon')
plt.xlabel('pred_horizon (seconds)')

# def check_if_is_late(X, y, y_pred, late_thr):
#   late_index = X[X['wait_before_ready_time'] < -late_thr].index
#   df_y_pred = pd.DataFrame(data=y_pred, index=y.index)
#   pct = df_y_pred.loc[late_index].mean().values[0]
#   print(pct, 'are predicted as wait but is late by', late_thr, 'seconds')
#   return pct

# late_thr = [120, 240, 360, 480, 600, 720, 840, 960]

# pct = []
# for thr in late_thr:
#   pct.append(check_if_is_late(df.loc[test_index], y_test, y_pred, thr))

# # plt.plot(late_thr, pct)
# # plt.xlabel('late (seconds)')
# # plt.ylabel('pct of late but was predicted as wait')

df_test = df.loc[test_index]

df_y_pred = pd.DataFrame(data=y_pred, index=y_test.index, columns=['prediction'])
df_test_wait = df_y_pred[df_y_pred['prediction'] == True]
wait_index = df_test_wait.index

print('all test', df_test.shape, 'test_wait', wait_index.shape)

assert df_test.shape[0] > df_test_wait.shape[0], 'incorrect test and test_wait shape'

def get_late_pct(thr):
  df_test_late = df_test[df_test['wait_before_ready_time'] < -thr]
  late_index = df_test_late.index

  wait_and_late_index = wait_index.intersection(late_index)
  
  print('\t\tlate shape', late_index.shape, 'wait shape', wait_index.shape, 'wait_and_late shape', wait_and_late_index.shape)
  pct = wait_and_late_index.shape[0]/wait_index.shape[0]
  
  return pct

late_thr = [0, 1, 3, 5, 7, 10, 20]
pcts = []  
for thr in late_thr:
  pct = get_late_pct(thr*60)
  print(thr, pct)
  pcts.append(pct)
  

plt.plot(late_thr, pcts)
plt.xlabel('threshold(minutes) to define lateness')
plt.ylabel('pct of pred_wait_but_late / pred_wait ')



df.loc[wait_index]['pred_horizon'].hist()
plt.title('Histogram for all points pred as wait')
plt.xlabel('pred_horizon (seconds)')
plt.ylabel('Freq')
print('min', df.loc[wait_index]['pred_horizon'].min(), 'mean', df.loc[wait_index]['pred_horizon'].mean())



