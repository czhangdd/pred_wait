# Python Notebook - is_wait_prediction_v3_model_extraction

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

# ! pip install lightgbm -t "/tmp" > /dev/null 2>&1
# from lightgbm import LGBMClassifier

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

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = Normalizer()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = df.loc[train_index, cols]
y_train = df.loc[train_index, 'wait_before_ready_time'] > wait_thr

X_test = df.loc[test_index, cols]
y_test = df.loc[test_index, 'wait_before_ready_time'] >= wait_thr


for i in range(X_train_scaled.shape[1]):
  print(X_train_scaled[:, i].max())

assert all(X_train_store_id_agg[feat_hist_agg].max() > 0.999), 'check if all scaled features max == 1'

# scaler2 = MinMaxScaler()
# df_train_agg_store_id = df.loc[train_index][['store_id'] + feat_hist_agg].copy()
# df_train_agg_store_id[feat_hist_agg] = scaler2.fit_transform(df_train_agg_store_id[feat_hist_agg])
# for f in df_train_agg_store_id.columns:
  # print(f, df_train_agg_store_id[f].max())

# # LR with class balancing
# # clf_balance = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train.values, y_train.values)
# # y_pred = clf_balance.predict(X_test)
# clf_balance = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train_scaled, y_train.values)
# y_pred_balance = clf_balance.predict(X_test_scaled)
# y_pred_proba_balance = clf_balance.predict_proba(X_test_scaled)

# eval_model(y_test, y_pred_balance, y_pred_proba_balance)

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

# # try lightGBM to see how good/bad the performance can be
# clf_lgbm = LGBMClassifier()

# clf_lgbm.fit(X_train_scaled, y_train.values)

# y_pred_lgbm = clf_lgbm.predict(X_test_scaled)
# y_pred_lgbm_proba = clf_lgbm.predict_proba(X_test_scaled)

# eval_model(y_test, y_pred_lgbm, y_pred_lgbm_proba)

print('coef', clf.coef_)
print('intercept', clf.intercept_)

df_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns, index=X_train.index)

assert all(df_train_scaled.index == train_index), 'X_train index and train_index do not match'

X_train_store_id = df.loc[train_index]['store_id'].to_frame().join(df_train_scaled)
X_train_store_id_agg = X_train_store_id[ ['store_id'] + feat_hist_agg]

print(X_train_store_id_agg.head(5))

df_store_level = X_train_store_id_agg.drop_duplicates(keep='first')
num_stores = X_train_store_id_agg['store_id'].nunique()
assert df_store_level.shape[0] == num_stores, 'unmatched after dropping dup'
print('num_stores', num_stores)

print(clf.coef_)
coef = list(clf.coef_[0])[:7]
print(coef)

intercept = clf.intercept_[0]
print(intercept)

hist_agg_value = df_store_level[feat_hist_agg] * coef
hist_agg_value['intercept'] = intercept
hist_agg_value = hist_agg_value.sum(axis=1)
hist_agg_value = hist_agg_value.to_frame()
hist_agg_value.columns = ['value']
print(hist_agg_value)

assert all(df_store_level.index == hist_agg_value.index), 'unmatched index'

df_pre_calculated_value = df_store_level['store_id'].to_frame().join(hist_agg_value)
print(df_pre_calculated_value)

default_value = df_pre_calculated_value['value'].mean()
df_pre_calculated_value.loc[len(df_pre_calculated_value)]=['no_match', default_value] 
df_pre_calculated_value.tail(4)

import notebooksalamode as mode
mode.export_csv(df_pre_calculated_value)

for f in feat_real_time:
  print('feature:', f, '\n\tmin:', X_train[f].max(), 'max:',  X_train[f].max())

### Testing for extracted csv/model coef

valid_X_test = df.loc[test_index][['store_id'] + cols]

valid_X_test['store_id']

valid_X_test = df.loc[test_index][['store_id'] + cols]
pre_cal_store = df_pre_calculated_value['store_id'].unique()

tmp = valid_X_test["store_id"].unique()

print(pre_cal_store.shape, tmp.shape)

print('in pre_cal not in testing', len(set(pre_cal_store) - set(tmp)))
print('in testing not in pre_cal', len(set(tmp) - set(pre_cal_store)), 'pct', len(set(tmp) - set(pre_cal_store))/len(pre_cal_store))

valid_X_test_pre_cal = pd.merge(valid_X_test, df_pre_calculated_value, on='store_id', how='left')
assert valid_X_test_pre_cal.shape[0] == valid_X_test.shape[0], 'unmatched shape'

valid_X_test_pre_cal['value'] = valid_X_test_pre_cal['value'].fillna(-3.921313)
assert valid_X_test_pre_cal['value'].isna().sum()

valid_X_test_pre_cal.head(5)


result = valid_X_test_pre_cal['value']
+10.64552989 * (valid_X_test_pre_cal['tip']/20000.0)
-22.46182226 * (valid_X_test_pre_cal['flf']/19.0)
+46.03914325 * (valid_X_test_pre_cal['pred_horizon']/8575.0)

proba = 1/(1+np.exp(-result))
proba
# binary = proba >=0.5


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



