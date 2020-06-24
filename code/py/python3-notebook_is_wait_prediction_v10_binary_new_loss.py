# Python Notebook - is_wait_prediction_v10_binary_new_loss

import numpy as np
from sklearn.metrics import log_loss, accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datetime import datetime, timedelta
from sklearn.feature_selection import RFECV
import collections

def eval_model(y_test, y_pred, y_pred_proba):
  acc_score = accuracy_score(y_test, y_pred)
  logloss = log_loss(y_test, y_pred_proba)
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

# def eval_model_multi_class(y_test, y_pred, y_pred_proba, labels):
#   counter = collections.Counter(y_test.values)
#   freq = [counter[i]/len(y_test) for i in np.unique(y_test.values)]
#   print('freq', freq)
  
#   acc_score = accuracy_score(y_test, y_pred)
#   print('acc_score', acc_score)
  
#   logloss = log_loss(y_test, y_pred_proba)
#   print('log loss pred', logloss)

#   conf_mat = confusion_matrix(y_test, y_pred, labels=labels)
#   print('conf matrix (percentage of all data)\n', np.array_str(conf_mat/len(y_test)*100, precision=4, suppress_small=True))

def create_store_level_hist_feat(df):
  d = {'wait_before_ready_time': ['mean', 'min', 'max'], 'd2r_duration': ['mean', 'min', 'max'], 'pred_horizon': ['mean', 'min', 'max']}
  df_agg = df.groupby(['store_id'], as_index=False).agg(d)
  
  df_agg.columns = ["_".join(x) for x in df_agg.columns.ravel()]
  df_agg.rename(columns={'store_id_':'store_id'}, inplace=True)
  
  df_hist = pd.merge(df, df_agg, on=['store_id'], how='left')
  return df_hist 

wait_thr = 5 * 60 #4min
long_wait_thr = 10 * 60
delay_thr = 0

# df = datasets['4_feat_filter']
df = datasets['7_wait_geo_remove_store']
df.columns = map(str.lower, df.columns)
raw_data_size = df.shape
print("raw_data_size", raw_data_size)
print(df.head(5))

f = 'pred_horizon'

df_clean = df.copy()
df_clean = df_clean[df_clean['flf'] < 5]
df_clean = df_clean[(df_clean['pred_horizon'] < 60 * 60) & (df_clean['pred_horizon'] >= 0)]

print('after remove pred_horizon and flf, shape', df_clean.shape)

df_clean[f].min(), df_clean[f].max()

## Outlier removal
print(">>> check NAs")
print(df.isna().sum() / len(df))

print(">>> remove data with invalid lables")
df_clean = df_clean[df_clean['wait_before_ready_time'].notna()]
df_clean.reindex()
print('>>> after dropping NAs in wait_time (label)', df_clean.shape, 1-df_clean.shape[0]/raw_data_size[0])

print('>>> drop NA with 0')
df_clean.dropna(inplace=True)
# df_clean = df_clean.fillna(0)

print('>>> after dropping NAs in other columns', df_clean.shape, 1-df_clean.shape[0]/raw_data_size[0])

df = df_clean
print(">>> check if na?", df.isnull().values.any())

df = create_store_level_hist_feat(df)
print(">>> adding agg features")
print(df.columns)

# add lateness labels
df['label'] = "ontime"
df.loc[df['wait_before_ready_time'] >= wait_thr, 'label'] = 'early'
df.loc[df['wait_before_ready_time'] <= -delay_thr, 'label'] = 'late'

df['label_int'] = 0
df.loc[df['label'] == 'early', 'label_int'] = 1
# df.loc[df['label'] == 'ontime', 'label_int'] = 0
# df.loc[df['label'] == 'late', 'label_int'] = 0

# assert (df['label_int'].value_counts(normalize=True).values ==  df['label'].value_counts(normalize=True).values).all(), "error in converting str label to int label"

print('Percentage of labels')
print(df['label'].value_counts(normalize=True))

feat_raw = [    'acceptance_rate_on_check_in', 
                # 'd2r_duration', ?? real-time feature available??
                # 'num_assigns',
                'flf', 'pred_horizon', 'subtotal', 'tip',
                'avg_num_assigns', 'avg_subtotal', 'avg_tip',
                'avg_d2r_duration', 'num_nearby_idle', 'num_busy',
                'wait_before_ready_time_mean', 'wait_before_ready_time_min', 'wait_before_ready_time_max', 
                'd2r_duration_mean', 'd2r_duration_min', 'd2r_duration_max', 
                'pred_horizon_mean', 'pred_horizon_min', 'pred_horizon_max']

# manually select top features
feat_hist_agg = ['wait_before_ready_time_mean', 'd2r_duration_mean', 'wait_before_ready_time_max',\
                 'd2r_duration_min', 'pred_horizon_mean', 'wait_before_ready_time_min' , 'avg_num_assigns']
feat_real_time = ['tip', 'flf', 'pred_horizon', 'num_busy', 'num_nearby_idle']
feat_manual = feat_hist_agg + feat_real_time

cols = feat_hist_agg + feat_real_time
# cols = feat_raw
for f in cols:
  df[f] = df[f].astype('float32')

df['created_at'] = pd.to_datetime(df['created_at'])
date_train_end = df['created_at'].dt.date.max() - timedelta(weeks=1)

train_index = df[df['created_at'] <= date_train_end].index
test_index = df[df['created_at'] > date_train_end].index
assert len(train_index.intersection(test_index)) != [], 'data leakage'
print('train data size', len(train_index), 'date', df.loc[train_index, 'created_at'].min(), df.loc[train_index, 'created_at'].max())
print('test data size', len(test_index), 'date', df.loc[test_index, 'created_at'].min(), df.loc[test_index, 'created_at'].max())

X_train = df.loc[train_index, cols]
y_train = df.loc[train_index, 'label']
y_train_int = df.loc[train_index, 'label_int']

X_test = df.loc[test_index, cols]
y_test = df.loc[test_index, 'label']
y_test_int = df.loc[test_index, 'label_int']

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = Normalizer()
scaler.fit(X_train)

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[X_train.columns] = scaler.transform(X_train)
X_test_scaled[X_test.columns] = scaler.transform(X_test)

for i in range(X_train_scaled.shape[1]):
  assert X_train_scaled.values[:, i].max() > 0.999, 'check if all scaled features max == 1'

sorted_corr_col = df[cols+['label_int']].corr()['label_int'].abs().sort_values(ascending=False).index

df[sorted_corr_col].corr()['label_int'][1:].plot.bar(rot=90, title="raw corr with label")

# from sklearn.feature_selection import RFE

# estimator = LogisticRegression()

# selector = RFE(estimator=estimator, n_features_to_select=5, step=0.2)
# # selector = RFECV(estimator, step=0.3, cv=5)
# selector = selector.fit(X_train_scaled, y_train.values)
# selector.support_

# feat_raw = np.array(feat_raw)
# feat_important = feat_raw[selector.support_]
# feat_manual = feat_hist_agg + feat_real_time
# print('feat_important', feat_important)
# print('in auto selected feat not in manual selected feat', set(feat_important) - set(feat_manual))
# print('in manual selected feat not in auto selected feat', set(feat_manual) - set(feat_important))
# print('len of feat_manual', len(feat_manual), 'len of feat_raw', len(feat_raw), 'len of feat_important', len(feat_important))

feat_auto = ['pred_horizon', 'tip', 'wait_before_ready_time_mean','wait_before_ready_time_min', 'pred_horizon_mean']

# feat_top = ['pred_horizon', 'tip', 'wait_before_ready_time_mean', 'wait_before_ready_time_min', 'pred_horizon_mean']
# X_train_scaled_reduced = X_train_scaled[feat_manual]
# X_test_scaled_reduced = X_test_scaled[feat_manual]

# clf_reduced = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(X_train_scaled_reduced, y_train.values) #newton
# y_pred = clf_reduced.predict(X_test_scaled_reduced)
# y_pred_proba = clf_reduced.predict_proba(X_test_scaled_reduced)

# print('classes', clf_reduced.classes_)
# eval_model_multi_class(y_test, y_pred, y_pred_proba, labels=clf_reduced.classes_)

# Multi-class
# LR default
# clf = LogisticRegression(random_state=0).fit(X_train.values, y_train.values)
# y_pred = clf.predict(X_test)

# clf = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train.values)
# y_pred = clf.predict(X_test_scaled)
# y_pred_proba = clf.predict_proba(X_test_scaled)

# print('classes', clf.classes_)
# eval_model_multi_class(y_test, y_pred, y_pred_proba, labels=clf.classes_)

# # Create regularization penalty space
# penalty = ['l1', 'l2']

# # Create regularization hyperparameter distribution using uniform distribution
# C = uniform(loc=0, scale=4)

# # Create hyperparameter options
# hyperparameters = dict(C=C, penalty=penalty)

# logistic = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train.values)

# clf_randomsearch = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=10, cv=5, verbose=2, n_jobs=2).fit(X_train_scaled, y_train.values)

# y_pred = clf_randomsearch.predict(X_test_scaled)
# y_pred_proba = clf_randomsearch.predict_proba(X_test_scaled)


# print('best params', clf_randomsearch.best_params_)

# # random search best params {'C': 2.67898414721392, 'penalty': 'l2'}
# print('classes', clf_randomsearch.best_estimator_.classes_)
# eval_model_multi_class(y_test, y_pred, y_pred_proba, labels=clf_randomsearch.best_estimator_.classes_)

import random
from sklearn.model_selection import KFold

def random_search(X, y, y_extra, n_iter, n_splits):
  gridParams = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    }
  
  results = []
  
  for i in range(0,n_iter,1):
      chosenParams = {
                'penalty': random.choice(gridParams['penalty']), 
                'C': random.choice(gridParams['C']), 
        }
        
      kf = KFold(n_splits=n_splits)
      score_cv = 0
      count = 1
      for train_index, test_index in kf.split(X):
        print('fitting', count, 'of', n_splits, 'folds')
        # print("TRAIN:", train_index, "TEST:", test_index)

        x_train, x_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        y_train_extra, y_valid_extra = y_extra[train_index], y_extra[test_index]
      
        model = LogisticRegression(penalty=chosenParams['penalty'], C=chosenParams['C'], random_state=0)
        model.fit(x_train, y_train)

        y_valid_predict = model.predict(x_valid)
      
        res = pd.DataFrame()
        res['y_valid'] = y_valid
        res['y_valid_pred'] = y_valid_predict
        res['y_valid_extra'] = y_valid_extra
      
        res['score'] = 0
        res.loc[(res['y_valid'] != res['y_valid_pred']) & (res['y_valid_extra']=='late'), 'score'] = 1.5 #1.25
        res.loc[(res['y_valid'] != res['y_valid_pred']) & (res['y_valid_extra']=='early'), 'score'] = 1.0
      
        score = res['score'].mean()
        score_cv += score
        count += 1

      r = [score_cv/n_splits, chosenParams]
      print(r)
      results.append(r)
      
  final_results = pd.DataFrame(results)
#   final_results.columns = ['test_log_loss','train_log_loss','parameters']
  # final_results.columns = ['test_rmse','train_rmse','parameters']
  final_results.columns = ['score', 'parameters']
  
  return final_results

# results = random_search(X_train[:5000].values, y_train_int[:5000].values, y_train[:5000].values, n_iter=4, n_splits=5)
results = random_search(X_train.values, y_train_int.values, y_train.values, n_iter=10, n_splits=5)

results.sort_values(['score'], ascending = True).head(1)['parameters'].values

print('>>> finish grid search')

clf_best = LogisticRegression(penalty='l2', C=1.1, random_state=0)
clf_best.fit(X_train, y_train)

y_pred = clf_best.predict(X_test)
y_pred_proba = clf_best.predict_proba(X_test)

eval_model(y_test, y_pred, y_pred_proba)

clf_final = clf_best
# clf_final = clf_randomsearch.best_estimator_
#multi-class model
coef = clf_final.coef_
coef_hist = coef[:, :7]
coef_realtime = coef[:, 7:]
intercept = clf_final.intercept_
print(coef_hist.shape, coef_realtime.shape, intercept.shape)
model_param = dict(zip(cols,coef.T))
model_param['intercept'] = intercept

model_param

X_test_scaled.shape, clf_final.coef_.T.shape, clf_final.intercept_.shape

# softmax, multi_class=multinomial in LR
r = np.matmul(X_test_scaled.values, clf_final.coef_.T) + clf_final.intercept_
# print(r)
r_exp = np.exp(r)
# print(r_exp.shape, np.sum(r_exp, axis=1).shape)

prob_softmax = r_exp/r_exp.sum(axis=1)[:,None]
# print(prob_softmax)
pred_label = np.argmax(prob_softmax, axis=1)
print(pred_label.shape)
print(np.mean(pred_label==0))

clf_final.classes_

print(((y_pred == 'late') * (y_test == 'early')).sum()/len(y_test))
print(((y_pred == 'early') * (y_test == 'early')).sum()/len(y_test))

df_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns, index=X_train.index)

assert all(df_train_scaled.index == train_index), 'X_train index and train_index do not match'

X_train_store_id = df.loc[train_index]['store_id'].to_frame().join(df_train_scaled)
X_train_store_id_agg = X_train_store_id[ ['store_id'] + feat_hist_agg]

print(X_train_store_id_agg.head(5))

num_stores = X_train_store_id_agg['store_id'].nunique()

df_store_level = X_train_store_id_agg.drop_duplicates(keep='first')
print('num_stores', num_stores)
assert df_store_level.shape[0] == num_stores, 'unmatched after dropping dup'


df_store_level[feat_hist_agg].shape, coef_hist.shape

#multi-class model
hist_agg_value = df_store_level[feat_hist_agg].dot(coef_hist.T)
# print(hist_agg_value.head(5))
hist_agg_value += intercept
# print(intercept)
# print(hist_agg_value.head(5))
# hist_agg_value['intercept'] = intercept
# hist_agg_value = hist_agg_value.sum(axis=1)
# hist_agg_value = hist_agg_value.to_frame()
# hist_agg_value.columns = ['value']
print(hist_agg_value)

assert all(df_store_level.index == hist_agg_value.index), 'unmatched index'

df_pre_calculated_value = df_store_level['store_id'].to_frame().join(hist_agg_value)
df_pre_calculated_value.columns = [['store_id'] + list(clf_final.classes_)]
print(df_pre_calculated_value)

df_pre_calculated_value.mean()

#multi-class model
# print(df_pre_calculated_value.tail(4))
df_pre_calculated_value.loc['mean'] = df_pre_calculated_value.mean()
df_pre_calculated_value.loc['mean', 'store_id'] = -1
print(df_pre_calculated_value.tail(4))

import notebooksalamode as mode
mode.export_csv(df_pre_calculated_value)

feat_real_time_min_max = {}
for f in feat_real_time:
  feat_real_time_min_max[f] = [X_train[f].min(), X_train[f].max()]
  print('feature:', f, '\n\tmin:', X_train[f].min(), 'max:',  X_train[f].max())
  
print(feat_real_time_min_max)

### Testing for extracted csv/model coef

valid_X_test = df.loc[test_index][['store_id'] + cols]

valid_X_test['store_id']

# check how many stores in training not in testing
# and how many stores in testing not in training
valid_X_test = df.loc[test_index][['store_id'] + cols]
pre_cal_store = df_pre_calculated_value['store_id'].unique()

tmp = valid_X_test["store_id"].unique()

print(pre_cal_store.shape, tmp.shape)

print('in pre_cal not in testing', len(set(pre_cal_store) - set(tmp)))
print('in testing not in pre_cal', len(set(tmp) - set(pre_cal_store)), 'pct', len(set(tmp) - set(pre_cal_store))/len(pre_cal_store))

valid_X_test_pre_cal = pd.merge(valid_X_test, df_pre_calculated_value, on='store_id', how='left')
assert valid_X_test_pre_cal.shape[0] == valid_X_test.shape[0], 'unmatched shape'
assert valid_X_test_pre_cal.isna().sum().any(), 'na found'

# result = valid_X_test_pre_cal['value']
# +10.64552989 * (valid_X_test_pre_cal['tip']/20000.0)
# -22.46182226 * (valid_X_test_pre_cal['flf']/19.0)
# +46.03914325 * (valid_X_test_pre_cal['pred_horizon']/8575.0)
# proba = 1/(1+np.exp(-result))
# proba
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
  close_index = X[X['pred_horizon'] < pred_horizon_thr*60].index
  df_y_pred = pd.DataFrame(data=y_pred, index=y.index, columns=['pred'])
  df_small_horizon = df_y_pred.loc[close_index, 'pred']
  # print(df_small_horizon)
  pct = df_small_horizon.value_counts()['early']/len(df_small_horizon)
  print(pct, 'are predicted as wait when pred_horizon is', pred_horizon_thr, 'min')
  return pct

#check if predict 'no wait' when time is close to food ready time
pct = []
pred_horizon_thr = [3, 5, 6, 7, 10]
for thr in pred_horizon_thr:
  pct.append(check_if_no_wait(X_test, y_test, y_pred, thr))

plt.plot(pred_horizon_thr, pct)
plt.xlabel('pred_horizon (min)')
plt.ylabel('pct of wait predicted by model')

test_sampled_index = X_test.sample(1000).index

X_test_sampled = X_test.loc[test_sampled_index].copy()
y_test_sampled = y_test.loc[test_sampled_index].copy()

res = []
ranges = range(0, 60*20, 60)
for var in ranges:
  # print(var)
  X_test_sampled['pred_horizon'] = var
  X_test_sampled_scaled = scaler.transform(X_test_sampled)
  y_pred_sampled_scaled = clf_randomsearch.predict(X_test_sampled_scaled)
  # y_pred_sampled_scaled_proba = clf_randomsearch.predict_proba(X_test_sampled_scaled)
  # eval_model(y_test_sampled, y_pred_sampled_scaled, y_pred_sampled_scaled_proba)
  res.append(y_pred_sampled_scaled)

res = np.array(res)
# print(res, res.shape)

res[res=='early'] = 1
res[res=='ontime'] = 0
res[res=='late'] = -1

# print(res, res.shape)
# unique, counts = np.unique(res, return_counts=True)
# print(unique, counts)

changing_point = []
new = []
for row in res.T:
  itemindex = np.where(row==1)[0]
  if len(itemindex) != 0:
    changing_point.append(itemindex[0])
    row[itemindex[0]] = 'x'
  new.append(row)
# print(changing_point)
new = np.array(new)
print(new)

new = np.where(new != 'x', 0, new)
new = np.where(new == 'x', 1, new)

res_mean = new.mean(axis=0)
print(res_mean.shape, res_mean)
# print(res_mean, res_mean.shape)
# res_mean = res.mean(axis=1)
# res_mean = (res == 'ontime').mean()

plt.plot(np.array(list(ranges))/60, res_mean)
# plt.plot(res[:, :10])
plt.title('Randomly sampled from testing, and artificially created pred_horizon')
plt.xlabel('pred_horizon (min)')
plt.ylabel('pct')

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
df_test_wait = df_y_pred[df_y_pred['prediction'] == 'early']
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
print('min', df.loc[wait_index]['pred_horizon'].min(), 'mean', df.loc[wait_index]['pred_horizon'].mean(), )

df_test = df.loc[test_index]
df_test.loc[df_test['label'] == 'late', "pred_horizon"]

flfs = [1.0, 1.5, 2.0]
tips = [300, 400, 500]

pred_horizon_sum = []
for flf in flfs:
  for tip in tips:
    pred_horizon_thr = (-df_pre_calculated_value[[0, 1, 2]] 
                        - (tip*model_param['tip']-feat_real_time_min_max['tip'][0])/feat_real_time_min_max['tip'][1]\
                        - (flf*model_param['flf']-feat_real_time_min_max['flf'][0])/feat_real_time_min_max['flf'][1]) \
                        / ((model_param['pred_horizon']-feat_real_time_min_max['pred_horizon'][0])/ feat_real_time_min_max['pred_horizon'][1])

    pred_horizon_thr_clean = pred_horizon_thr[(pred_horizon_thr < pred_horizon_thr.quantile(.99)) & (pred_horizon_thr > pred_horizon_thr.quantile(.01))]
    pred_horizon_thr_clean.hist(alpha=0.5)
    pred_horizon_sum.append(pred_horizon_thr_clean.mean())
    
np.array(pred_horizon_sum).mean()

print(df_pre_calculated_value)

df_tta = datasets['5_analysis_tta']
df_tta.columns = map(str.lower, df_tta.columns)
print(df_tta.head(5))

df_model = df.loc[test_index]
df_model['is_wait'] = df_model['wait_before_ready_time'] > wait_thr
print(df_model.head(5))
# df_comp = df_model.join()

assert df_tta['delivery_id'].nunique() == df_tta.shape[0], 'duplicated delivery_ids in model prediction'
assert df_model['delivery_id'].nunique() == df_model.shape[0], 'duplicated delivery_ids in tta'

assert df_model.shape[0] == y_pred.shape[0], 'shape of x_test and y_test does not match'
assert all(df_model.index == y_test.index), 'index of x_test and y_test does not match'

df_y_test = y_test.to_frame()
df_y_test.columns = ['model_is_wait']
print(df_y_test)

df_model = df_model.join(df_y_test)
print(df_model)

df_comp = pd.merge(df_model[['delivery_id', 'is_wait', 'model_is_wait', 'original_timestamp']], df_tta[['tta_is_assign', 'delivery_id','original_timestamp']], \
                   left_on='delivery_id', right_on='delivery_id', how='inner',\
                   suffixes=('_model', '_tta'))
print(df_comp[['original_timestamp_model', 'original_timestamp_tta']])

df_comp.loc[df_comp['original_timestamp_model'] < df_comp['original_timestamp_tta'], 'tta_is_assign'] = 0

print(df_comp[['tta_is_assign']].mean())

print('pct of model decision making time >= tta assign time', (df_comp['original_timestamp_model'] >= df_comp['original_timestamp_tta']).mean())


num_total = df_comp.shape[0]
print('total num of data points', num_total)
num_stopped_by_model_missed_by_tta = df_comp[(df_comp['is_wait'] == True) & (df_comp['model_is_wait']==True)& (df_comp['tta_is_assign']==1)].shape[0]
print('correctly stopped by model, but not by tta', num_stopped_by_model_missed_by_tta, '\npct of total',num_stopped_by_model_missed_by_tta/num_total)

num_stopped_by_tta_missed_by_model = df_comp[(df_comp['is_wait'] == True) & (df_comp['model_is_wait']==False)& (df_comp['tta_is_assign']==0)].shape[0]
print('correctly stopped by tta, but not by model', num_stopped_by_tta_missed_by_model, '\npct of total',num_stopped_by_tta_missed_by_model/num_total)

1-df_comp['tta_is_assign'].mean(), df_comp['model_is_wait'].mean(), df_comp['is_wait'].mean()



