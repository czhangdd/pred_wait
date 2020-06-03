# Python Notebook - is_wait_prediction_v5

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

wait_thr = 5 * 60

datasets['3_feat_v2_more_hist'].shape, datasets['4_feat_filter'].shape, datasets['6_wait_geo_table']

# df = datasets['4_feat_filter']
df = datasets['6_wait_geo_table']
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

## Outlier removal

print('before remove pred_horizon<0, shape', df.shape)
df = df[df['pred_horizon'] >= 0]
print('after remove pred_horizon<0, shape', df.shape)

feat_hist_agg = ['wait_before_ready_time_mean', 'd2r_duration_mean', 'wait_before_ready_time_max',\
                 'd2r_duration_min', 'pred_horizon_mean', 'wait_before_ready_time_min' , 'avg_num_assigns']
# feat_real_time = ['tip', 'flf', 'pred_horizon']
feat_real_time = ['tip', 'flf', 'pred_horizon', 'num_nearby_idle', 'num_nearby_busy']
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
y_train = df.loc[train_index, 'wait_before_ready_time'] > wait_thr

X_test = df.loc[test_index, cols]
y_test = df.loc[test_index, 'wait_before_ready_time'] >= wait_thr


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = Normalizer()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

for i in range(X_train_scaled.shape[1]):
  print(X_train_scaled[:, i].max())

# assert all(X_train_store_id_agg[feat_hist_agg].max() > 0.999), 'check if all scaled features max == 1'

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

coef = clf.coef_[0]
coef_hist = coef[:7]
coef_realtime = coef[7:]
intercept = clf.intercept_[0]
print(coef_hist.shape, coef_realtime.shape)
model_param = dict(zip(cols,coef))
model_param['intercept'] = intercept

model_param

df_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns, index=X_train.index)

assert all(df_train_scaled.index == train_index), 'X_train index and train_index do not match'

X_train_store_id = df.loc[train_index]['store_id'].to_frame().join(df_train_scaled)
X_train_store_id_agg = X_train_store_id[ ['store_id'] + feat_hist_agg]

print(X_train_store_id_agg.head(5))

num_stores = X_train_store_id_agg['store_id'].nunique()

df_store_level = X_train_store_id_agg.drop_duplicates(keep='first')
print('num_stores', num_stores)
assert df_store_level.shape[0] == num_stores, 'unmatched after dropping dup'


# coef = list(clf.coef_[0])[:7]
# print(coef)

# intercept = clf.intercept_[0]
# print(intercept)

hist_agg_value = df_store_level[feat_hist_agg] * coef_hist
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
assert valid_X_test_pre_cal['value'].isna().sum() == 0, 'na found'

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
pred_horizon_thr = [240, 300, 360, 480, 600, 720, 840, 960]
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
print('min', df.loc[wait_index]['pred_horizon'].min(), 'mean', df.loc[wait_index]['pred_horizon'].mean(), )

flfs = [1.0, 1.5, 2.0]
tips = [300, 400, 500]

pred_horizon_sum = []
for flf in flfs:
  for tip in tips:
    pred_horizon_thr = (-df_pre_calculated_value['value'] 
                        - (tip*feat_coef['tip']-feat_real_time_min_max['tip'][0])/feat_real_time_min_max['tip'][1]\
                        - (flf*feat_coef['flf']-feat_real_time_min_max['flf'][0])/feat_real_time_min_max['flf'][1]) \
                        / ((feat_coef['pred_horizon']-feat_real_time_min_max['pred_horizon'][0])/ feat_real_time_min_max['pred_horizon'][1])

    pred_horizon_thr_clean = pred_horizon_thr[(pred_horizon_thr < pred_horizon_thr.quantile(.99)) & (pred_horizon_thr > pred_horizon_thr.quantile(.01))]
    pred_horizon_thr_clean.hist(alpha=0.5)
    pred_horizon_sum.append(pred_horizon_thr_clean.mean())
    
np.array(pred_horizon_sum).mean()

df_pre_calculated_value['value']

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
df_stopped_by_model_missed_by_tta = df_comp[(df_comp['is_wait'] == True) & (df_comp['model_is_wait']==True)& (df_comp['tta_is_assign']==1)]
num_stopped_by_model_missed_by_tta = df_stopped_by_model_missed_by_tta.shape[0]
print('correctly stopped by model, but not by tta', num_stopped_by_model_missed_by_tta, '\npct of total',num_stopped_by_model_missed_by_tta/num_total)

num_stopped_by_tta_missed_by_model = df_comp[(df_comp['is_wait'] == True) & (df_comp['model_is_wait']==False)& (df_comp['tta_is_assign']==0)].shape[0]
print('correctly stopped by tta, but not by model', num_stopped_by_tta_missed_by_model, '\npct of total',num_stopped_by_tta_missed_by_model/num_total)

1-df_comp['tta_is_assign'].mean(), df_comp['model_is_wait'].mean(), df_comp['is_wait'].mean()

df

df_stopped_by_model_missed_by_tta_pred_horizon = df.loc[df_stopped_by_model_missed_by_tta.index]['pred_horizon']
print('mean of long wait deli stopped by model missed by tta', df_stopped_by_model_missed_by_tta_pred_horizon.mean())
df_stopped_by_model_missed_by_tta_pred_horizon.hist()
plt.title('long wait deliveries stopped by model missed by tta')
plt.xlabel('pred_horizon (sec)')
plt.ylabel('freq')



